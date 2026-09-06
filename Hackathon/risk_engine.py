# risk_engine.py
"""
Interpretable PAIMANA risk calculation.

The ML model detects unusual project patterns.
This rule-based layer explains concrete reasons for risk.

The final score is a weighted combination of:
    1. ML anomaly score
    2. schedule risk
    3. progress/spending imbalance
    4. cost risk
    5. revision/schedule signals
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _clip(value: float, low: float = 0, high: float = 100) -> float:
    return max(low, min(high, float(value)))


def calculate_project_risk(
    row: pd.Series,
    ml_anomaly_score: float,
    snapshot_date: pd.Timestamp,
) -> dict[str, Any]:
    """Calculate a 0-100 risk score and human-readable risk factors."""

    progress = float(row.get("Physical Progress (in %)", 0) or 0)
    original_cost = float(row.get("Original Cost (in cr.)", 0) or 0)
    expenditure = float(row.get("Expenditure (in cr.)", 0) or 0)

    revised_cost = row.get("Revised Cost (in cr.)")
    revised_cost = (
        float(revised_cost) if pd.notna(revised_cost) else None
    )

    original_date = row.get("Original Date of Commissioning")
    revised_date = row.get("Revised Date of Commissioning")

    factors: list[str] = []
    score_components: list[tuple[float, float]] = []

    # ---------------------------------------------------------
    # 1. ML anomaly component: 30%
    # ---------------------------------------------------------
    ml_score = _clip(ml_anomaly_score)
    score_components.append((ml_score, 0.30))

    if ml_score >= 75:
        factors.append("Project pattern is highly unusual compared with the portfolio.")
    elif ml_score >= 55:
        factors.append("Project pattern is more unusual than most projects in the portfolio.")

    # ---------------------------------------------------------
    # 2. Schedule component: 30%
    # ---------------------------------------------------------
    schedule_score = 0.0

    if pd.notna(original_date):
        days_past_original = max(
            0, (snapshot_date - pd.Timestamp(original_date)).days
        )

        if days_past_original > 365 and progress < 90:
            schedule_score = 100
            factors.append("Original commissioning date has passed by more than one year.")
        elif days_past_original > 180 and progress < 90:
            schedule_score = 85
            factors.append("Original commissioning date has passed by more than six months.")
        elif days_past_original > 0 and progress < 90:
            schedule_score = 65
            factors.append("Original commissioning date has already passed.")

    if pd.notna(revised_date):
        revised_date = pd.Timestamp(revised_date)

        if revised_date < snapshot_date and progress < 95:
            schedule_score = max(schedule_score, 95)
            factors.append("Revised commissioning date has passed while the project is below 95% progress.")

        extension_days = (
            revised_date - pd.Timestamp(original_date)
        ).days if pd.notna(original_date) else 0

        if extension_days > 730:
            schedule_score = max(schedule_score, 90)
            factors.append("Commissioning schedule has been extended by more than two years.")
        elif extension_days > 365:
            schedule_score = max(schedule_score, 75)
            factors.append("Commissioning schedule has been extended by more than one year.")
        elif extension_days > 180:
            schedule_score = max(schedule_score, 55)
            factors.append("Commissioning schedule has been extended by more than six months.")
    else:
        # Missing revised date is NOT automatically treated as high risk.
        # It simply means there is no revised schedule to evaluate.
        pass

    score_components.append((_clip(schedule_score), 0.30))

    # ---------------------------------------------------------
    # 3. Spending vs progress component: 20%
    # ---------------------------------------------------------
    effective_cost = revised_cost if revised_cost and revised_cost > 0 else original_cost

    expenditure_ratio = (
        expenditure / effective_cost * 100
        if effective_cost > 0
        else 0
    )

    gap = expenditure_ratio - progress

    if gap >= 40:
        spend_score = 100
        factors.append("Expenditure is substantially ahead of physical progress.")
    elif gap >= 25:
        spend_score = 80
        factors.append("Expenditure is significantly ahead of physical progress.")
    elif gap >= 15:
        spend_score = 60
        factors.append("Expenditure is moderately ahead of physical progress.")
    elif gap >= 5:
        spend_score = 35
    else:
        spend_score = 10

    score_components.append((_clip(spend_score), 0.20))

    # ---------------------------------------------------------
    # 4. Cost component: 10%
    # ---------------------------------------------------------
    cost_score = 0.0

    if revised_cost is not None and original_cost > 0:
        cost_overrun = (revised_cost - original_cost) / original_cost * 100

        if cost_overrun >= 30:
            cost_score = 100
            factors.append(f"Project cost has increased by {cost_overrun:.1f}%.")
        elif cost_overrun >= 20:
            cost_score = 80
            factors.append(f"Project cost has increased by {cost_overrun:.1f}%.")
        elif cost_overrun >= 10:
            cost_score = 60
            factors.append(f"Project cost has increased by {cost_overrun:.1f}%.")
        elif cost_overrun > 0:
            cost_score = 30

    # Missing revised cost is neutral, not zero-risk and not high-risk.
    score_components.append((_clip(cost_score), 0.10))

    # ---------------------------------------------------------
    # 5. Progress component: 10%
    # ---------------------------------------------------------
    progress_score = 0.0

    if pd.notna(original_date):
        planned_days = max(
            1,
            (pd.Timestamp(original_date) - pd.Timestamp(row["Sanction Date"])).days
            if pd.notna(row.get("Sanction Date"))
            else 1,
        )
        elapsed_days = max(
            0,
            (snapshot_date - pd.Timestamp(row["Sanction Date"])).days
            if pd.notna(row.get("Sanction Date"))
            else 0,
        )
        elapsed_ratio = min(1.5, elapsed_days / planned_days)

        # If a large portion of planned time has elapsed but progress is low.
        expected_progress = min(100, elapsed_ratio * 100)

        if expected_progress > 20:
            progress_gap = expected_progress - progress
            progress_score = _clip(progress_gap * 1.5)

            if progress_gap >= 30:
                factors.append("Physical progress is substantially behind the planned timeline.")

    score_components.append((_clip(progress_score), 0.10))

    # Weighted final score.
    final_score = sum(component * weight for component, weight in score_components)
    final_score = round(_clip(final_score), 1)

    if final_score >= 70:
        level = "HIGH"
    elif final_score >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    # Avoid an overly repetitive UI.
    factors = list(dict.fromkeys(factors))[:6]

    if not factors:
        factors.append("No major risk signal was detected by the current rules.")

    return {
        "risk_score": final_score,
        "risk_level": level,
        "ml_anomaly_score": round(ml_score, 1),
        "risk_factors": factors,
        "metrics": {
            "physical_progress": round(progress, 2),
            "expenditure_ratio": round(expenditure_ratio, 2),
            "spend_progress_gap": round(gap, 2),
        },
    }
