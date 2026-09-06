# data_pipeline.py
"""
PAIMANA data cleaning + feature engineering.

The pipeline is designed around the PAIMANA Project Overview CSV schema.
It deliberately preserves missing revised cost/date information instead
of pretending that the information exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


PAIMANA_COLUMNS = [
    "Sr. No.",
    "Sector Name",
    "Line Ministry",
    "Implementing Agency",
    "Project Code",
    "Project Name",
    "Original Cost (in cr.)",
    "Revised Cost (in cr.)",
    "Expenditure (in cr.)",
    "Physical Progress (in %)",
    "Original Date of Commissioning",
    "Revised Date of Commissioning",
    "Sanction Date",
]

NUMERIC_COLUMNS = [
    "Original Cost (in cr.)",
    "Revised Cost (in cr.)",
    "Expenditure (in cr.)",
    "Physical Progress (in %)",
]

DATE_COLUMNS = [
    "Original Date of Commissioning",
    "Revised Date of Commissioning",
    "Sanction Date",
]


def load_paimana_csv(path: str | Path) -> pd.DataFrame:
    """Load one PAIMANA CSV and return a cleaned raw dataframe."""
    path = Path(path)
    df = pd.read_csv(path)

    # PAIMANA exports can contain a repeated header row.
    if "Project Code" in df.columns:
        df = df[df["Project Code"].astype(str).str.strip().ne("Project Code")]

    # Make sure the expected schema exists.
    missing = [c for c in PAIMANA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing PAIMANA columns: {missing}")

    return clean_paimana_data(df[PAIMANA_COLUMNS].copy())


def clean_paimana_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean values without destroying useful missingness.

    Revised Cost = 0 is treated as "no revised cost recorded".
    Revised Date missing is treated as "no revised date recorded".
    We keep indicator columns so the model/risk engine can distinguish
    'not recorded' from a genuine value.
    """
    df = df.copy()

    # Remove accidental repeated header rows and completely empty rows.
    df = df[df["Project Code"].astype(str).str.strip().ne("Project Code")]
    df = df.dropna(how="all").reset_index(drop=True)

    # Numeric conversion: invalid text becomes NaN instead of crashing.
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # PAIMANA uses 0 in Revised Cost when no revision is recorded.
    # Keep the information in an indicator before converting 0 to missing.
    df["has_cost_revision"] = (
        df["Revised Cost (in cr.)"].fillna(0) > 0
    ).astype(int)

    df.loc[df["Revised Cost (in cr.)"] <= 0, "Revised Cost (in cr.)"] = np.nan

    # Date conversion. Invalid/missing dates become NaT.
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce",
            dayfirst=True,
        )

    df["has_schedule_revision"] = (
        df["Revised Date of Commissioning"].notna()
    ).astype(int)

    # Progress is constrained to a sensible range for safety.
    df["Physical Progress (in %)"] = (
        df["Physical Progress (in %)"].clip(0, 100)
    )

    return df


def create_features(
    df: pd.DataFrame,
    snapshot_date: Optional[str | pd.Timestamp] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create numeric ML features.

    Returns:
        cleaned_df: cleaned project data
        X: model-ready numeric feature dataframe
    """
    cleaned = clean_paimana_data(df)

    # Use the requested month/year date when supplied.
    # This is much safer than silently using the machine's current date.
    if snapshot_date is not None:
        snapshot = pd.Timestamp(snapshot_date)
    else:
        snapshot = None

    f = pd.DataFrame(index=cleaned.index)

    original_cost = cleaned["Original Cost (in cr.)"]
    revised_cost = cleaned["Revised Cost (in cr.)"]
    expenditure = cleaned["Expenditure (in cr.)"]
    progress = cleaned["Physical Progress (in %)"]

    # If there is no revised cost, original cost is the best known
    # denominator for expenditure calculations.
    effective_cost = revised_cost.fillna(original_cost).replace(0, np.nan)

    f["original_cost"] = original_cost
    f["revised_cost"] = revised_cost
    f["expenditure"] = expenditure
    f["physical_progress"] = progress
    f["has_cost_revision"] = cleaned["has_cost_revision"]
    f["has_schedule_revision"] = cleaned["has_schedule_revision"]

    # Cost overrun is only meaningful when a revised cost exists.
    f["cost_overrun_pct"] = (
        (revised_cost - original_cost) / original_cost.replace(0, np.nan) * 100
    )

    # Expenditure relative to the best available cost.
    f["expenditure_ratio"] = expenditure / effective_cost * 100

    # A rough "spend vs physical progress" imbalance.
    # Example: 80% money spent but only 40% physical progress => +40.
    f["spend_progress_gap"] = f["expenditure_ratio"] - progress

    original_date = cleaned["Original Date of Commissioning"]
    revised_date = cleaned["Revised Date of Commissioning"]
    sanction_date = cleaned["Sanction Date"]

    # Planned project duration from sanction to original commissioning.
    f["planned_duration_days"] = (
        original_date - sanction_date
    ).dt.days

    # Schedule extension only exists when a revised date exists.
    f["schedule_extension_days"] = (
        revised_date - original_date
    ).dt.days

    # If we know the snapshot date, calculate whether the original/revised
    # completion date has already passed.
    if snapshot is not None:
        f["days_past_original_date"] = (
            snapshot - original_date
        ).dt.days.clip(lower=0)

        f["days_until_revised_date"] = (
            revised_date - snapshot
        ).dt.days

        f["days_past_revised_date"] = (
            snapshot - revised_date
        ).dt.days.clip(lower=0)
    else:
        f["days_past_original_date"] = 0.0
        f["days_until_revised_date"] = 0.0
        f["days_past_revised_date"] = 0.0

    # Model-ready cleanup. Median filling is learned later from the
    # training data, so here we only replace infinities.
    X = f.replace([np.inf, -np.inf], np.nan)

    return cleaned, X
