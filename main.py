# main.py
"""
FastAPI backend for PAIMANA.

This backend:
    1. Accepts month/year from the frontend.
    2. Gets a matching cached CSV when available.
    3. Otherwise calls scraper.scrape_paimana(month, year).
    4. Cleans/features the data.
    5. Runs the trained ML anomaly model.
    6. Runs the interpretable risk engine.
    7. Returns JSON to the frontend.

The exact frontend API can be adjusted without changing the ML code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_pipeline import create_features, load_paimana_csv
from ml_pipeline import PaimanaMLModel
from risk_engine import calculate_project_risk


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(
    title="PAIMANA Infrastructure Risk API",
    version="1.0.0",
)

# For development/demo. Tighten this to your frontend's domain before
# production if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    month: str = Field(pattern=r"^(0[1-9]|1[0-2])$")
    year: int = Field(ge=2000, le=2100)


jobs: dict[str, dict[str, Any]] = {}


def get_cached_csv(month: str, year: int) -> Path | None:
    """
    Look for a previously downloaded dataset.

    Example:
        project_overview_04_2026.csv
    """
    candidates = [
        DATA_DIR / f"project_overview_{month}_{year}.csv",
        BASE_DIR.parent / "paimana_output" / f"project_overview_{month}_{year}.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def run_analysis(job_id: str, month: str, year: int) -> None:
    """Background job: scrape/load -> ML -> risk -> results."""
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["stage"] = "Fetching project intelligence..."

        csv_path = get_cached_csv(month, year)

        if csv_path is None:
            # Import only when needed so the API can still start if Selenium
            # is not installed in a frontend-only environment.
            from scrapper import scrape_paimana

            df = scrape_paimana(month, year)

            # Keep a cache so a temporary PAIMANA/network outage does not
            # necessarily destroy a later demo.
            cache_path = DATA_DIR / f"project_overview_{month}_{year}.csv"
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
            csv_path = cache_path

        jobs[job_id]["stage"] = "Preparing project features..."
        raw_df = load_paimana_csv(csv_path)

        snapshot_date = pd.Timestamp(year=year, month=int(month), day=1)
        cleaned_df, X = create_features(
            raw_df,
            snapshot_date=snapshot_date,
        )

        jobs[job_id]["stage"] = "Evaluating project risk..."

        try:
            model = PaimanaMLModel.load()
        except FileNotFoundError:
            raise RuntimeError(
                "ML model is not trained. Run train_model.py before starting the API."
            )

        ml_scores = model.score(X)

        projects = []

        for position, (_, row) in enumerate(cleaned_df.iterrows()):
            risk = calculate_project_risk(
                row=row,
                ml_anomaly_score=float(ml_scores[position]),
                snapshot_date=snapshot_date,
            )

            projects.append(
                {
                    "project_code": str(row["Project Code"]),
                    "project_name": str(row["Project Name"]),
                    "sector": str(row["Sector Name"]),
                    "ministry": str(row["Line Ministry"]),
                    "implementing_agency": str(row["Implementing Agency"]),
                    **risk,
                }
            )

        # Highest risk first.
        projects.sort(key=lambda item: item["risk_score"], reverse=True)

        high = sum(p["risk_level"] == "HIGH" for p in projects)
        medium = sum(p["risk_level"] == "MEDIUM" for p in projects)
        low = sum(p["risk_level"] == "LOW" for p in projects)

        jobs[job_id]["stage"] = "Preparing project analysis..."
        jobs[job_id]["result"] = {
            "period": {
                "month": month,
                "year": year,
            },
            "summary": {
                "total_projects": len(projects),
                "high_risk": high,
                "medium_risk": medium,
                "low_risk": low,
            },
            "projects": projects,
        }

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["stage"] = "Analysis complete."

    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["stage"] = "Analysis failed."
        jobs[job_id]["error"] = str(exc)


@app.get("/")
def root():
    return {
        "service": "PAIMANA Infrastructure Risk API",
        "status": "online",
    }


@app.post("/api/analyze")
def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid4())

    jobs[job_id] = {
        "status": "queued",
        "stage": "Starting analysis...",
        "period": {
            "month": request.month,
            "year": request.year,
        },
    }

    background_tasks.add_task(
        run_analysis,
        job_id,
        request.month,
        request.year,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "period": {
            "month": request.month,
            "year": request.year,
        },
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return job

@app.get("/api/analyze/{job_id}")
def get_analysis(job_id: str):
    job = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    return job

@app.get("/api/projects")
def get_projects(month: str = "04", year: int = 2026):
    """Convenience endpoint for the dashboard."""
    csv_path = get_cached_csv(month, year)

    if csv_path is None:
        raise HTTPException(
            status_code=404,
            detail="Dataset not cached. Start /api/analyze first.",
        )

    raw_df = load_paimana_csv(csv_path)
    snapshot_date = pd.Timestamp(year=year, month=int(month), day=1)
    cleaned_df, X = create_features(raw_df, snapshot_date=snapshot_date)

    model = PaimanaMLModel.load()
    scores = model.score(X)

    projects = []
    for position, (_, row) in enumerate(cleaned_df.iterrows()):
        projects.append({
            "project_code": str(row["Project Code"]),
            "project_name": str(row["Project Name"]),
            "sector": str(row["Sector Name"]),
            "risk": calculate_project_risk(
                row,
                float(scores[position]),
                snapshot_date,
            ),
        })

    projects.sort(key=lambda x: x["risk"]["risk_score"], reverse=True)
    return projects


@app.get("/api/summary")
def get_summary(month: str = "04", year: int = 2026):
    projects = get_projects(month, year)

    return {
        "total_projects": len(projects),
        "high_risk": sum(p["risk"]["risk_level"] == "HIGH" for p in projects),
        "medium_risk": sum(p["risk"]["risk_level"] == "MEDIUM" for p in projects),
        "low_risk": sum(p["risk"]["risk_level"] == "LOW" for p in projects),
    }
