from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from uuid import uuid4
from datetime import datetime

from scrapper import scrape_paimana


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="PAIMANA Infrastructure Intelligence API",
    version="1.0.0"
)


# ============================================================
# TEMPORARY JOB STORAGE
# ============================================================

jobs = {}


# ============================================================
# REQUEST MODEL
# ============================================================

class AnalysisRequest(BaseModel):

    month: str
    year: int


# ============================================================
# BACKGROUND ANALYSIS
# ============================================================

def run_analysis(job_id, month, year):

    try:

        jobs[job_id]["status"] = "scraping"
        jobs[job_id]["stage"] = "Fetching PAIMANA data"

        df = scrape_paimana(
            month,
            year
        )

        # ----------------------------------------------------
        # ML PIPELINE WILL GO HERE
        # ----------------------------------------------------

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["stage"] = "Analysis complete"

        jobs[job_id]["result"] = {
            "period": f"{month}/{year}",
            "total_projects": len(df),

            # Temporary values.
            # REMOVE when the actual ML brain exists.
            "high_risk_projects": 0,
            "medium_risk_projects": 0,
            "low_risk_projects": 0,
            "average_risk_score": 0
        }

    except Exception as e:

        jobs[job_id]["status"] = "failed"
        jobs[job_id]["stage"] = "Analysis failed"
        jobs[job_id]["error"] = str(e)


# ============================================================
# START ANALYSIS
# ============================================================

@app.post("/api/analyze")
def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):

    # --------------------------------------------------------
    # Validate month
    # --------------------------------------------------------

    try:

        month = int(request.month)

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="Month must be numeric."
        )

    if month < 1 or month > 12:

        raise HTTPException(
            status_code=400,
            detail="Month must be between 01 and 12."
        )

    month = str(month).zfill(2)

    # --------------------------------------------------------
    # Validate year
    # --------------------------------------------------------

    if request.year < 2000 or request.year > 2100:

        raise HTTPException(
            status_code=400,
            detail="Invalid year."
        )

    # --------------------------------------------------------
    # Create job
    # --------------------------------------------------------

    job_id = str(uuid4())

    jobs[job_id] = {

        "status": "queued",

        "stage": "Waiting to start",

        "created_at": datetime.now().isoformat(),

        "result": None,

        "error": None
    }

    # --------------------------------------------------------
    # Start background job
    # --------------------------------------------------------

    background_tasks.add_task(
        run_analysis,
        job_id,
        month,
        request.year
    )

    # --------------------------------------------------------
    # Return immediately
    # --------------------------------------------------------

    return {

        "job_id": job_id,

        "status": "queued",

        "period": {
            "month": month,
            "year": request.year
        }
    }


# ============================================================
# JOB STATUS
# ============================================================

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):

    if job_id not in jobs:

        raise HTTPException(
            status_code=404,
            detail="Job not found."
        )

    return jobs[job_id]