import streamlit as st
import requests
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PAIMANA Project Intelligence",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = "http://127.0.0.1:8000"

MONTHS = {
    "January": "01",
    "February": "02",
    "March": "03",
    "April": "04",
    "May": "05",
    "June": "06",
    "July": "07",
    "August": "08",
    "September": "09",
    "October": "10",
    "November": "11",
    "December": "12",
}

YEARS = list(range(2020, 2027))


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# API FUNCTIONS
# ============================================================

def start_analysis(month, year):

    response = requests.post(
        f"{API_BASE_URL}/api/analyze",
        json={
            "month": month,
            "year": year,
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def get_analysis(job_id):

    response = requests.get(
        f"{API_BASE_URL}/api/analyze/{job_id}",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# HEADER
# ============================================================

st.title("📊 PAIMANA Project Intelligence")

st.caption(
    "Infrastructure Project Monitoring • ML Anomaly Detection • "
    "Explainable Risk Assessment"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Analysis Controls")

    selected_month_name = st.selectbox(
        "Select Month",
        list(MONTHS.keys()),
        index=3,  # April
    )

    selected_year = st.selectbox(
        "Select Year",
        YEARS,
        index=YEARS.index(2026),
    )

    selected_month = MONTHS[selected_month_name]

    st.write(
        f"**Selected Period:** "
        f"{selected_month_name} {selected_year}"
    )

    run_analysis_button = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.caption("Local SIH Demo")
    st.caption("Streamlit → FastAPI → PAIMANA")


# ============================================================
# START ANALYSIS
# ============================================================

if run_analysis_button:

    try:

        with st.status(
            "Running project analysis...",
            expanded=True,
        ) as status:

            st.write(
                f"Starting analysis for "
                f"{selected_month_name} {selected_year}..."
            )

            job = start_analysis(
                selected_month,
                selected_year,
            )

            job_id = job["job_id"]

            st.write(
                f"Job ID: `{job_id}`"
            )

            # ------------------------------------------------
            # POLLING
            # ------------------------------------------------

            while True:

                current = get_analysis(job_id)

                current_status = current.get("status")
                stage = current.get("stage")

                if stage:
                    st.write(stage)

                # ----------------------------
                # COMPLETED
                # ----------------------------

                if current_status == "completed":

                    result = current.get("result")

                    if result is None:
                        raise RuntimeError(
                            "Backend completed the analysis "
                            "but returned no result."
                        )

                    st.session_state.result = result

                    status.update(
                        label="Analysis completed successfully.",
                        state="complete",
                        expanded=False,
                    )

                    st.rerun()

                # ----------------------------
                # FAILED
                # ----------------------------

                if current_status == "failed":

                    error = current.get(
                        "error",
                        "Unknown backend error.",
                    )

                    status.update(
                        label="Analysis failed.",
                        state="error",
                    )

                    st.error(error)

                    break

                # ----------------------------
                # STILL RUNNING
                # ----------------------------

                time.sleep(1.5)

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the FastAPI backend.\n\n"
            "Make sure this is running:\n\n"
            "`python -m uvicorn main:app --reload --port 8000`"
        )

    except requests.exceptions.Timeout:

        st.error(
            "❌ Backend request timed out."
        )

    except requests.exceptions.HTTPError as error:

        st.error(
            f"❌ Backend HTTP error: {error}"
        )

    except Exception as error:

        st.error(
            f"❌ Unexpected error: {error}"
        )


# ============================================================
# DASHBOARD
# ============================================================

result = st.session_state.result


if result is None:

    st.info(
        "Select a month and year from the sidebar, "
        "then click **Run Analysis**."
    )

    st.subheader("System Workflow")

    st.markdown(
        """
        ```text
        PAIMANA
            ↓
        Data Processing
            ↓
        Feature Engineering
            ↓
        ML Anomaly Detection
            ↓
        Risk Engine
            ↓
        FastAPI
            ↓
        Streamlit Dashboard
        ```
        """
    )

else:

    period = result.get("period", {})
    summary = result.get("summary", {})
    projects = result.get("projects", [])

    month_number = period.get("month")
    year = period.get("year")

    month_name = next(
        (
            name
            for name, number in MONTHS.items()
            if number == month_number
        ),
        month_number,
    )

    # ========================================================
    # TITLE
    # ========================================================

    st.subheader(
        f"Project Risk Overview — {month_name} {year}"
    )

    # ========================================================
    # SUMMARY CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Projects",
        summary.get("total_projects", 0),
    )

    col2.metric(
        "🔴 High Risk",
        summary.get("high_risk", 0),
    )

    col3.metric(
        "🟠 Medium Risk",
        summary.get("medium_risk", 0),
    )

    col4.metric(
        "🟢 Low Risk",
        summary.get("low_risk", 0),
    )

    st.divider()

    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    st.subheader("Risk Distribution")

    chart_data = {
        "High Risk": summary.get("high_risk", 0),
        "Medium Risk": summary.get("medium_risk", 0),
        "Low Risk": summary.get("low_risk", 0),
    }

    st.bar_chart(chart_data)

    st.divider()

    # ========================================================
    # PROJECT FILTERS
    # ========================================================

    st.subheader("Project Intelligence")

    col1, col2 = st.columns(2)

    with col1:

        risk_filter = st.selectbox(
            "Filter by Risk",
            [
                "ALL",
                "HIGH",
                "MEDIUM",
                "LOW",
            ],
        )

    with col2:

        search = st.text_input(
            "Search Project",
            placeholder="Enter project name or code...",
        ).lower()

    filtered_projects = projects

    # Risk filter

    if risk_filter != "ALL":

        filtered_projects = [
            project
            for project in filtered_projects
            if str(
                project.get("risk_level", "")
            ).upper() == risk_filter
        ]

    # Search filter

    if search:

        filtered_projects = [
            project
            for project in filtered_projects
            if (
                search
                in str(
                    project.get(
                        "project_name",
                        "",
                    )
                ).lower()
            )
            or (
                search
                in str(
                    project.get(
                        "project_code",
                        "",
                    )
                ).lower()
            )
        ]

    st.caption(
        f"Showing {len(filtered_projects):,} "
        f"of {len(projects):,} projects"
    )

    # ========================================================
    # PROJECT TABLE
    # ========================================================

    if filtered_projects:

        table_data = []

        for project in filtered_projects:

            metrics = project.get(
                "metrics",
                {},
            )

            level = str(
                project.get(
                    "risk_level",
                    "",
                )
            ).upper()

            if level == "HIGH":
                icon = "🔴"

            elif level == "MEDIUM":
                icon = "🟠"

            else:
                icon = "🟢"

            table_data.append(
                {
                    "Risk": icon,
                    "Score": project.get(
                        "risk_score",
                        0,
                    ),
                    "Project": project.get(
                        "project_name",
                        "",
                    ),
                    "Code": project.get(
                        "project_code",
                        "",
                    ),
                    "Sector": project.get(
                        "sector",
                        "",
                    ),
                    "Physical Progress": (
                        metrics.get(
                            "physical_progress",
                            0,
                        )
                    ),
                    "Expenditure Ratio": (
                        metrics.get(
                            "expenditure_ratio",
                            0,
                        )
                    ),
                }
            )

        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.warning(
            "No projects match the selected filters."
        )

    st.divider()

    # ========================================================
    # PROJECT DETAILS
    # ========================================================

    st.subheader("Project Risk Details")

    if filtered_projects:

        project_options = [
            (
                f"{project.get('project_code', '')} — "
                f"{project.get('project_name', '')}"
            )
            for project in filtered_projects
        ]

        selected_project = st.selectbox(
            "Select Project",
            project_options,
        )

        selected_index = project_options.index(
            selected_project
        )

        project = filtered_projects[selected_index]

        metrics = project.get(
            "metrics",
            {},
        )

        level = str(
            project.get(
                "risk_level",
                "UNKNOWN",
            )
        ).upper()

        score = float(
            project.get(
                "risk_score",
                0,
            )
        )

        # ----------------------------------------------------
        # PROJECT HEADER
        # ----------------------------------------------------

        st.markdown(
            f"## {project.get('project_name', 'Unknown Project')}"
        )

        if level == "HIGH":

            st.error(
                f"🔴 HIGH RISK — Score: {score:.1f}/100"
            )

        elif level == "MEDIUM":

            st.warning(
                f"🟠 MEDIUM RISK — Score: {score:.1f}/100"
            )

        else:

            st.success(
                f"🟢 LOW RISK — Score: {score:.1f}/100"
            )

        # ----------------------------------------------------
        # PROJECT INFORMATION
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Project Code:** "
                f"{project.get('project_code', 'N/A')}"
            )

            st.write(
                f"**Sector:** "
                f"{project.get('sector', 'N/A')}"
            )

            st.write(
                f"**Ministry:** "
                f"{project.get('ministry', 'N/A')}"
            )

        with col2:

            st.write(
                f"**Implementing Agency:** "
                f"{project.get('implementing_agency', 'N/A')}"
            )

            st.write(
                f"**ML Anomaly Score:** "
                f"{float(project.get('ml_anomaly_score', 0)):.1f}/100"
            )

        st.divider()

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Risk Score",
            f"{score:.1f}/100",
        )

        col2.metric(
            "Physical Progress",
            f"{float(metrics.get('physical_progress', 0)):.2f}%",
        )

        col3.metric(
            "Expenditure Ratio",
            f"{float(metrics.get('expenditure_ratio', 0)):.2f}%",
        )

        st.progress(
            min(max(score / 100, 0), 1),
            text=f"Risk Score: {score:.1f}/100",
        )

        # ----------------------------------------------------
        # RISK FACTORS
        # ----------------------------------------------------

        st.subheader("⚠️ Risk Factors")

        factors = project.get(
            "risk_factors",
            [],
        )

        if factors:

            for factor in factors:

                st.warning(
                    factor
                )

        else:

            st.success(
                "No major risk factors detected."
            )

        # ----------------------------------------------------
        # SPENDING VS PROGRESS
        # ----------------------------------------------------

        st.subheader(
            "📈 Spending vs Physical Progress"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Physical Progress",
            f"{float(metrics.get('physical_progress', 0)):.2f}%",
        )

        col2.metric(
            "Expenditure Ratio",
            f"{float(metrics.get('expenditure_ratio', 0)):.2f}%",
        )

        col3.metric(
            "Spend / Progress Gap",
            f"{float(metrics.get('spend_progress_gap', 0)):.2f}%",
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PAIMANA Project Intelligence System • "
    "Risk scores are generated by the FastAPI backend."
)
