# scraper.py

from pathlib import Path
import time

import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# ============================================================
# CONFIG
# ============================================================

URL = "https://paimana-proj.mospi.gov.in/Home/PublicDashboardNew"


# ------------------------------------------------------------
# PATH VARIABLES
# ------------------------------------------------------------
#
# Path is a Python class for working with file/folder paths.
#
# Path.cwd() means:
#     "Current Working Directory"
#
# For example, if Python is running from:
#
#     C:/PythonProject
#
# then:
#
#     Path.cwd()
#
# represents that folder.
#
# We can then use "/" to safely build another path:
#
#     Path.cwd() / "paimana_output"
#
# This is preferable to manually writing:
#
#     "C:/PythonProject/paimana_output"
#
# because Path handles operating-system path differences.
# ------------------------------------------------------------

DOWNLOAD_DIR = Path.cwd() / "paimana_downloads"
OUTPUT_DIR = Path.cwd() / "paimana_output"


# ------------------------------------------------------------
# Create directories
# ------------------------------------------------------------
#
# mkdir() creates a folder.
#
# parents=True:
#     Creates missing parent folders if necessary.
#
# exist_ok=True:
#     Doesn't throw an error if the folder already exists.
#
# Therefore, our scraper can run repeatedly without failing
# just because these folders already exist.
# ------------------------------------------------------------

DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COLUMN NAMES
# ============================================================

PROJECT_OVERVIEW_COLUMNS = [
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
    "Sanction Date"
]


# ============================================================
# DATA CLEANING
# ============================================================

def clean_missing_values(df):

    """
    Clean missing-value placeholders from PAIMANA data.

    Important:
    We are NOT blindly filling missing values.

    Some missing values actually contain information.

    For example:

        Revised Cost = 0

    may mean:

        "No revised cost has been recorded."

    rather than:

        "The project genuinely costs 0 crore."

    Therefore, we preserve the information by creating
    indicator columns before converting the placeholder
    into a missing value.
    """

    print("\nCleaning missing values...")


    # --------------------------------------------------------
    # Make a copy
    # --------------------------------------------------------
    #
    # copy() creates an independent DataFrame.
    #
    # This prevents us from accidentally modifying the
    # original DataFrame outside this function.
    # --------------------------------------------------------

    df = df.copy()


    # --------------------------------------------------------
    # Convert cost columns to numeric
    # --------------------------------------------------------
    #
    # CSV files often contain numbers as strings.
    #
    # pd.to_numeric() converts them into actual numbers.
    #
    # errors="coerce" means:
    #
    #     If Pandas cannot convert a value into a number,
    #     replace that value with NaN instead of crashing.
    #
    # NaN = Not a Number
    #
    # Pandas commonly uses NaN to represent missing numerical
    # data.
    # --------------------------------------------------------

    cost_columns = [
        "Original Cost (in cr.)",
        "Revised Cost (in cr.)",
        "Expenditure (in cr.)"
    ]

    for column in cost_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # --------------------------------------------------------
    # Convert Physical Progress to numeric
    # --------------------------------------------------------

    df["Physical Progress (in %)"] = pd.to_numeric(
        df["Physical Progress (in %)"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Handle Revised Cost
    # --------------------------------------------------------
    #
    # PAIMANA may use 0 as a placeholder when no revised cost
    # has been recorded.
    #
    # We don't want the model to interpret this as:
    #
    #     Revised Cost = ₹0 crore
    #
    # Instead:
    #
    #     0 → missing value
    #
    # But BEFORE doing that, we create:
    #
    #     has_cost_revision
    #
    # 1 = revised cost exists
    # 0 = no revised cost recorded
    #
    # This preserves useful information for the future model.
    # --------------------------------------------------------

    df["has_cost_revision"] = (
        df["Revised Cost (in cr.)"] > 0
    ).astype(int)


    # Convert placeholder 0 into a missing value.
    df["Revised Cost (in cr.)"] = (
        df["Revised Cost (in cr.)"]
        .replace(0, pd.NA)
    )


    # --------------------------------------------------------
    # Handle Revised Commissioning Date
    # --------------------------------------------------------
    #
    # A missing revised date does NOT necessarily mean the
    # data is bad.
    #
    # It can mean:
    #
    #     "No revised commissioning date has been recorded."
    #
    # So we create:
    #
    #     has_schedule_revision
    #
    # 1 = revised date exists
    # 0 = revised date is missing
    # --------------------------------------------------------

    df["has_schedule_revision"] = (
        df["Revised Date of Commissioning"]
        .notna()
        .astype(int)
    )


    # --------------------------------------------------------
    # Convert dates into Pandas datetime
    # --------------------------------------------------------
    #
    # Dates initially come from the CSV as text.
    #
    # pd.to_datetime() converts them into actual datetime
    # objects so we can perform calculations with them later.
    #
    # dayfirst=True is useful because PAIMANA uses dates such
    # as:
    #
    #     13/07/2024
    #
    # which means:
    #
    #     13 July 2024
    #
    # errors="coerce" means invalid/missing dates become NaT.
    #
    # NaT = Not a Time
    # --------------------------------------------------------

    date_columns = [
        "Original Date of Commissioning",
        "Revised Date of Commissioning",
        "Sanction Date"
    ]

    for column in date_columns:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
            dayfirst=True
        )


    # --------------------------------------------------------
    # Missing-value summary
    # --------------------------------------------------------
    #
    # isna() checks every value in the DataFrame.
    #
    # Missing value → True
    # Existing value → False
    #
    # sum() then counts the True values for each column.
    #
    # Therefore:
    #
    #     df.isna().sum()
    #
    # gives us the number of missing values in each column.
    # --------------------------------------------------------

    missing_summary = df.isna().sum()

    print("\nMissing-value summary:")

    print(
        missing_summary[
            missing_summary > 0
        ]
    )


    print("✓ Missing-value handling complete.")

    return df


# ============================================================
# BASIC FEATURE ENGINEERING
# ============================================================

def create_basic_features(df):

    """
    Create basic project-health features.

    These are NOT necessarily our final ML features.

    We are creating useful measurements that we can inspect
    and later decide whether they should be used by the model.
    """

    print("\nCreating basic features...")

    df = df.copy()


    # --------------------------------------------------------
    # Cost Overrun
    # --------------------------------------------------------
    #
    # Formula:
    #
    #     (Revised Cost - Original Cost)
    #     -------------------------------- × 100
    #            Original Cost
    #
    # Example:
    #
    # Original Cost = 500 crore
    # Revised Cost  = 600 crore
    #
    # Cost Overrun =
    #
    #     (600 - 500) / 500 × 100
    #
    #     = 20%
    #
    # If no revised cost exists, the result stays missing.
    # --------------------------------------------------------

    df["cost_overrun_pct"] = (
        (
            df["Revised Cost (in cr.)"]
            - df["Original Cost (in cr.)"]
        )
        / df["Original Cost (in cr.)"]
    ) * 100


    # --------------------------------------------------------
    # Expenditure Ratio
    # --------------------------------------------------------
    #
    # This tells us how much of the project budget has been
    # spent.
    #
    # Example:
    #
    # Expenditure = 600 crore
    # Revised Cost = 800 crore
    #
    #     600 / 800 = 0.75
    #
    # Therefore:
    #
    #     75% of the budget has been spent.
    #
    # If revised cost is unavailable, we temporarily fall back
    # to the original cost.
    #
    # We may change this later after studying the dataset.
    # --------------------------------------------------------

    cost_for_ratio = (
        df["Revised Cost (in cr.)"]
        .fillna(
            df["Original Cost (in cr.)"]
        )
    )

    df["expenditure_ratio"] = (
        df["Expenditure (in cr.)"]
        / cost_for_ratio
    )


    # --------------------------------------------------------
    # Schedule Extension
    # --------------------------------------------------------
    #
    # Calculate:
    #
    #     Revised Date - Original Date
    #
    # Pandas returns a timedelta here.
    #
    # .dt.days extracts the number of days.
    #
    # Example:
    #
    # Original = 01/01/2025
    # Revised  = 01/07/2025
    #
    # Result ≈ 181 days
    #
    # If the revised date is missing, the result remains NaN.
    # --------------------------------------------------------

    df["schedule_extension_days"] = (
        df["Revised Date of Commissioning"]
        - df["Original Date of Commissioning"]
    ).dt.days


    print("✓ Basic features created.")

    return df


# ============================================================
# CHROME SETUP
# ============================================================


def create_driver():

    # --------------------------------------------------------
    # Selenium Options
    # --------------------------------------------------------

    options = Options()

    # --------------------------------------------------------
    # Chrome download preferences
    # --------------------------------------------------------

    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(
                DOWNLOAD_DIR.resolve()
            ),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
    )

    # --------------------------------------------------------
    # Browser configuration
    # --------------------------------------------------------

    options.add_argument("--start-maximized")

    # Keep Chrome running normally.
    # We intentionally DO NOT use --headless because
    # PAIMANA may behave differently in headless Chrome.

    # --------------------------------------------------------
    # Start Chrome
    # --------------------------------------------------------

    driver = webdriver.Chrome(
        options=options
    )

    # --------------------------------------------------------
    # Minimize Chrome
    # --------------------------------------------------------
    #
    # Chrome still runs normally, but the window is minimized.
    # This avoids the headless-mode problems while keeping the
    # browser out of the user's way.
    # --------------------------------------------------------

    driver.minimize_window()

    return driver



# ============================================================
# PROJECT OVERVIEW
# ============================================================

def project_overview_data(driver, month, year):

    print("\n[Project Overview]")
    print(f"Requested period: {month}/{year}")


    # --------------------------------------------------------
    # Click Project Overview
    # --------------------------------------------------------

    print("Clicking Project Overview...")

    project_overview_button = WebDriverWait(
        driver,
        20
    ).until(
        EC.element_to_be_clickable(
            (
                By.CSS_SELECTOR,
                "div.card.statsblock"
            )
        )
    )

    project_overview_button.click()

    print("✓ Project Overview opened.")

    time.sleep(3)


    # --------------------------------------------------------
    # Click CSV
    # --------------------------------------------------------

    print("Clicking CSV...")

    csv_button = WebDriverWait(
        driver,
        30
    ).until(
        EC.presence_of_element_located(
            (By.ID, "downloadCSV")
        )
    )

    driver.execute_script(
        "arguments[0].click();",
        csv_button
    )

    print("✓ CSV clicked.")


    # --------------------------------------------------------
    # Wait for download
    # --------------------------------------------------------

    print("Waiting for download...")

    downloaded_file = None

    for _ in range(30):

        time.sleep(1)

        # ----------------------------------------------------
        # Path.iterdir()
        # ----------------------------------------------------
        #
        # iterdir() lets us iterate through everything inside
        # a directory.
        #
        # Here we check the download folder and keep only
        # files whose extension is ".csv".
        # ----------------------------------------------------

        csv_files = [
            file
            for file in DOWNLOAD_DIR.iterdir()
            if file.suffix.lower() == ".csv"
        ]

        if csv_files:

            # ------------------------------------------------
            # Find the newest CSV
            # ------------------------------------------------
            #
            # file.stat().st_mtime gives the file's modification
            # time.
            #
            # max(..., key=...) therefore gives us the newest
            # downloaded CSV.
            # ------------------------------------------------

            downloaded_file = max(
                csv_files,
                key=lambda file: file.stat().st_mtime
            )

            break


    if downloaded_file is None:

        raise RuntimeError(
            "Project Overview CSV was not downloaded."
        )


    print(
        f"✓ Downloaded: {downloaded_file.name}"
    )


    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------
    #
    # pd.read_csv() loads CSV data into a Pandas DataFrame.
    #
    # skiprows=2:
    #     Skip the first two rows of the CSV.
    #
    # header=None:
    #     Don't treat the first remaining row as column names.
    #
    # We assign our own column names below.
    # --------------------------------------------------------

    print("Reading CSV...")

    df = pd.read_csv(
        downloaded_file,
        skiprows=2,
        header=None
    )

    print("✓ CSV loaded.")


    # --------------------------------------------------------
    # Add our column names
    # --------------------------------------------------------

    df.columns = PROJECT_OVERVIEW_COLUMNS


    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------
    #
    # dropna() removes rows containing missing values.
    #
    # how="all" is important:
    #
    # It means:
    #
    #     Remove the row ONLY if every value is missing.
    #
    # We don't want to remove a project just because one of
    # its columns is missing.
    # --------------------------------------------------------

    df = df.dropna(
        how="all"
    ).reset_index(
        drop=True
    )


    print(
        f"✓ Rows: {len(df)}"
    )

    print(
        f"✓ Columns: {len(df.columns)}"
    )


    # ========================================================
    # CLEAN MISSING DATA
    # ========================================================

    df = clean_missing_values(df)


    # ========================================================
    # CREATE BASIC FEATURES
    # ========================================================

    df = create_basic_features(df)


    # --------------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------------
    #
    # DataFrame.to_csv() writes our DataFrame back into CSV.
    #
    # index=False:
    #     Don't save Pandas' DataFrame index as an extra column.
    #
    # encoding="utf-8-sig":
    #     Helps spreadsheet programs such as Excel correctly
    #     interpret UTF-8 text.
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        f"project_overview_{month}_{year}.csv"
    )


    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        f"✓ Dataset saved: {output_file}"
    )


    return df


# ============================================================
# MAIN SCRAPER FUNCTION
# ============================================================

def scrape_paimana(month, year):

    """
    Scrape PAIMANA Project Overview data
    for the requested month and year.

    Returns:
        pandas.DataFrame
    """

    driver = create_driver()

    try:

        print("=" * 60)
        print("PAIMANA SCRAPER")
        print(f"Period: {month}/{year}")
        print("=" * 60)


        # ----------------------------------------------------
        # Open PAIMANA
        # ----------------------------------------------------

        print("\nOpening PAIMANA...")

        driver.get(URL)

        WebDriverWait(
            driver,
            60
        ).until(
            lambda d: d.execute_script(
                "return document.readyState"
            ) == "complete"
        )

        print("✓ Dashboard loaded.")

        time.sleep(3)


        # ----------------------------------------------------
        # Fetch Project Overview
        # ----------------------------------------------------

        df = project_overview_data(
            driver,
            month,
            year
        )


        print("\n✓ SCRAPING COMPLETE")

        return df


    finally:

        # ----------------------------------------------------
        # Always close the browser
        # ----------------------------------------------------
        #
        # finally runs whether the scraping succeeds or fails.
        #
        # This is important because we don't want Chrome
        # processes remaining open if an exception occurs.
        # ----------------------------------------------------

        driver.quit()

        print("✓ Browser closed.")

# ============================================================
# RUN SCRAPER
# ============================================================

if __name__ == "__main__":
    import sys  # Built-in tool to read terminal inputs

    # Check if the user forgot to pass month and year
    if len(sys.argv) < 3:
        print("\n❌ Error: Please provide month and year.")
        print("Usage: python scraper.py <month> <year>")
        print("Example: python scraper.py 04 2026\n")
        sys.exit(1)

    # Automatically read inputs directly from the terminal execution line
    month = sys.argv[1].strip()
    year = sys.argv[2].strip()

    df = scrape_paimana(
        month,
        year
    )
    
