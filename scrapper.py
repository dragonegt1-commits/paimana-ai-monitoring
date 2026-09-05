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

DOWNLOAD_DIR = Path.cwd() / "paimana_downloads"
OUTPUT_DIR = Path.cwd() / "paimana_output"

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
# CHROME SETUP
# ============================================================

def create_driver():

    options = Options()

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

    options.add_argument("--start-maximized")

    # --------------------------------------------------------
    # HEADLESS MODE
    # --------------------------------------------------------

    options.add_argument("--headless=new")

    # Useful for headless environments
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        options=options
    )

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

        csv_files = [
            file
            for file in DOWNLOAD_DIR.iterdir()
            if file.suffix.lower() == ".csv"
        ]

        if csv_files:

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

    print("Reading CSV...")

    df = pd.read_csv(
        downloaded_file,
        skiprows=2,
        header=None
    )

    print("✓ CSV loaded.")

    # --------------------------------------------------------
    # Add column names
    # --------------------------------------------------------

    df.columns = PROJECT_OVERVIEW_COLUMNS

    # --------------------------------------------------------
    # Remove empty rows
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

    # --------------------------------------------------------
    # Save
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

        driver.quit()

        print("✓ Browser closed.")