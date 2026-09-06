from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import json

# ── Browser Setup ──────────────────────────────────────────────
options = Options()
# options.add_argument("--headless")  # Uncomment after confirming it works visually
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)
options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# ── Navigate ───────────────────────────────────────────────────
URL = "https://paimana-proj.mospi.gov.in"
driver.get(URL)

# Wait for the page to fully load (adjust selector after inspecting the site)
try:
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.TAG_NAME, "table"))
    )
    print("Table found — extracting data...")
except Exception:
    print("No <table> found within timeout. Trying to extract all visible text...")

time.sleep(3)  # Extra buffer for JS rendering

# ── Extract HTML & Parse ───────────────────────────────────────
soup = BeautifulSoup(driver.page_source, "html.parser")

# --- Extract all tables ---
tables = soup.find_all("table")
all_dataframes = []

for i, table in enumerate(tables):
    try:
        df = pd.read_html(str(table))[0]
        print(f"\n--- Table {i+1} ---")
        print(df.head())
        all_dataframes.append(df)
    except Exception as e:
        print(f"Could not parse table {i+1}: {e}")

# --- Save to CSV ---
if all_dataframes:
    combined = pd.concat(all_dataframes, ignore_index=True)
    combined.to_csv("paimana_data.csv", index=False)
    print("\nData saved to paimana_data.csv")
else:
    # Fallback: extract all visible text
    text = soup.get_text(separator="\n", strip=True)
    with open("paimana_raw_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("No tables found. Raw text saved to paimana_raw_text.txt")

driver.quit()