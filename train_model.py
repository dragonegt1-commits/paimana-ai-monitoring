# train_model.py
"""Train the PAIMANA anomaly model from a CSV snapshot."""

import sys  # Built-in tool to read terminal inputs
from pathlib import Path
from data_pipeline import create_features, load_paimana_csv
from ml_pipeline import train_and_save

# Check if the user forgot to pass month and year
if len(sys.argv) < 3:
    print("\n❌ Error: Please provide month and year.")
    print("Usage: python train_model.py <month> <year>")
    print("Example: python train_model.py 04 2026\n")
    sys.exit(1)

# Automatically read inputs from terminal execution line
month = sys.argv[1].strip()
year = sys.argv[2].strip()

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Dataset location
CSV_PATH = BASE_DIR / "paimana_output" / f"project_overview_{month}_{year}.csv"


if __name__ == "__main__":
    print("=" * 60)
    print("PAIMANA ML MODEL TRAINING")
    print("=" * 60)

    # Check that the dataset exists before loading
    if not CSV_PATH.exists():
        print("\n❌ Dataset not found!")
        print(f"Expected location:\n{CSV_PATH}")
        raise SystemExit(1)

    print(f"\nDataset: {CSV_PATH}")

    df = load_paimana_csv(CSV_PATH)

    cleaned, X = create_features(
        df,
        snapshot_date=f"{year}-{month}-01",
    )

    print(f"Projects used: {len(cleaned)}")
    print(f"Features used: {len(X.columns)}")

    model = train_and_save(X)

    print("\n✓ Model trained and saved.")
    print("✓ Model can now be loaded by FastAPI.")

