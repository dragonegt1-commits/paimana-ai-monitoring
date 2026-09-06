# train_model.py
"""Train the PAIMANA anomaly model from a CSV snapshot."""

from pathlib import Path

from data_pipeline import create_features, load_paimana_csv
from ml_pipeline import train_and_save


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "project_overview_april_2026.csv"


if __name__ == "__main__":
    print("=" * 60)
    print("PAIMANA ML MODEL TRAINING")
    print("=" * 60)

    df = load_paimana_csv(CSV_PATH)

    # April 2026 is the snapshot used for the first model.
    cleaned, X = create_features(
        df,
        snapshot_date="2026-04-01",
    )

    print(f"Projects used: {len(cleaned)}")
    print(f"Features used: {len(X.columns)}")

    model = train_and_save(X)

    print("\n✓ Model trained and saved.")
    print("✓ Model can now be loaded by FastAPI.")
