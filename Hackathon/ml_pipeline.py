# ml_pipeline.py
"""
PAIMANA ML layer.

Current model:
    Isolation Forest

Why?
The April 2026 dataset contains project observations but does not contain
a trustworthy historical "failed/delayed" label. A supervised classifier
would therefore require an invented target. Isolation Forest lets us learn
unusual project patterns without inventing labels.

When multiple monthly PAIMANA snapshots are collected, this module can be
upgraded to a supervised delay-prediction model using future outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "paimana_anomaly_model.joblib"
META_PATH = MODEL_DIR / "paimana_model_metadata.json"


class PaimanaMLModel:
    """Train, save, load and score the PAIMANA anomaly model."""

    def __init__(self) -> None:
        self.pipeline: Pipeline | None = None
        self.feature_columns: list[str] = []

    def build_model(self) -> Pipeline:
        # Median imputation handles missing revised cost/date-derived values.
        # RobustScaler is less sensitive to huge infrastructure projects
        # than ordinary standard scaling.
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
                (
                    "model",
                    IsolationForest(
                        n_estimators=400,
                        contamination="auto",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame) -> "PaimanaMLModel":
        if X.empty:
            raise ValueError("Cannot train ML model on an empty dataset.")

        self.feature_columns = list(X.columns)
        self.pipeline = self.build_model()
        self.pipeline.fit(X[self.feature_columns])
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return anomaly scores from 0 to 100.

        Higher = more unusual compared with the training portfolio.
        """
        if self.pipeline is None:
            raise RuntimeError("ML model has not been trained/loaded.")

        X = X.reindex(columns=self.feature_columns)

        # IsolationForest decision_function: larger = more normal.
        raw = self.pipeline.decision_function(X)

        # Convert the current batch to an interpretable 0-100 anomaly scale.
        # This is a presentation score, not a probability.
        low, high = np.percentile(raw, [5, 95])
        if high <= low:
            normalized = np.full(len(raw), 50.0)
        else:
            normalized = 100 * (high - raw) / (high - low)

        return np.clip(normalized, 0, 100)

    def predict_labels(self, X: pd.DataFrame) -> np.ndarray:
        """Return Isolation Forest's native -1 anomaly / +1 normal labels."""
        if self.pipeline is None:
            raise RuntimeError("ML model has not been trained/loaded.")
        X = X.reindex(columns=self.feature_columns)
        return self.pipeline.predict(X)

    def save(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("Nothing to save; train the model first.")

        joblib.dump(self.pipeline, MODEL_PATH)
        META_PATH.write_text(
            json.dumps(
                {
                    "feature_columns": self.feature_columns,
                    "model": "IsolationForest",
                    "version": 1,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls) -> "PaimanaMLModel":
        if not MODEL_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                "Trained PAIMANA model not found. Run train_model.py first."
            )

        obj = cls()
        obj.pipeline = joblib.load(MODEL_PATH)
        metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
        obj.feature_columns = metadata["feature_columns"]
        return obj


def train_and_save(X: pd.DataFrame) -> PaimanaMLModel:
    """Train the model and save it for FastAPI prediction."""
    model = PaimanaMLModel().fit(X)
    model.save()
    return model
