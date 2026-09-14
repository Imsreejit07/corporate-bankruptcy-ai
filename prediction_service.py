"""Prediction service for the calibrated corporate bankruptcy model.

The model artifacts are treated as immutable inputs. This module only loads
them and applies the prediction and explanation steps needed by the API.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb


MODEL_DIR = Path(__file__).resolve().parent / "model_artifacts"
CONFIG_PATH = MODEL_DIR / "config.json"
MODEL_PATH = MODEL_DIR / "xgb_model.json"
CALIBRATOR_PATH = MODEL_DIR / "platt_calibrator.pkl"

FEATURE_LABELS = {
    "X1": "Current assets",
    "X2": "Cost of goods sold",
    "X3": "Depreciation and amortization",
    "X4": "EBITDA",
    "X5": "Inventory",
    "X6": "Net income",
    "X7": "Total receivables",
    "X8": "Market value",
    "X9": "Net sales",
    "X10": "Total assets",
    "X11": "Total long-term debt",
    "X12": "EBIT",
    "X13": "Gross profit",
    "X14": "Total current liabilities",
    "X15": "Retained earnings",
    "X17": "Total liabilities",
    "X18": "Total operating expenses",
}


@dataclass(frozen=True)
class ModelConfig:
    features: tuple[str, ...]
    threshold: float
    low_risk_limit: float
    high_risk_limit: float


class ModelArtifactError(RuntimeError):
    """Raised when the immutable model artifacts cannot be loaded safely."""


def _load_config() -> ModelConfig:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            raw = json.load(config_file)
        config = ModelConfig(
            features=tuple(raw["features"]),
            threshold=float(raw["threshold"]),
            low_risk_limit=float(raw["low_risk_limit"]),
            high_risk_limit=float(raw["high_risk_limit"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ModelArtifactError(f"Unable to load model configuration: {exc}") from exc

    if len(config.features) != 17 or "X16" in config.features:
        raise ModelArtifactError("The model configuration must contain 17 features and no X16.")
    if set(config.features) != set(FEATURE_LABELS):
        raise ModelArtifactError("The configured feature set does not match the application metadata.")
    if not 0 <= config.low_risk_limit < config.threshold < config.high_risk_limit <= 1:
        raise ModelArtifactError("Model risk boundaries are not ordered or bounded.")
    return config


def _load_booster(expected_features: tuple[str, ...]) -> xgb.Booster:
    try:
        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH))
    except Exception as exc:  # xgboost raises several native exception types
        raise ModelArtifactError(f"Unable to load the XGBoost model: {exc}") from exc

    if tuple(booster.feature_names or ()) != expected_features:
        raise ModelArtifactError("The XGBoost feature order does not match config.json.")
    return booster


@dataclass
class LoadedModel:
    config: ModelConfig
    booster: xgb.Booster
    calibrator: Any

    @classmethod
    def load(cls) -> "LoadedModel":
        config = _load_config()
        booster = _load_booster(config.features)
        try:
            calibrator = joblib.load(CALIBRATOR_PATH)
        except Exception as exc:
            raise ModelArtifactError(f"Unable to load the Platt calibrator: {exc}") from exc

        if getattr(calibrator, "n_features_in_", None) != 1:
            raise ModelArtifactError("The Platt calibrator must accept one raw probability feature.")
        return cls(config=config, booster=booster, calibrator=calibrator)

    def predict(self, values: dict[str, float]) -> dict[str, Any]:
        ordered_values = np.asarray(
            [[float(values[feature]) for feature in self.config.features]],
            dtype=np.float32,
        )
        if not np.isfinite(ordered_values).all():
            raise ValueError("All model inputs must be finite numbers.")

        matrix = xgb.DMatrix(ordered_values, feature_names=list(self.config.features))
        raw_probability = float(self.booster.predict(matrix)[0])
        calibrated_probability = float(
            self.calibrator.predict_proba(np.asarray([[raw_probability]], dtype=np.float64))[0, 1]
        )
        calibrated_probability = min(max(calibrated_probability, 0.0), 1.0)

        # XGBoost's native pred_contribs is its TreeSHAP implementation. The
        # final value is the bias term; only the 17 feature contributions are
        # returned to the client.
        contribution_row = np.asarray(self.booster.predict(matrix, pred_contribs=True)[0])
        contributions = contribution_row[: len(self.config.features)]
        strongest = sorted(
            zip(self.config.features, contributions),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[:10]
        feature_contributions = [
            {
                "feature": feature,
                "label": FEATURE_LABELS[feature],
                "input_value": float(values[feature]),
                "shap_value": float(shap_value),
                "absolute_shap_value": abs(float(shap_value)),
                "direction": (
                    "higher bankruptcy risk"
                    if shap_value > 0
                    else "lower bankruptcy risk"
                    if shap_value < 0
                    else "neutral"
                ),
            }
            for feature, shap_value in strongest
        ]

        if calibrated_probability < self.config.low_risk_limit:
            risk_category = "LOW RISK"
        elif calibrated_probability >= self.config.high_risk_limit:
            risk_category = "HIGH RISK"
        else:
            risk_category = "MODERATE RISK"

        flagged = calibrated_probability >= self.config.threshold
        decision = "FLAG FOR REVIEW" if flagged else "DO NOT FLAG"
        if risk_category == "HIGH RISK":
            suggested_action = (
                "Flag for senior review and investigate the strongest model signals."
            )
        elif flagged:
            suggested_action = (
                "Flag for review and validate the strongest model signals with current financial records."
            )
        elif risk_category == "LOW RISK":
            suggested_action = (
                "Continue routine monitoring; the calibrated probability is below the low-risk boundary."
            )
        else:
            suggested_action = (
                "Continue monitoring; the calibrated probability is below the decision threshold."
            )

        return {
            "bankruptcy_probability": calibrated_probability,
            "raw_probability": raw_probability,
            "risk_category": risk_category,
            "decision": decision,
            "suggested_action": suggested_action,
            "decision_threshold": self.config.threshold,
            "low_risk_limit": self.config.low_risk_limit,
            "high_risk_limit": self.config.high_risk_limit,
            "feature_contributions": feature_contributions,
        }


MODEL = LoadedModel.load()
