"""Prediction service for the calibrated corporate bankruptcy model.

Pure Python implementation without native C++ / compiled library dependencies (no xgboost, no numpy).
This guarantees deployment compliance under Vercel's 500MB function size limit (~10MB total bundle size).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_DIR = Path(__file__).resolve().parent / "model_artifacts"
CONFIG_PATH = MODEL_DIR / "config.json"
MODEL_PATH = MODEL_DIR / "xgb_model.json"

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


# TreeTuple = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[float, ...], tuple[bool, ...]]


def _load_parsed_trees(expected_features: tuple[str, ...]) -> tuple[list[Any], dict[str, int]]:
    try:
        with MODEL_PATH.open("r", encoding="utf-8") as model_file:
            data = json.load(model_file)
        learner = data["learner"]
        feature_names = learner["feature_names"]
        if tuple(feature_names) != expected_features:
            raise ModelArtifactError("The XGBoost feature order does not match config.json.")
        
        feature_idx_map = {name: i for i, name in enumerate(feature_names)}
        raw_trees = learner["gradient_booster"]["model"]["trees"]
        
        parsed_trees: list[Any] = []
        for t in raw_trees:
            parsed_trees.append((
                tuple(t["left_children"]),
                tuple(t["right_children"]),
                tuple(t["split_indices"]),
                tuple(t["split_conditions"]),
                tuple(bool(x) for x in t["default_left"]),
            ))
        return parsed_trees, feature_idx_map
    except Exception as exc:
        raise ModelArtifactError(f"Unable to load the XGBoost trees: {exc}") from exc


@dataclass
class LoadedModel:
    config: ModelConfig
    parsed_trees: list[Any]
    feature_idx_map: dict[str, int]

    @classmethod
    def load(cls) -> "LoadedModel":
        config = _load_config()
        parsed_trees, feature_idx_map = _load_parsed_trees(config.features)
        return cls(config=config, parsed_trees=parsed_trees, feature_idx_map=feature_idx_map)

    def predict(self, values: dict[str, float]) -> dict[str, Any]:
        val_list = [float(values[feature]) for feature in self.config.features]
        for val in val_list:
            if not math.isfinite(val):
                raise ValueError("All model inputs must be finite numbers.")

        margin = 0.0
        contributions = [0.0] * len(val_list)

        for lefts, rights, indices, conds, defaults in self.parsed_trees:
            node = 0
            path = []
            while lefts[node] != -1:
                feat_idx = indices[node]
                val = val_list[feat_idx]
                thresh = conds[node]
                path.append(feat_idx)
                if math.isnan(val):
                    node = lefts[node] if defaults[node] else rights[node]
                elif val < thresh:
                    node = lefts[node]
                else:
                    node = rights[node]
            leaf_val = conds[node]
            margin += leaf_val
            if path:
                weight = leaf_val / len(path)
                for f_idx in path:
                    contributions[f_idx] += weight

        raw_probability = 1.0 / (1.0 + math.exp(-margin))

        # Platt scaling (Logistic Regression on raw probability):
        # coef = 3.1184013906623225, intercept = -3.9089253782211704
        logit = 3.1184013906623225 * raw_probability - 3.9089253782211704
        calibrated_probability = 1.0 / (1.0 + math.exp(-logit))
        calibrated_probability = min(max(calibrated_probability, 0.0), 1.0)

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
