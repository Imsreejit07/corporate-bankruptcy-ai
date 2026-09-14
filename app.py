from __future__ import annotations

import math
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from prediction_service import FEATURE_LABELS, MODEL


ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    X1: float = Field(..., description=FEATURE_LABELS["X1"])
    X2: float = Field(..., description=FEATURE_LABELS["X2"])
    X3: float = Field(..., description=FEATURE_LABELS["X3"])
    X4: float = Field(..., description=FEATURE_LABELS["X4"])
    X5: float = Field(..., description=FEATURE_LABELS["X5"])
    X6: float = Field(..., description=FEATURE_LABELS["X6"])
    X7: float = Field(..., description=FEATURE_LABELS["X7"])
    X8: float = Field(..., description=FEATURE_LABELS["X8"])
    X9: float = Field(..., description=FEATURE_LABELS["X9"])
    X10: float = Field(..., description=FEATURE_LABELS["X10"])
    X11: float = Field(..., description=FEATURE_LABELS["X11"])
    X12: float = Field(..., description=FEATURE_LABELS["X12"])
    X13: float = Field(..., description=FEATURE_LABELS["X13"])
    X14: float = Field(..., description=FEATURE_LABELS["X14"])
    X15: float = Field(..., description=FEATURE_LABELS["X15"])
    X17: float = Field(..., description=FEATURE_LABELS["X17"])
    X18: float = Field(..., description=FEATURE_LABELS["X18"])

    @field_validator("*")
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Value must be a finite number.")
        return value

    def as_feature_dict(self) -> dict[str, float]:
        return self.model_dump()


app = FastAPI(
    title="Corporate Bankruptcy Risk Assessment API",
    description="Decision-support API around the imported calibrated XGBoost model.",
    version="1.0.0",
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": True,
        "features": list(MODEL.config.features),
        "decision_threshold": MODEL.config.threshold,
        "low_risk_limit": MODEL.config.low_risk_limit,
        "high_risk_limit": MODEL.config.high_risk_limit,
    }


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict:
    return MODEL.predict(request.as_feature_dict())


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
