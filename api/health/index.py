from fastapi import FastAPI
from prediction_service import MODEL

app = FastAPI()

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
