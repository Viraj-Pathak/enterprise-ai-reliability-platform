from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .schemas import MetricInput, RiskPrediction, RecommendationResponse
from .ml.inference import ReliabilityModel, generate_recommendations
from .utils.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Enterprise AI Reliability Intelligence Platform",
    version="1.0.0",
    description="Predicts risk and recommends remediation from system metrics.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ReliabilityModel()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=RiskPrediction)
def predict_risk(metrics: MetricInput):
    metrics_dict = metrics.dict()
    logger.info("Received metrics: %s", metrics_dict)

    result = model.predict(metrics_dict)

    return RiskPrediction(
        risk_level=result["risk_level"],
        risk_score=max(result["risk_probabilities"].values()),
        anomaly_score=result["anomaly_score"],
        details={"probabilities": result["risk_probabilities"]},
    )

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(metrics: MetricInput):
    metrics_dict = metrics.dict()
    pred = model.predict(metrics_dict)
    rec = generate_recommendations(pred["risk_level"], metrics_dict)

    return RecommendationResponse(
        risk_level=rec["risk_level"],
        summary=rec["summary"],
        recommended_actions=rec["recommended_actions"],
    )
