from pathlib import Path
import pandas as pd
import joblib
from .data_preprocessing import (
    FEATURE_COLUMNS,
    transform_features,
    load_scaler,
    MODELS_DIR,
)
from ..utils.logger import get_logger

logger = get_logger("inference")

RISK_MODEL_PATH = MODELS_DIR / "risk_classifier.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detector.joblib"

class ReliabilityModel:
    def __init__(self) -> None:
        self.scaler = load_scaler()
        self.risk_model = joblib.load(RISK_MODEL_PATH)
        self.anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
        logger.info("Loaded models for inference")

    def predict(self, metrics: dict) -> dict:
        df = pd.DataFrame([metrics], columns=FEATURE_COLUMNS)
        X_scaled = transform_features(df, self.scaler)

        risk_label = self.risk_model.predict(X_scaled)[0]
        proba = self.risk_model.predict_proba(X_scaled)[0]
        classes = list(self.risk_model.classes_)
        prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}

        anomaly_score = float(self.anomaly_model.decision_function(X_scaled)[0])

        result = {
            "risk_level": str(risk_label),
            "risk_probabilities": prob_dict,
            "anomaly_score": anomaly_score,
        }
        return result

def generate_recommendations(risk_level: str, metrics: dict) -> dict:
    cpu = metrics["cpu_usage"]
    mem = metrics["memory_usage"]
    disk = metrics["disk_usage"]
    latency = metrics["network_latency_ms"]
    error_rate = metrics["error_rate"]
    packet_loss = metrics["packet_loss"]
    rpm = metrics["requests_per_min"]

    actions = []
    summary_parts = []

    if risk_level == "LOW":
        summary_parts.append("System is healthy with low risk.")
        actions.append("Continue regular monitoring of key metrics.")
        actions.append("Review alerts configuration weekly.")
    elif risk_level == "MEDIUM":
        summary_parts.append("System shows moderate risk.")
        actions.append("Review recent deployments and configuration changes.")
        actions.append("Increase logging temporarily.")
    else:
        summary_parts.append("System is under high risk of failure.")
        actions.append("Trigger incident response.")
        actions.append("Scale affected services or reduce load.")
        actions.append("Capture detailed logs for review.")

    if cpu > 80:
        actions.append("Investigate CPU spikes and heavy workloads.")
    if mem > 80:
        actions.append("Check memory leaks and optimize usage.")
    if disk > 85:
        actions.append("Clean unused files or expand storage.")
    if latency > 250:
        actions.append("Check dependencies causing high latency.")
    if error_rate > 5:
        actions.append("Investigate failing services and roll back changes.")
    if packet_loss > 3:
        actions.append("Check network routing or load balancer.")
    if rpm > 4000:
        actions.append("Enable autoscaling or reduce incoming requests.")

    summary = " ".join(summary_parts)

    return {
        "risk_level": risk_level,
        "summary": summary,
        "recommended_actions": actions,
    }
