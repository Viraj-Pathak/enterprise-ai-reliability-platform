from pydantic import BaseModel, Field

class MetricInput(BaseModel):
    cpu_usage: float = Field(..., ge=0, le=100)
    memory_usage: float = Field(..., ge=0, le=100)
    disk_usage: float = Field(..., ge=0, le=100)
    network_latency_ms: float = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=100)
    packet_loss: float = Field(..., ge=0, le=100)
    requests_per_min: float = Field(..., ge=0)

class RiskPrediction(BaseModel):
    risk_level: str
    risk_score: float
    anomaly_score: float
    details: dict

class RecommendationResponse(BaseModel):
    risk_level: str
    summary: str
    recommended_actions: list[str]
