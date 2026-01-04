import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "network_latency_ms",
    "error_rate",
    "packet_loss",
    "requests_per_min",
]

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SCALER_PATH = MODELS_DIR / "scaler.joblib"

def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(42)

    cpu = rng.normal(loc=50, scale=15, size=n_samples).clip(0, 100)
    mem = rng.normal(loc=55, scale=18, size=n_samples).clip(0, 100)
    disk = rng.normal(loc=60, scale=20, size=n_samples).clip(0, 100)
    latency = rng.normal(loc=120, scale=60, size=n_samples).clip(5, None)
    error_rate = rng.normal(loc=2, scale=3, size=n_samples).clip(0, 100)
    packet_loss = rng.normal(loc=1, scale=2, size=n_samples).clip(0, 100)
    rpm = rng.normal(loc=2000, scale=800, size=n_samples).clip(50, None)

    data = pd.DataFrame(
        {
            "cpu_usage": cpu,
            "memory_usage": mem,
            "disk_usage": disk,
            "network_latency_ms": latency,
            "error_rate": error_rate,
            "packet_loss": packet_loss,
            "requests_per_min": rpm,
        }
    )

    risk_score = (
        0.25 * (data["cpu_usage"] / 100)
        + 0.2 * (data["memory_usage"] / 100)
        + 0.2 * (data["disk_usage"] / 100)
        + 0.15 * (data["network_latency_ms"] / 500)
        + 0.1 * (data["error_rate"] / 100)
        + 0.05 * (data["packet_loss"] / 100)
        + 0.05 * (data["requests_per_min"] / 5000)
    )

    risk_score = risk_score.clip(0, 1)
    data["risk_score"] = risk_score

    labels = ["LOW", "MEDIUM", "HIGH"]
    data["risk_level"] = pd.cut(
        risk_score, bins=[-0.01, 0.3, 0.6, 1.0], labels=labels
    )

    return data

def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[FEATURE_COLUMNS])
    joblib.dump(scaler, SCALER_PATH)
    return scaler

def load_scaler() -> StandardScaler:
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    return joblib.load(SCALER_PATH)

def transform_features(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    scaled = scaler.transform(df[FEATURE_COLUMNS])
    scaled_df = pd.DataFrame(scaled, columns=FEATURE_COLUMNS)
    return scaled_df
