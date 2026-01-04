from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .data_preprocessing import (
    generate_synthetic_data,
    fit_scaler,
    transform_features,
    FEATURE_COLUMNS,
    MODELS_DIR,
)
from ..utils.logger import get_logger

logger = get_logger("train_model")

RISK_MODEL_PATH = MODELS_DIR / "risk_classifier.joblib"
ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_detector.joblib"

def train():
    logger.info("Starting training for reliability models")

    df = generate_synthetic_data(n_samples=8000)
    logger.info("Generated synthetic dataset")

    scaler = fit_scaler(df)
    X_scaled = transform_features(df, scaler)
    y = df["risk_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )

    clf.fit(X_train, y_train)
    logger.info("Trained RandomForest risk classifier")

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info("Risk classifier report:\n%s", report)

    joblib.dump(clf, RISK_MODEL_PATH)
    logger.info("Saved risk classifier to %s", RISK_MODEL_PATH)

    iso = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42,
    )
    iso.fit(X_scaled)
    joblib.dump(iso, ANOMALY_MODEL_PATH)
    logger.info("Saved anomaly detector to %s", ANOMALY_MODEL_PATH)

if __name__ == "__main__":
    train()
