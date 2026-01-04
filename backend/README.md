# Enterprise AI Reliability Intelligence Platform

An AI-powered Reliability & Operations Intelligence backend that predicts
system failure risk and recommends remediation actions using machine learning.

## 🚀 What this platform does
✔ Predicts infrastructure failure risk (Low, Medium, High)  
✔ Detects abnormal infrastructure behavior using anomaly detection  
✔ Generates actionable remediation steps for DevOps / SRE teams  
✔ Provides production-ready REST APIs built on FastAPI  
✔ Extensible architecture ready for dashboards and cloud deployment  

---

## 🧠 Machine Learning Intelligence
The platform uses:

- RandomForestClassifier → Reliability Risk Prediction
- IsolationForest → Behavior Anomaly Detection
- StandardScaler → Feature normalization
- Synthetic cloud metrics dataset → CPU, Memory, Disk, Latency, Packet Loss, Errors, Traffic load

Models automatically save to:
