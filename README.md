# Enterprise AI Reliability Intelligence Platform

Backend service that predicts infrastructure failure risk and recommends remediation actions using machine learning.

This repository currently contains the **backend** implementation built with:

- Python
- FastAPI
- Scikit Learn (RandomForest + IsolationForest)
- Synthetic cloud metrics dataset (CPU, Memory, Disk, Latency, Packet Loss, Error Rate, Traffic)

👉 Detailed documentation and setup instructions are in  
[`backend/README.md`](backend/README.md)

Planned next steps:
- React reliability dashboard
- Cloud deployment
- Real cloud metrics integration
