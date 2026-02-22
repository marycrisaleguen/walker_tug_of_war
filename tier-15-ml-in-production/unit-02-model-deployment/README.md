# Unit 2 — Model Deployment

## Topics

### 1. Deployment Options
- REST API (most common): wrap model in an HTTP server; clients POST features, receive predictions
- Batch inference: process data in bulk on a schedule
- Embedded model: model runs on-device (mobile, edge) — TensorFlow Lite, ONNX Runtime
- Streaming: Kafka + Flink for high-throughput real-time predictions

### 2. Building a REST API with FastAPI
- FastAPI: modern Python web framework with automatic documentation
- Define request/response schemas with Pydantic
- Load model once at startup, use it for every request
- Async support for concurrent requests

### 3. Containerization with Docker
- Package application + dependencies + model into a portable container
- Same image runs identically on laptop, cloud server, Kubernetes
- Key files: `Dockerfile`, `requirements.txt`
- Key commands: `docker build`, `docker run`, `docker push`

### 4. Cloud Deployment
- **AWS SageMaker**: managed ML training, deployment, and monitoring
- **Google Vertex AI**: similar managed service on GCP
- **Azure Machine Learning**: Microsoft's ML platform
- **Kubernetes**: orchestrate containers at scale
- For simple models: managed serverless (AWS Lambda, Cloud Functions) + API Gateway

### 5. Model Serialization Formats
- **Pickle** (sklearn): `joblib.dump(model, "model.pkl")`
- **Keras SavedModel / H5**: native TensorFlow format
- **PyTorch** `.pt`: `torch.save(model.state_dict(), "model.pt")`
- **ONNX**: open, framework-agnostic format; run with ONNX Runtime

### 6. Serving Considerations
- Latency: P95/P99 response time SLA (e.g., < 100ms)
- Throughput: requests per second
- Model versioning: blue-green deployments, canary releases
- Authentication and rate limiting

---

## Exercises

### Exercise 1 — FastAPI Prediction Service
```python
# app.py
# pip install fastapi uvicorn scikit-learn joblib pydantic
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Prediction Service")

# Load model at startup (pre-trained)
# In a real project: joblib.load("model.pkl")
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler().fit(X)
model  = LogisticRegression(max_iter=1000).fit(scaler.transform(X), y)

class PredictRequest(BaseModel):
    features: list[float]

class PredictResponse(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    features = np.array(request.features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    pred  = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0, 1]
    return PredictResponse(prediction=int(pred), probability=float(proba))

@app.get("/health")
def health():
    return {"status": "healthy"}

# Run with: uvicorn app:app --reload
# Test: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [...]}'
```

### Exercise 2 — Dockerfile for ML Service
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app.py .
COPY model.pkl .
COPY scaler.pkl .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t ml-service:v1 .
docker run -p 8000:8000 ml-service:v1

# Push to registry
docker tag ml-service:v1 myregistry/ml-service:v1
docker push myregistry/ml-service:v1
```

### Exercise 3 — Save and Load Models (sklearn, PyTorch)
```python
import joblib
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# sklearn: save with joblib
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=1000).fit(X, y)
joblib.dump(model, "/tmp/iris_model.pkl")
loaded_model = joblib.load("/tmp/iris_model.pkl")
print("sklearn model loaded, predictions:", loaded_model.predict(X[:3]))

# PyTorch: save/load state dict
import torch.nn as nn
net = nn.Linear(4, 3)
torch.save(net.state_dict(), "/tmp/iris_net.pt")
net2 = nn.Linear(4, 3)
net2.load_state_dict(torch.load("/tmp/iris_net.pt"))
print("PyTorch model loaded")
```

---

## Questions & Answers

**Q1: Why use FastAPI instead of Flask for ML serving?**
> FastAPI has several advantages: automatic request/response validation via Pydantic, async support (can serve multiple requests concurrently), automatic interactive API documentation (Swagger UI at `/docs`), and superior performance (based on Starlette + uvicorn). For simple use cases, Flask also works well.

**Q2: Why containerize ML models with Docker?**
> Docker ensures the model runs in exactly the same environment everywhere: same Python version, same library versions, same system dependencies. This eliminates "it works on my machine" problems. Containers are portable, can be orchestrated at scale with Kubernetes, and integrate with cloud deployment platforms.

**Q3: What is model versioning and why does it matter?**
> Each deployed model should have a version number. When you update a model, you can deploy the new version alongside the old (blue-green deployment) and gradually shift traffic (canary release). If the new model has issues, you roll back to the previous version immediately. Without versioning, a bad model update can take down an entire service.

**Q4: What is ONNX and when should I use it?**
> ONNX (Open Neural Network Exchange) is an open format for representing ML models. You export from PyTorch/TensorFlow once, then run inference with ONNX Runtime on any platform (CPU, GPU, mobile, edge). It often gives 2–5× speedup over native framework inference. Use when you need: framework-agnostic serving, edge deployment, or maximum inference speed.

**Q5: What should a model's health check endpoint return?**
> At minimum: `{"status": "healthy"}` with HTTP 200. A production health check should also verify: model is loaded and can make predictions, dependencies (database, feature store) are reachable, and GPU (if used) is available. Kubernetes uses health checks to automatically restart unhealthy pods.
