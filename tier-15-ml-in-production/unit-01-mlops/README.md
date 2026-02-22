# Unit 1 — MLOps Fundamentals

## Topics

### 1. What is MLOps?
- The practice of deploying, monitoring, and maintaining ML models in production reliably and efficiently
- Combines ML, Software Engineering, and DevOps
- Goal: move from "one-off notebook experiments" to "continuously improving production systems"

### 2. The ML Lifecycle
1. Problem definition and data collection
2. Data preprocessing and EDA
3. Feature engineering
4. Model training and evaluation
5. Deployment
6. Monitoring and maintenance
7. Retraining → back to step 3

### 3. Experiment Tracking with MLflow
- Track runs: hyperparameters, metrics, artifacts, code version
- `mlflow.log_param()`, `mlflow.log_metric()`, `mlflow.log_artifact()`
- MLflow UI: compare runs, visualize metrics, retrieve models
- Model Registry: version, stage (Staging/Production), and manage models

### 4. Data and Model Versioning with DVC
- DVC (Data Version Control): git-like versioning for large data files and models
- Stores data in remote storage (S3, GCS) with lightweight pointers in git
- `dvc add data.csv`, `dvc push`, `dvc pull`
- Reproducibility: `dvc repro` reruns the pipeline with any data or code change

### 5. ML Pipelines
- Encode the entire ML workflow as a DAG (Directed Acyclic Graph)
- Tools: DVC pipelines, Airflow, Prefect, Kubeflow Pipelines
- Benefits: reproducibility, automation, auditability

### 6. CI/CD for ML
- Continuous Integration: automatically test data quality, model quality, and code on every commit
- Continuous Delivery: automatically deploy a new model when it passes evaluation criteria
- GitHub Actions, GitLab CI, CircleCI for automation

---

## Exercises

### Exercise 1 — Experiment Tracking with MLflow
```python
# pip install mlflow
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

mlflow.set_experiment("breast_cancer_classification")

for C in [0.01, 0.1, 1.0, 10.0]:
    with mlflow.start_run(run_name=f"logreg_C={C}"):
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_tr_s, y_tr)
        y_pred  = model.predict(X_te_s)
        y_proba = model.predict_proba(X_te_s)[:, 1]
        acc = accuracy_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_proba)

        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")
        print(f"C={C}: accuracy={acc:.4f}, AUC={auc:.4f}")

print("\nView experiments: run `mlflow ui` in your terminal")
```

### Exercise 2 — DVC Workflow (Shell Commands)
```bash
# Initialize DVC in a git repository
git init my-ml-project && cd my-ml-project
dvc init

# Track a data file
dvc add data/train.csv
git add data/train.csv.dvc data/.gitignore
git commit -m "Track training data with DVC"

# Define a pipeline stage
dvc stage add -n train \
    -d src/train.py -d data/train.csv \
    -o models/model.pkl \
    python src/train.py

# Run the pipeline
dvc repro

# Push data to remote (S3 example)
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

### Exercise 3 — GitHub Actions for ML CI (YAML)
```yaml
# .github/workflows/ml-ci.yml
name: ML CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run data validation
        run: python scripts/validate_data.py
      - name: Train and evaluate model
        run: python scripts/train.py --output metrics.json
      - name: Check model performance threshold
        run: python scripts/check_metrics.py --min-accuracy 0.90
```

---

## Questions & Answers

**Q1: Why is reproducibility important in ML?**
> A model in a notebook cannot be deployed, audited, or improved systematically. Reproducibility means: given the same data and code, anyone can produce the same model. This requires tracking: (1) data version, (2) code version (git), (3) dependencies (requirements.txt or conda environment), and (4) hyperparameters and random seeds. MLflow and DVC handle 1, 3, and 4; git handles 2.

**Q2: What is the difference between MLflow and DVC?**
> MLflow focuses on experiment tracking — logging metrics, parameters, and model artifacts for comparison across runs. DVC focuses on data and pipeline versioning — managing large data files in storage and encoding the data processing pipeline as a reproducible DAG. They complement each other: use both for a complete MLOps workflow.

**Q3: What is a feature store?**
> A feature store is a centralized repository for computed features that can be shared across teams and ML models. It ensures: (1) features computed consistently for training and serving, (2) no duplication of feature engineering code, (3) point-in-time correctness for time-sensitive features. Examples: Feast, Tecton, Hopsworks.

**Q4: What does "CI/CD for ML" add compared to regular CI/CD?**
> Regular CI/CD tests code correctness. ML CI/CD also tests: data quality (schema checks, distribution drift), model quality (does the new model meet performance thresholds?), and training pipeline correctness (does the training run complete without errors?). A model that passes code tests but uses corrupted training data is still bad.

**Q5: What is the difference between online and batch predictions?**
> Batch prediction: run inference on a large dataset periodically, store results. Lower latency requirements, can use powerful hardware, results may be stale. Online prediction (real-time): serve predictions on demand via an API. Requires low latency, scalable infrastructure. Choose based on business requirements: recommendation emails (batch) vs fraud detection (online).
