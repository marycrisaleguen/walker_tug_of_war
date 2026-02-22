# Unit 3 — Monitoring & Maintenance

## Topics

### 1. Why Monitor ML Models?
- Models degrade over time: the world changes, but the model doesn't
- Silent failures: a model returns predictions without errors even when they're wrong
- Business impact: a degraded recommendation model silently reduces revenue

### 2. Data Drift (Covariate Shift)
- The distribution of input features changes between training and production
- Example: a model trained on summer data deployed in winter
- Detection: compare feature distributions (KS test, PSI — Population Stability Index)
- Response: retrain the model on recent data

### 3. Concept Drift
- The relationship between features and target changes over time
- Example: a fraud model becomes outdated as fraudsters adapt
- Harder to detect (requires monitoring predictions AND outcomes)
- Types: sudden drift, gradual drift, seasonal drift

### 4. Model Performance Monitoring
- Track prediction metrics over time: accuracy, AUC, precision, recall
- Requires ground truth labels — often delayed (can't always get them immediately)
- Use proxy metrics: outcome feedback, user actions, downstream business KPIs
- Set up alerts when metrics drop below thresholds

### 5. Monitoring Infrastructure
- Tools: Prometheus + Grafana (metrics), Evidently AI, Fiddler, Arize AI
- Log every prediction: features, prediction, timestamp, optionally ground truth
- Dashboards: real-time and batch monitoring views

### 6. Retraining Strategies
- **Scheduled retraining**: retrain every week/month on new data
- **Triggered retraining**: retrain when drift is detected or performance drops
- **Online learning**: update model continuously with each new sample (not always practical)
- **Champion-Challenger**: new model trained in parallel; promoted only if better than champion

### 7. A/B Testing Models
- Split traffic between two model versions
- Measure business impact, not just ML metrics
- Run for sufficient duration to reach statistical significance
- Shadow mode: new model receives traffic but doesn't serve predictions (safe testing)

---

## Exercises

### Exercise 1 — Data Drift Detection
```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)
# Simulate training data distribution
train_feature = np.random.normal(50, 10, 1000)
# Simulate production data (drifted — mean shifted by 10)
prod_feature  = np.random.normal(60, 10, 500)

# KS Test
ks_stat, p_value = stats.ks_2samp(train_feature, prod_feature)
print(f"KS Statistic: {ks_stat:.4f}, p-value: {p_value:.6f}")
print(f"Drift detected: {p_value < 0.05}")

# Population Stability Index (PSI)
def compute_psi(expected, actual, n_bins=10):
    bins = np.percentile(expected, np.linspace(0, 100, n_bins+1))
    bins[0] -= 0.01; bins[-1] += 0.01
    exp_counts = np.histogram(expected, bins)[0] / len(expected)
    act_counts = np.histogram(actual, bins)[0] / len(actual)
    exp_counts = np.clip(exp_counts, 1e-8, None)
    act_counts = np.clip(act_counts, 1e-8, None)
    psi = np.sum((act_counts - exp_counts) * np.log(act_counts / exp_counts))
    return psi

psi = compute_psi(train_feature, prod_feature)
print(f"PSI: {psi:.4f} — {'No drift' if psi < 0.1 else 'Moderate drift' if psi < 0.25 else 'Significant drift'}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(train_feature, bins=30, alpha=0.5, label="Train", density=True)
ax.hist(prod_feature,  bins=30, alpha=0.5, label="Production", density=True)
ax.set_title("Feature Distribution: Train vs Production")
ax.legend(); plt.tight_layout(); plt.show()
```

### Exercise 2 — Prediction Logging Scaffold
```python
import json
import time
from datetime import datetime
import numpy as np

class ModelLogger:
    """Logs predictions for monitoring and retraining."""
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, features, prediction, probability, model_version):
        record = {
            "timestamp":     datetime.utcnow().isoformat(),
            "features":      features.tolist(),
            "prediction":    int(prediction),
            "probability":   float(probability),
            "model_version": model_version
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load_logs(self):
        records = []
        with open(self.log_file) as f:
            for line in f:
                records.append(json.loads(line))
        return records

# Usage
logger = ModelLogger("/tmp/predictions.jsonl")
np.random.seed(0)
for _ in range(10):
    features = np.random.randn(5)
    logger.log(features, prediction=int(features[0] > 0), probability=0.7, model_version="v1.2")

logs = logger.load_logs()
print(f"Logged {len(logs)} predictions")
print("First log entry:", logs[0])
```

---

## Questions & Answers

**Q1: How do I detect model degradation without ground truth labels?**
> Use proxy metrics: (1) track prediction distribution — if the model's positive rate changes dramatically, something changed. (2) Monitor input feature distributions for data drift. (3) Track business outcomes (CTR, conversion, user ratings) as proxies for model quality. (4) If you can get delayed labels (e.g., fraud confirmed after review), track model performance on those.

**Q2: What is Population Stability Index (PSI) and how do I interpret it?**
> PSI measures how much the distribution of a variable has shifted between two periods. PSI < 0.1: no significant drift. 0.1 ≤ PSI < 0.25: moderate drift (investigate). PSI ≥ 0.25: significant drift (action required, consider retraining). PSI is widely used in financial ML for monitoring scorecards.

**Q3: When should I retrain a model?**
> Retrain when: (1) model performance metrics drop below business thresholds, (2) significant data drift detected (PSI > 0.25), (3) a new batch of labeled data is available that could improve the model, (4) business requirements change (new target, new features). For non-critical models, weekly/monthly scheduled retraining is often sufficient.

**Q4: What is shadow mode deployment?**
> The new model receives a copy of all production traffic and generates predictions, but those predictions are not served to users. The responses are logged for offline evaluation. This lets you safely evaluate the new model's behavior on real production data before fully deploying it, catching issues that don't appear in test sets.

**Q5: What is the difference between A/B testing and Champion-Challenger?**
> Both compare two model versions on live traffic. A/B testing randomly assigns users to two groups and measures business metrics over time. Champion-Challenger is a specific RL-like approach where the champion (current best) serves most traffic (e.g., 90%) and challengers receive small portions. The challenger is promoted only if it beats the champion on defined criteria over a sufficient period.
