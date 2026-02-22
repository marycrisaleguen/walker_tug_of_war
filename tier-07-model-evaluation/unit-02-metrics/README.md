# Unit 2 — Evaluation Metrics

## Topics

### 1. Classification Metrics
- **Accuracy**: (TP + TN) / n — misleading with imbalanced classes
- **Precision**: TP / (TP + FP) — "of all predicted positives, how many are correct?"
- **Recall (Sensitivity)**: TP / (TP + FN) — "of all actual positives, how many did we find?"
- **F1 Score**: 2 · (Precision · Recall) / (Precision + Recall) — harmonic mean
- **Confusion Matrix**: 2×2 (binary) or k×k (multi-class) table of TP/FP/TN/FN

### 2. Precision vs Recall Tradeoff
- Lowering the classification threshold → higher recall, lower precision
- Raising the threshold → higher precision, lower recall
- Choose based on business cost: missing a cancer (FN) vs false alarm (FP)
- PR Curve: precision vs recall at all thresholds

### 3. ROC-AUC
- ROC curve: True Positive Rate vs False Positive Rate at all thresholds
- AUC (Area Under the Curve): probability that the model ranks a positive higher than a negative
- AUC = 0.5: random classifier; AUC = 1.0: perfect
- More informative than accuracy for imbalanced datasets

### 4. Regression Metrics
- **MAE** (Mean Absolute Error): mean(|yᵢ − ŷᵢ|) — robust to outliers, interpretable
- **MSE** (Mean Squared Error): mean((yᵢ − ŷᵢ)²) — penalizes large errors
- **RMSE**: √MSE — same units as y
- **R²** (Coefficient of Determination): 1 − SSres/SStot — proportion of variance explained
- **MAPE** (Mean Absolute Percentage Error): useful when targets span multiple orders of magnitude

### 5. Multi-Class Metrics
- Macro-average: compute metric per class, average equally
- Weighted-average: compute metric per class, weight by support (sample count)
- Micro-average: aggregate TP/FP/FN across classes before computing

---

## Exercises

### Exercise 1 — Binary Classification Metrics
```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
X_tr = StandardScaler().fit_transform(X_tr)
model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
X_te_s = StandardScaler().fit(X_tr).transform(X_te)
y_pred = model.predict(X_te_s)
y_proba = model.predict_proba(X_te_s)[:, 1]

print(f"Accuracy:  {accuracy_score(y_te, y_pred):.4f}")
print(f"Precision: {precision_score(y_te, y_pred):.4f}")
print(f"Recall:    {recall_score(y_te, y_pred):.4f}")
print(f"F1:        {f1_score(y_te, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_te, y_proba):.4f}")

ConfusionMatrixDisplay(confusion_matrix(y_te, y_pred)).plot()
plt.title("Confusion Matrix"); plt.show()
```

### Exercise 2 — ROC Curve
```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_te, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1], 'k--', label="Random")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(); plt.show()
```

### Exercise 3 — Threshold Tuning
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = np.linspace(0.01, 0.99, 99)
precisions, recalls, f1s = [], [], []
for t in thresholds:
    pred = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_te, pred, zero_division=0))
    recalls.append(recall_score(y_te, pred, zero_division=0))
    f1s.append(f1_score(y_te, pred, zero_division=0))

best_t = thresholds[np.argmax(f1s)]
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls,    label="Recall")
plt.plot(thresholds, f1s,        label="F1")
plt.axvline(best_t, linestyle='--', color='gray', label=f"Best t={best_t:.2f}")
plt.xlabel("Threshold"); plt.legend(); plt.title("Metrics vs Threshold"); plt.show()
```

---

## Questions & Answers

**Q1: When should I use F1 score instead of accuracy?**
> Use F1 (or precision/recall separately) when the classes are imbalanced. A classifier that labels everything as the majority class can get 90% accuracy on a 90/10 split — but F1 reveals the zero precision/recall on the minority class.

**Q2: What does AUC = 0.85 mean?**
> If you pick a random positive example and a random negative example, there is an 85% chance the model assigns a higher probability to the positive example. It measures the model's discriminative ability independent of the classification threshold.

**Q3: When is recall more important than precision?**
> When false negatives are costly: cancer screening (missing a cancer is worse than a false alarm), fraud detection (missing fraud is worse than investigating a legitimate transaction), or safety-critical systems. When false positives are costly (e.g., spam filter — don't want to delete legitimate emails), precision is more important.

**Q4: What is the difference between macro and weighted F1?**
> Macro-F1 computes F1 for each class independently and averages them equally — all classes contribute equally regardless of size. Weighted-F1 weights each class's F1 by its support (frequency). For imbalanced classes, macro-F1 better reflects minority class performance; weighted-F1 is closer to overall performance.

**Q5: Can RMSE be negative?**
> No. RMSE = √MSE ≥ 0 always. R² can be negative (a model worse than predicting the mean). MAE and MSE are always non-negative. If you see negative RMSE from sklearn, check if you used `neg_mean_squared_error` scoring — sklearn negates it so that "higher is better" convention holds in GridSearchCV.
