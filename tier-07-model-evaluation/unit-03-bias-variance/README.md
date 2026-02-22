# Unit 3 — Bias-Variance Tradeoff

## Topics

### 1. Sources of Error
- **Bias**: error from overly simplistic assumptions. A high-bias model underfits.
- **Variance**: error from sensitivity to fluctuations in training data. High-variance model overfits.
- **Noise** (irreducible error): inherent randomness in the data — cannot be reduced
- Total expected error = Bias² + Variance + Noise

### 2. Underfitting (High Bias)
- Model is too simple to capture the true relationship
- Signs: high training error AND high test error
- Fix: increase model complexity, add features, reduce regularization

### 3. Overfitting (High Variance)
- Model has memorized the training data including noise
- Signs: low training error but high test error (large gap)
- Fix: reduce model complexity, add regularization, collect more data, early stopping

### 4. The Bias-Variance Tradeoff
- As model complexity increases: bias ↓, variance ↑
- Optimal complexity minimizes total error (bias² + variance)
- There is no free lunch: you cannot always reduce both simultaneously

### 5. Learning Curves
- Plot train and validation error vs training set size
- High bias: both curves converge at a high error value
- High variance: large gap between train (low) and validation (high) curves
- More data helps high-variance models more than high-bias models

### 6. Diagnosing and Fixing
| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| High train & val error | High bias | Add complexity, better features |
| Low train, high val error | High variance | Regularize, more data, reduce features |
| Both errors are low | Good fit | ✓ |

---

## Exercises

### Exercise 1 — Learning Curves
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_s = StandardScaler().fit_transform(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for ax, model, title in [
    (ax1, LogisticRegression(max_iter=500),         "Logistic Regression (Low Variance)"),
    (ax2, DecisionTreeClassifier(max_depth=None),   "Decision Tree (High Variance)")
]:
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_s, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy"
    )
    ax.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    ax.plot(train_sizes, val_scores.mean(axis=1),   label="Validation", linestyle='--')
    ax.fill_between(train_sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    ax.set_title(title); ax.set_xlabel("Training Set Size"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.set_ylim(0.7, 1.02)
plt.tight_layout(); plt.show()
```

### Exercise 2 — Complexity Curve (Validation Curve)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_s = StandardScaler().fit_transform(X)

depths = np.arange(1, 21)
train_sc, val_sc = validation_curve(
    DecisionTreeClassifier(random_state=42), X_s, y,
    param_name="max_depth", param_range=depths,
    cv=5, scoring="accuracy"
)

plt.plot(depths, train_sc.mean(axis=1), label="Train")
plt.plot(depths, val_sc.mean(axis=1),   label="Validation", linestyle='--')
plt.fill_between(depths,
                 val_sc.mean(axis=1) - val_sc.std(axis=1),
                 val_sc.mean(axis=1) + val_sc.std(axis=1), alpha=0.2)
plt.xlabel("Max Depth"); plt.ylabel("Accuracy")
plt.title("Bias-Variance: Decision Tree Depth")
plt.legend(); plt.show()
```

---

## Questions & Answers

**Q1: Can I simultaneously reduce both bias and variance?**
> Sometimes — by collecting more data, using a better feature representation, or using an ensemble method that combines many weak learners (see Tier 8). In general, for a fixed algorithm and dataset size, reducing one increases the other. The goal is to find the optimal tradeoff that minimizes total test error.

**Q2: How does adding more training data affect bias and variance?**
> More data reduces variance (the model has more information and is less sensitive to any single training sample) but does not change bias. A high-bias model will not improve much with more data. A high-variance model will show significant improvement.

**Q3: How does regularization affect bias and variance?**
> Regularization reduces variance by constraining model complexity but introduces bias (it prevents the model from fitting perfectly). Stronger regularization → higher bias, lower variance. The regularization strength is a bias-variance tradeoff hyperparameter.

**Q4: What is the "double descent" phenomenon?**
> For very large models (like overparameterized neural networks), the test error follows a "double descent" curve: it decreases, then increases (classical overfitting), then decreases again as model size grows further. This challenges the traditional U-shaped bias-variance curve and is an active area of research.

**Q5: What does a learning curve with a persistent gap tell you?**
> If train accuracy is high and validation accuracy is lower with a persistent gap even as training set size increases, the model has high variance (overfitting). The gap narrows with more data but may never close completely. The fix is to reduce model complexity or apply regularization.
