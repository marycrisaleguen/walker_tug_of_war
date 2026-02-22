# Unit 1 — Random Forests

## Topics

### 1. Bootstrap Aggregation (Bagging)
- Train many models on different bootstrap samples (sampling with replacement)
- Combine by majority vote (classification) or averaging (regression)
- Reduces variance without increasing bias significantly

### 2. Random Forests = Bagging + Feature Subsampling
- At each split, only a random subset of features is considered: sqrt(p) for classification, p/3 for regression
- Feature subsampling decorrelates the trees — trees become more independent
- `RandomForestClassifier` / `RandomForestRegressor` in sklearn

### 3. Key Hyperparameters
- `n_estimators`: number of trees (more is better up to a point)
- `max_features`: size of feature subset at each split
- `max_depth`: maximum tree depth (None = fully grown)
- `min_samples_leaf`: minimum samples in a leaf
- `bootstrap`: whether to use bootstrap sampling

### 4. Out-of-Bag (OOB) Evaluation
- Each tree trains on ~63% of data; ~37% "out-of-bag" samples serve as a free validation set
- `oob_score=True` gives an unbiased estimate of generalization error without CV

### 5. Feature Importance
- `feature_importances_`: mean decrease in impurity across all trees
- More stable than single-tree importance
- Also use permutation importance for a model-agnostic, less-biased estimate

---

## Exercises

### Exercise 1 — Random Forest Classifier
```python
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt

X, y = load_wine(return_X_y=True)
feature_names = load_wine().feature_names
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_tr, y_tr)

print(f"Test accuracy: {accuracy_score(y_te, rf.predict(X_te)):.4f}")
print(f"OOB score:     {rf.oob_score_:.4f}")

# Feature importance
fi = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)
fi.plot(kind='barh', figsize=(8, 6))
plt.title("Random Forest Feature Importance"); plt.tight_layout(); plt.show()
```

### Exercise 2 — n_estimators Effect
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)
n_trees = [1, 5, 10, 25, 50, 100, 200, 500]
cv_scores = [cross_val_score(RandomForestClassifier(n, random_state=42), X, y, cv=5).mean()
             for n in n_trees]

plt.semilogx(n_trees, cv_scores, marker='o')
plt.xlabel("Number of Trees"); plt.ylabel("CV Accuracy")
plt.title("Random Forest: n_estimators vs Accuracy"); plt.grid(True); plt.show()
```

---

## Questions & Answers

**Q1: Why does Random Forest outperform a single decision tree?**
> A single deep tree has high variance — it overfits training data. Random Forest trains many decorrelated trees (via bootstrap sampling and random feature selection) and averages their predictions. Averaging reduces variance while maintaining low bias from each individual tree.

**Q2: What is the difference between bagging and random forests?**
> Bagging is the general technique of training models on bootstrap samples and averaging. A "Bagged Decision Tree" uses all features at each split. Random Forest adds feature subsampling at each split, further decorrelating the trees and improving generalization.

**Q3: How many trees are enough?**
> Error decreases as you add trees and levels off. For most problems, 100–500 trees are sufficient. Adding more trees never hurts accuracy (only compute cost). Use the OOB error vs n_estimators curve to see when it plateaus.

**Q4: What is permutation feature importance and why prefer it over MDI?**
> Permutation importance works by randomly shuffling one feature's values and measuring how much the model's validation score decreases. Features that are truly important will cause a large drop when shuffled. MDI (Mean Decrease in Impurity, the default `feature_importances_`) is biased toward high-cardinality and correlated features, while permutation importance is model-agnostic and unbiased.

**Q5: Can random forests overfit?**
> Yes, especially with very deep trees, very small datasets, or very noisy labels. However, they are much more resistant to overfitting than single trees because of the variance reduction from averaging. Tuning `max_depth` and `min_samples_leaf` can further prevent overfitting.
