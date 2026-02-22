# Unit 1 — Cross-Validation

## Topics

### 1. The Need for Validation
- A model that has seen the test set during training gives optimistic performance estimates
- A held-out test set estimates real-world performance
- Validation set (from training data) used for hyperparameter tuning

### 2. Train / Validation / Test Split
- Typical split: 60% train / 20% validation / 20% test
- Never touch the test set until final evaluation
- For small datasets, cross-validation replaces the separate validation set

### 3. K-Fold Cross-Validation
- Split training data into K folds; rotate through them as validation sets
- Train K models, average the validation scores
- Common K: 5 or 10; higher K = lower bias, higher variance, more compute

### 4. Stratified K-Fold
- Preserves class proportions in each fold — important for imbalanced datasets
- `StratifiedKFold` in sklearn
- Always use for classification problems

### 5. Leave-One-Out Cross-Validation (LOOCV)
- K = n: use every single sample as the validation set once
- Unbiased but high variance and expensive for large datasets
- Useful for very small datasets

### 6. Time-Series Cross-Validation
- Cannot shuffle time series — future must not leak into past
- `TimeSeriesSplit`: expanding window or sliding window
- Walk-forward validation

### 7. Nested Cross-Validation
- Outer loop: estimate generalization error
- Inner loop: hyperparameter selection
- Produces unbiased performance estimates for models tuned with cross-validation

---

## Exercises

### Exercise 1 — K-Fold vs Stratified K-Fold
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_s = StandardScaler().fit_transform(X)
model = LogisticRegression(max_iter=500)

kf_scores  = cross_val_score(model, X_s, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
skf_scores = cross_val_score(model, X_s, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

print(f"K-Fold:            {kf_scores.mean():.4f} ± {kf_scores.std():.4f}")
print(f"Stratified K-Fold: {skf_scores.mean():.4f} ± {skf_scores.std():.4f}")
```

### Exercise 2 — Hyperparameter Selection with CV
```python
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X_s = StandardScaler().fit_transform(X)

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid={"n_neighbors": range(1, 21)},
    cv=5, scoring="accuracy"
)
grid.fit(X_s, y)
print(f"Best K: {grid.best_params_['n_neighbors']}, CV accuracy: {grid.best_score_:.4f}")
```

### Exercise 3 — Time Series Split
```python
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

n = 100
X = np.arange(n).reshape(-1, 1)
tscv = TimeSeriesSplit(n_splits=5)

fig, ax = plt.subplots(figsize=(10, 4))
for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
    ax.scatter(train_idx, [i]*len(train_idx), c='blue', marker='|', s=100)
    ax.scatter(val_idx,  [i]*len(val_idx),   c='red',  marker='|', s=100)
ax.set_xlabel("Sample Index"); ax.set_ylabel("Fold")
ax.set_title("Time Series Cross-Validation Splits")
plt.tight_layout(); plt.show()
```

---

## Questions & Answers

**Q1: Why do we need a separate test set if we use cross-validation?**
> Cross-validation is used to select and tune models (hyperparameter search). Once you pick the best hyperparameters based on CV scores, those hyperparameters are selected to maximize validation performance — they overfit slightly to the folds used in CV. The separate test set, never seen during CV, gives an unbiased estimate of true generalization.

**Q2: How does stratification help with imbalanced datasets?**
> Without stratification, some folds might happen to contain only majority-class examples, making the fold's validation score unreliable. Stratification ensures each fold has approximately the same class ratio as the full dataset, giving more stable and representative CV scores.

**Q3: What is the difference between cross-validation score and test set score?**
> CV score is the average over the validation folds — it estimates how well the model generalizes to data not used for training. The test score is evaluated on data held out before the entire experiment. If the test score is much lower than the CV score, hyperparameters may be overfit to the CV folds (especially with a small dataset and many hyperparameters).

**Q4: What is data leakage in cross-validation?**
> Data leakage occurs when preprocessing steps (e.g., scaling, imputation) are fit on data that includes the validation fold. The correct approach is to put preprocessing inside a `Pipeline` — sklearn's CV then fits preprocessing only on the training fold and applies it to the validation fold.

**Q5: When should I use LOOCV over K-Fold?**
> Use LOOCV when your dataset is very small (<50 samples) and you cannot afford to remove many samples for a validation fold. It provides the most data for training. Its main drawbacks are high computational cost (n models trained) and high variance in the resulting score estimate.
