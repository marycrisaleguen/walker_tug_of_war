# Unit 2 — Gradient Boosting

## Topics

### 1. Boosting Intuition
- Train models sequentially; each new model corrects the errors of the previous ones
- Unlike bagging (parallel), boosting is sequential
- Final prediction: weighted sum of all models

### 2. Gradient Boosting Algorithm
- Fit model F₁; compute residuals r = y − F₁(x)
- Fit model F₂ to residuals; set F₂(x) = F₁(x) + η·F₂(x)
- Repeat for M rounds
- Generalization of AdaBoost: fits new trees to pseudo-residuals = negative gradient of loss

### 3. XGBoost
- "Extreme Gradient Boosting" — regularized, efficient implementation
- Adds L1/L2 regularization on tree leaf weights
- Handles missing values natively
- Parallel construction of trees (column subsampling)
- `xgb.XGBClassifier`, `xgb.XGBRegressor`

### 4. LightGBM
- Leaf-wise (best-first) tree growth vs level-wise in XGBoost
- Much faster on large datasets; lower memory
- GOSS (gradient-based one-side sampling) and EFB (exclusive feature bundling)
- `lgb.LGBMClassifier`

### 5. Key Hyperparameters
- `n_estimators` / `num_boost_round`: number of trees
- `learning_rate` / `eta`: shrinkage per tree (smaller = more trees needed, but better generalization)
- `max_depth`: tree depth (3–6 is typical for boosting)
- `subsample`: fraction of samples per tree
- `colsample_bytree`: fraction of features per tree
- `reg_alpha` (L1), `reg_lambda` (L2): regularization

---

## Exercises

### Exercise 1 — XGBoost Classifier
```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", use_label_encoder=False, random_state=42
)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
print(f"Test AUC: {roc_auc_score(y_te, model.predict_proba(X_te)[:,1]):.4f}")
```

### Exercise 2 — Early Stopping
```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=4,
    eval_metric="logloss", use_label_encoder=False, random_state=42
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    early_stopping_rounds=20,
    verbose=False
)
print(f"Best iteration: {model.best_iteration}")
print(f"Best score: {model.best_score:.4f}")
```

---

## Questions & Answers

**Q1: What is the difference between boosting and bagging?**
> Bagging trains models in parallel on bootstrap samples; prediction is an average. Boosting trains models sequentially; each model focuses on examples the previous models got wrong. Bagging reduces variance; boosting reduces both bias and variance but is more prone to overfitting if the learning rate is too high or too many rounds are used.

**Q2: What does the learning rate do in gradient boosting?**
> Each new tree is multiplied by the learning rate η before being added: F ← F + η·fₙₑₓₜ. A small η shrinks the contribution of each tree, requiring more trees but usually resulting in better generalization. Common practice: set η = 0.05–0.1 and use early stopping.

**Q3: When should I use LightGBM vs XGBoost?**
> Both are excellent. LightGBM is generally faster on large datasets (millions of rows) due to its leaf-wise growth and sampling techniques. XGBoost has a slightly more interpretable tree-building process (level-wise) and is more battle-tested. For competitive ML, try both and cross-validate.

**Q4: What is early stopping and why is it important for gradient boosting?**
> Early stopping monitors validation error during training and stops when it stops improving (or starts getting worse). Without it, you need to tune `n_estimators` separately. With early stopping, you can set a large `n_estimators` and let the algorithm find the optimal number automatically.

**Q5: Why does gradient boosting fit to "pseudo-residuals" (negative gradients) rather than actual residuals?**
> For MSE loss, the negative gradient equals the residuals. But the framework generalizes: for other loss functions (log-loss, quantile loss, etc.), fitting to the negative gradient is mathematically equivalent to gradient descent in function space. This makes gradient boosting applicable to any differentiable loss function.
