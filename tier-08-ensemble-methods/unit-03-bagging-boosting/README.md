# Unit 3 — Bagging, Boosting & Stacking

## Topics

### 1. Bagging (Bootstrap Aggregating)
- Train B estimators on B bootstrap samples; aggregate predictions
- Reduces variance: averaging reduces noise independent of the base estimator
- `BaggingClassifier` / `BaggingRegressor` in sklearn

### 2. Boosting (Recap)
- Sequential training: each model focuses on the errors of the previous
- Reduces bias and variance
- Examples: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost

### 3. AdaBoost
- Assigns higher weights to misclassified samples
- Each weak learner (shallow tree) is trained on weighted data
- Final prediction: weighted majority vote
- `AdaBoostClassifier` in sklearn

### 4. Stacking (Stacked Generalization)
- Layer 1: train k diverse base learners on training data
- Layer 2 (meta-learner): train a model on the out-of-fold predictions from layer 1
- Often achieves better results than any single model
- `StackingClassifier` / `StackingRegressor` in sklearn

### 5. Blending
- Simpler variant of stacking: hold out a fixed blend set instead of using CV
- Layer 1 models trained on training set, predict on blend set
- Meta-learner trained on blend set predictions
- Risk: less data available and blend set may not be representative

### 6. When to Use Each
| Method | Best For |
|--------|----------|
| Bagging | High-variance base models (deep trees) |
| Boosting | Achieving the best performance on tabular data |
| Stacking | Squeezing out last performance gains in competitions |

---

## Exercises

### Exercise 1 — AdaBoost
```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

X, y = load_breast_cancer(return_X_y=True)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # "stumps"
    n_estimators=200, learning_rate=1.0, random_state=42
)
scores = cross_val_score(ada, X, y, cv=5, scoring="roc_auc")
print(f"AdaBoost CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Exercise 2 — Stacking
```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = load_wine(return_X_y=True)

base_learners = [
    ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
    ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
    ("dt",  DecisionTreeClassifier(max_depth=4, random_state=42)),
]
stacker = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)
scores = cross_val_score(stacker, X, y, cv=5, scoring="accuracy")
print(f"Stacking CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## Questions & Answers

**Q1: What type of base learners work best for bagging?**
> High-variance, low-bias base learners — like deep, unpruned decision trees. Bagging's variance reduction is most effective when the base learners are unstable (small data changes cause large model changes). Low-variance models (e.g., linear regression) benefit very little from bagging.

**Q2: What is the role of diversity in ensemble methods?**
> Ensemble predictions are most accurate when the individual models' errors are uncorrelated — i.e., the models disagree on different samples. If all models make the same mistakes, averaging does not help. Diversity is achieved through different: training subsets (bagging), features (random forests), algorithms (stacking), hyperparameters.

**Q3: Does more data help boosting or bagging more?**
> Both benefit from more data, but boosting generally benefits more because it is trying to reduce bias (not just variance). Bagging's variance reduction is mainly about averaging independent errors, not about having more data per se. However, boosting is more prone to overfitting small datasets.

**Q4: What should the meta-learner be in a stacking ensemble?**
> A simple, regularized model — typically logistic regression or a shallow tree. The meta-learner's job is to learn how to combine base learner predictions, not to learn the problem from scratch. A complex meta-learner may overfit the out-of-fold predictions.

**Q5: Is stacking worth the extra complexity in production?**
> It depends on the use case. In competitions (Kaggle), stacking commonly yields 1–3% improvements. In production, the added complexity (multiple models, slower inference, harder maintenance) often makes it less attractive unless that small improvement has large business impact. Often, a well-tuned XGBoost or LightGBM single model is a better engineering choice.
