# Unit 3 — Regularization

## Topics

### 1. Why Regularization?
- Prevents overfitting by penalizing large coefficient values
- Forces the model to be simpler and more generalizable
- Especially important when: many features, correlated features, small training set

### 2. Ridge Regression (L2)
- Objective: min ‖**y** − X**w**‖² + λ‖**w**‖₂²
- Shrinks all coefficients toward zero, but rarely to exactly zero
- Closed-form: **w** = (XᵀX + λI)⁻¹Xᵀ**y**
- Hyperparameter α (λ): `Ridge(alpha=1.0)` in sklearn

### 3. Lasso Regression (L1)
- Objective: min ‖**y** − X**w**‖² + λ‖**w**‖₁
- Produces sparse solutions — drives some coefficients to exactly zero
- Acts as automatic feature selection
- `Lasso(alpha=1.0)` in sklearn

### 4. ElasticNet
- Combines L1 and L2: λ[ρ‖**w**‖₁ + (1−ρ)‖**w**‖₂²]
- `l1_ratio` controls the mix (0 = pure Ridge, 1 = pure Lasso)
- Useful when many correlated features; groups correlated features together

### 5. Choosing α (Regularization Strength)
- Too small: effectively no regularization (overfitting)
- Too large: underfitting (all coefficients → 0)
- Use `RidgeCV`, `LassoCV`, or `GridSearchCV` with cross-validation

### 6. Regularization Path
- Plot coefficient magnitudes as a function of α
- As α → ∞, all Lasso coefficients → 0
- Use `lasso_path()` to visualize feature selection order

---

## Exercises

### Exercise 1 — Ridge vs Lasso on Correlated Features
```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n, p = 100, 20
X = np.random.randn(n, p)
# Only 3 truly relevant features
true_w = np.zeros(p)
true_w[[0, 5, 10]] = [3.0, -2.0, 1.5]
y = X @ true_w + np.random.randn(n) * 0.5

X_s = StandardScaler().fit_transform(X)

for alpha in [0.01, 1.0, 10.0]:
    ridge = Ridge(alpha=alpha).fit(X_s, y)
    lasso = Lasso(alpha=alpha/n, max_iter=10000).fit(X_s, y)
    print(f"\nalpha={alpha}")
    print(f"  Ridge non-zero: {(abs(ridge.coef_) > 0.01).sum()}")
    print(f"  Lasso non-zero: {(abs(lasso.coef_) > 0.01).sum()}")
```

### Exercise 2 — Cross-Validated Ridge with RidgeCV
```python
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(0)
X = np.random.randn(200, 10)
y = X @ np.random.randn(10) + np.random.randn(200)*0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

alphas = np.logspace(-3, 3, 100)
ridge_cv = RidgeCV(alphas=alphas, cv=5).fit(X_train_s, y_train)
y_pred = ridge_cv.predict(X_test_s)

print(f"Best alpha: {ridge_cv.alpha_:.4f}")
print(f"Test R²:    {r2_score(y_test, y_pred):.4f}")
```

### Exercise 3 — Regularization Path for Lasso
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

np.random.seed(1)
n, p = 50, 10
X = np.random.randn(n, p)
true_w = np.zeros(p); true_w[[0,3]] = [2, -1.5]
y = X @ true_w + 0.2*np.random.randn(n)

X_s = StandardScaler().fit_transform(X)
alphas, coefs, _ = lasso_path(X_s, y, eps=1e-3)

plt.figure(figsize=(10, 5))
for i in range(p):
    plt.plot(np.log10(alphas), coefs[i], label=f"w{i}" if i in [0,3] else None)
plt.gca().invert_xaxis()
plt.xlabel("log10(alpha)"); plt.ylabel("Coefficient value")
plt.title("Lasso Regularization Path")
plt.legend(); plt.tight_layout(); plt.show()
```

---

## Questions & Answers

**Q1: When should I use Ridge vs Lasso?**
> Use **Ridge** when you believe most features are relevant but you want to reduce their magnitudes (no feature selection needed). Use **Lasso** when you want automatic feature selection — useful when you suspect many features are irrelevant. Use **ElasticNet** when features are correlated and you still want sparsity.

**Q2: Why does L1 regularization produce sparse solutions?**
> Geometrically, the L1 constraint set (diamond shape) has corners on the axes. The loss function contours are likely to meet the constraint set at a corner, where some coefficients are exactly zero. L2 (sphere) has no corners, so the optimal solution almost never has exactly-zero coefficients.

**Q3: Should I standardize features before applying regularization?**
> Yes — always. Regularization penalizes coefficient magnitudes equally. If features are on different scales, the penalty disproportionately affects features with larger natural units (since they require larger coefficients). Standardize first so the penalty is fair across all features.

**Q4: What happens when α → ∞?**
> All coefficients are shrunk to zero (only the intercept survives in most implementations). The model predicts the mean of y for every input — equivalent to the null model. Validation error will be high because the model is completely underfit.

**Q5: What is the difference between regularization in regression and early stopping in neural networks?**
> Both prevent overfitting, but via different mechanisms. L1/L2 regularization adds a penalty term to the loss function, constraining weight magnitude throughout training. Early stopping halts training before the model has had time to overfit, using a held-out validation set to monitor generalization. Early stopping is the most common "regularization" technique for deep neural networks.
