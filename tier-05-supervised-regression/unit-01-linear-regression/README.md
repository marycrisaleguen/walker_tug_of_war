# Unit 1 — Linear Regression

## Topics

### 1. The Regression Problem
- Goal: predict a continuous output y from inputs **x**
- Model: ŷ = **w**ᵀ**x** + b = w₁x₁ + w₂x₂ + … + wₙxₙ + b
- Matrix form: **ŷ** = X**w** (with bias absorbed into X via a column of 1s)

### 2. Cost Function: Mean Squared Error
- MSE = (1/n) Σ(yᵢ − ŷᵢ)²
- Why squared? Penalizes large errors more, is differentiable, has unique minimum
- MAE alternative: less sensitive to outliers but not differentiable at 0

### 3. Closed-Form Solution (Normal Equation)
- **w** = (XᵀX)⁻¹Xᵀ**y**
- Exact solution — no iteration
- Limitation: O(n_features³) inversion — slow for many features

### 4. Gradient Descent Solution
- Compute gradient: ∇L = (−2/n) Xᵀ(**y** − X**w**)
- Update: **w** ← **w** − α · ∇L
- Stochastic GD (SGD): update using one sample; mini-batch GD: use a batch

### 5. Assumptions of Linear Regression
- Linearity: E[y|x] is linear in x
- Independence of errors
- Homoscedasticity: constant variance of residuals
- Normality of residuals (needed for inference, not for prediction)
- No multicollinearity

### 6. Evaluating Regression Models
- R² = 1 − SSres/SStot — proportion of variance explained (0 to 1; higher is better)
- RMSE = √MSE — in same units as y
- MAE = mean(|yᵢ − ŷᵢ|)

---

## Exercises

### Exercise 1 — Linear Regression from Scratch
```python
import numpy as np

np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
true_w = np.array([3.0, -1.5])
y = X @ true_w + 2.0 + 0.5 * np.random.randn(n)  # intercept = 2

# Add bias column
X_b = np.column_stack([np.ones(n), X])

# Normal equation
w_hat = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Estimated: intercept={w_hat[0]:.3f}, w={w_hat[1:]}")
print(f"True:      intercept=2.000, w={true_w}")
```

### Exercise 2 — sklearn Linear Regression
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)
X = np.random.randn(200, 3)
y = 1.5*X[:,0] - 2.0*X[:,1] + 0.5*X[:,2] + np.random.randn(200)*0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
print(f"Coefficients: {model.coef_}")
```

### Exercise 3 — Residual Analysis
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(1)
X = np.random.randn(100, 1)
y = 2*X.ravel() + np.random.randn(100)

model = LinearRegression().fit(X, y)
residuals = y - model.predict(X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.scatter(model.predict(X), residuals, alpha=0.7)
ax1.axhline(0, color='red', linestyle='--')
ax1.set_xlabel("Predicted"); ax1.set_ylabel("Residuals")
ax1.set_title("Residuals vs Fitted")

ax2.hist(residuals, bins=20, edgecolor='black')
ax2.set_title("Residual Distribution")
plt.tight_layout(); plt.show()
```

---

## Questions & Answers

**Q1: What is the difference between linear regression and a linear model?**
> A linear regression model is a specific linear model where we minimize the MSE loss. A "linear model" is any model whose predictions are a linear function of the input features. Linear regression, logistic regression (before the sigmoid), SVMs (linear kernel), and the final layer of a neural network are all linear models.

**Q2: When should I use the Normal Equation vs Gradient Descent?**
> Normal Equation is exact and requires no hyperparameters, but is O(p³) in the number of features — slow for p > ~10,000. Gradient Descent scales to millions of features and is used when the dataset or feature space is large. For small datasets and few features, the Normal Equation is preferred.

**Q3: What does R² = 0.85 mean?**
> The model explains 85% of the variance in the target variable. The remaining 15% is unexplained by the features. R² = 1 is a perfect fit; R² = 0 means the model does no better than predicting the mean; R² < 0 means the model is worse than predicting the mean.

**Q4: What is multicollinearity and how do I detect it?**
> Multicollinearity occurs when two or more features are highly correlated. It makes coefficients unstable and hard to interpret — small data changes cause large coefficient swings. Detect it with the Variance Inflation Factor (VIF): VIF > 5–10 indicates a problem. Solutions: remove one of the correlated features, use PCA, or apply Ridge regression.

**Q5: Why does linear regression assume normally distributed residuals?**
> The normality assumption is needed for hypothesis testing (t-tests on coefficients) and confidence intervals. It is NOT required for obtaining unbiased coefficient estimates (that only needs linearity + independence). For pure prediction, you can often ignore this assumption.
