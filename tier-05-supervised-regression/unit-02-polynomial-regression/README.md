# Unit 2 — Polynomial Regression

## Topics

### 1. Motivation
- Linear regression fits a straight line — but many relationships are curved
- Polynomial regression adds higher-degree terms: x, x², x³, …
- Still a linear model in the coefficients (linear in **w**), nonlinear in x

### 2. Feature Expansion
- Create new features: x → [x, x², x³, …, x^d]
- `PolynomialFeatures(degree=d)` from sklearn
- The same normal equation or gradient descent works on expanded features

### 3. Choosing the Degree
- Too low → underfitting (high bias)
- Too high → overfitting (high variance, wiggly curve)
- Use cross-validation or validation curves to select d

### 4. Overfitting and Underfitting
- **Underfitting** (high bias): training error is high; model is too simple
- **Overfitting** (high variance): training error is low, test error is high; model memorizes noise
- The "sweet spot" balances bias and variance

### 5. Validation Curves
- Plot train and validation error as a function of polynomial degree
- Training error decreases monotonically with degree
- Validation error has a U-shape — minimum is the optimal degree

---

## Exercises

### Exercise 1 — Polynomial Regression from Scratch
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

np.random.seed(0)
X = np.sort(np.random.uniform(-3, 3, 50)).reshape(-1, 1)
y = 0.5*X.ravel()**3 - X.ravel()**2 + 2 + np.random.randn(50)

degrees = [1, 3, 10]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

for ax, d in zip(axes, degrees):
    model = Pipeline([("poly", PolynomialFeatures(d)), ("lr", LinearRegression())])
    model.fit(X, y)
    ax.scatter(X, y, s=20, alpha=0.7)
    ax.plot(X_plot, model.predict(X_plot), color='red')
    ax.set_title(f"Degree {d}")
plt.tight_layout(); plt.show()
```

### Exercise 2 — Validation Curve (Degree Selection)
```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.sort(np.random.uniform(-3, 3, 100)).reshape(-1, 1)
y = np.sin(X.ravel()) + 0.1*np.random.randn(100)

model = Pipeline([("poly", PolynomialFeatures()), ("lr", LinearRegression())])
degrees = np.arange(1, 12)
train_scores, val_scores = validation_curve(
    model, X, y,
    param_name="poly__degree",
    param_range=degrees,
    cv=5, scoring="neg_mean_squared_error"
)
train_err = -train_scores.mean(axis=1)
val_err   = -val_scores.mean(axis=1)

plt.plot(degrees, train_err, label="Train MSE")
plt.plot(degrees, val_err,   label="Val MSE", linestyle='--')
plt.xlabel("Polynomial Degree"); plt.ylabel("MSE")
plt.title("Validation Curve"); plt.legend(); plt.show()
print(f"Best degree: {degrees[val_err.argmin()]}")
```

---

## Questions & Answers

**Q1: Is polynomial regression a linear model?**
> Yes! Once you expand x into [1, x, x², x³], the model ŷ = w₀ + w₁x + w₂x² + w₃x³ is linear in the weights **w**. The same OLS solution applies. "Linear" in "linear regression" refers to linearity in the parameters, not in the input features.

**Q2: How do I know if my model is underfitting or overfitting?**
> - **Underfitting**: training error AND validation error are both high.
> - **Overfitting**: training error is low but validation error is high (large gap).
> - **Good fit**: both training and validation errors are low and close together.
> Plot learning curves (error vs training set size) or validation curves (error vs model complexity) to diagnose.

**Q3: Why does increasing polynomial degree always decrease training error?**
> A higher-degree polynomial can fit the training data more closely, eventually passing through every training point exactly (interpolation). This makes training error approach zero. But the overly flexible model captures noise in the training data, performing poorly on unseen data.

**Q4: What is the relationship between polynomial degree and model variance?**
> Higher degree → more model complexity → higher variance. The model is more sensitive to small changes in training data. This is the bias-variance tradeoff: increasing degree reduces bias but increases variance.

**Q5: Can I use polynomial features with multiple input variables?**
> Yes. `PolynomialFeatures(degree=2)` with 2 input features [x₁, x₂] produces [1, x₁, x₂, x₁², x₁x₂, x₂²]. The number of features grows combinatorially: C(n+d, d) total features for n inputs and degree d. For n=10, d=3 you get 286 features. Use regularization to handle this.
