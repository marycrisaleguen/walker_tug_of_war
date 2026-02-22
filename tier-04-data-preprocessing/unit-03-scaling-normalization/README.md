# Unit 3 — Scaling & Normalization

## Topics

### 1. Why Scaling Matters
- Algorithms that use distance (KNN, SVM, K-Means) or gradient descent (linear/logistic regression, neural networks) are sensitive to feature scales
- A feature with range [0, 1000] dominates one with range [0, 1]
- Tree-based models (Random Forest, XGBoost) are scale-invariant

### 2. Min-Max Scaling (Normalization)
- Formula: x' = (x − x_min) / (x_max − x_min) → range [0, 1]
- `MinMaxScaler` from sklearn
- Sensitive to outliers (they compress the range)

### 3. Standardization (Z-score Normalization)
- Formula: x' = (x − μ) / σ → mean 0, std 1
- `StandardScaler` from sklearn
- Preferred when distribution is approximately Normal
- More robust to outliers than Min-Max

### 4. Robust Scaling
- Formula: x' = (x − Q2) / IQR → uses median and interquartile range
- `RobustScaler` from sklearn
- Best when data has many outliers

### 5. Log and Power Transforms
- `np.log1p(x)` or `np.log(x+1)`: compress right-skewed distributions
- `np.sqrt(x)`: milder compression
- `PowerTransformer` (Box-Cox, Yeo-Johnson): make distributions more Gaussian

### 6. Fit-Transform Pattern
- **Always** fit scalers on the training set only
- Use `scaler.transform()` on validation and test sets
- Prevents data leakage of test statistics into the model

---

## Exercises

### Exercise 1 — Compare Scalers
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

np.random.seed(42)
data = np.concatenate([np.random.normal(50, 10, 97), [200, 250, -100]])
data = data.reshape(-1, 1)

scalers = {
    "MinMax":   MinMaxScaler(),
    "Standard": StandardScaler(),
    "Robust":   RobustScaler()
}
results = {"Original": data.flatten()}
for name, scaler in scalers.items():
    results[name] = scaler.fit_transform(data).flatten()

df = pd.DataFrame(results)
print(df.describe().round(3))
```

### Exercise 2 — Correct Train/Test Scaling
```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(0)
X = np.random.randn(200, 3) * np.array([100, 1, 0.01])

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_scaled  = scaler.transform(X_test)         # transform (no fit) on test

print("Train mean:", X_train_scaled.mean(axis=0).round(6))
print("Train std:",  X_train_scaled.std(axis=0).round(6))
print("Test mean:",  X_test_scaled.mean(axis=0).round(3))  # close but not exactly 0
```

### Exercise 3 — Log Transform for Skewed Data
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
income = np.random.exponential(50000, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.hist(income, bins=50, edgecolor='black')
ax1.set_title("Raw Income (right-skewed)")

ax2.hist(np.log1p(income), bins=50, edgecolor='black', color='green')
ax2.set_title("Log-Transformed Income (more Gaussian)")
plt.tight_layout(); plt.show()

print(f"Skewness before: {pd.Series(income).skew():.2f}")
import pandas as pd
print(f"Skewness after:  {pd.Series(np.log1p(income)).skew():.2f}")
```

---

## Questions & Answers

**Q1: Does scaling affect tree-based models like Random Forest?**
> No. Decision trees split on thresholds in a single feature at a time. Whether that feature is on its original scale or standardized, the split will find the equivalent threshold. Feature importance and predictions are identical. You can skip scaling for XGBoost, Random Forest, and other tree ensembles.

**Q2: What is data leakage in the context of scaling?**
> If you fit the scaler on the combined train+test data, the test set statistics (mean, std) influence the scaling of the training data. This causes the model to implicitly "see" test information during training, producing inflated performance metrics that do not generalize. Always `fit` on train, `transform` on test.

**Q3: When should I use Min-Max vs Standard scaling?**
> Use **Min-Max** when you need output in [0, 1] (e.g., neural network inputs with sigmoid activation, image pixel values). Use **Standard** (Z-score) as the default for most ML algorithms with gradient descent or distance metrics. Use **Robust** when your data has significant outliers.

**Q4: What does the Box-Cox transform do?**
> Box-Cox is a family of power transforms: x' = (x^λ − 1) / λ for λ ≠ 0, and ln(x) for λ = 0. The optimal λ is estimated from data to best Gaussianize the distribution. It requires strictly positive values. The Yeo-Johnson transform is a generalization that works on any real-valued data.

**Q5: Should I scale target variable (y) in regression?**
> It depends on the algorithm. For linear regression and neural networks predicting continuous targets, scaling y can help with gradient descent convergence (especially for very large or very small target values). Remember to inverse-transform predictions at the end: `y_pred = scaler_y.inverse_transform(y_pred_scaled)`. Tree-based models do not require y scaling.
