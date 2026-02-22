# Unit 2 — Feature Engineering

## Topics

### 1. What is Feature Engineering?
- Creating new input features from raw data to improve model performance
- Domain knowledge is the most powerful feature engineering tool
- "Feature engineering is the process of using domain knowledge to extract features from raw data"

### 2. Encoding Categorical Variables
- **Ordinal encoding**: map categories to integers (preserves order)
- **One-hot encoding (OHE)**: binary dummy variable per category; drop one to avoid multicollinearity
- **Target encoding**: replace category with mean of target variable (risk of leakage!)
- **Frequency encoding**: replace category with its frequency in the dataset

### 3. Binning / Discretization
- Converting continuous features into bins (e.g., age → age groups)
- `pd.cut()` (equal width), `pd.qcut()` (equal frequency)
- Useful when relationship between feature and target is non-monotonic

### 4. Polynomial and Interaction Features
- `PolynomialFeatures(degree=2)` from sklearn
- Creates x₁², x₂², x₁·x₂ etc.
- Allows linear models to capture non-linear patterns
- Can explode feature count — use carefully

### 5. Date and Time Features
- Year, month, day, day-of-week, hour, is_weekend, days_since_event
- Cyclical encoding for periodic features: sin(2π·month/12), cos(2π·month/12)

### 6. Aggregation Features
- Group-level statistics: mean/std/min/max of a feature per group
- Example: customer_avg_purchase, store_item_sales_rank

### 7. Text Features (Brief)
- Bag of words, TF-IDF (covered in Tier 13)
- Character counts, word counts, presence of specific keywords

---

## Exercises

### Exercise 1 — One-Hot Encoding
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame({"color": ["red", "green", "blue", "red", "blue"],
                   "size":  ["S", "M", "L", "XL", "S"],
                   "price": [10, 20, 30, 25, 15]})

# Using pandas get_dummies
df_encoded = pd.get_dummies(df, columns=["color", "size"], drop_first=True)
print(df_encoded)
```

### Exercise 2 — Date Feature Extraction
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=10, freq="11D"),
    "sales": np.random.randint(100, 500, 10)
})
df["year"]       = df["date"].dt.year
df["month"]      = df["date"].dt.month
df["dayofweek"]  = df["date"].dt.dayofweek  # 0=Monday
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
# Cyclical encoding for month
df["month_sin"]  = np.sin(2*np.pi*df["month"]/12)
df["month_cos"]  = np.cos(2*np.pi*df["month"]/12)
print(df)
```

### Exercise 3 — Polynomial Features
```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[2, 3], [4, 5]])
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print("Original features:", X.shape)
print("Polynomial features:", X_poly.shape)
print("Feature names:", poly.get_feature_names_out(["x1", "x2"]))
```

### Exercise 4 — Group Aggregation
```python
import pandas as pd
import numpy as np

np.random.seed(1)
df = pd.DataFrame({
    "customer_id": np.random.choice(["A","B","C"], 20),
    "purchase":    np.random.randint(10, 500, 20)
})
# Add group-level feature: each customer's average purchase
df["avg_purchase"] = df.groupby("customer_id")["purchase"].transform("mean")
print(df.head(10))
```

---

## Questions & Answers

**Q1: Why should you drop one dummy variable in one-hot encoding?**
> With k categories, k one-hot columns are created. These k columns are perfectly collinear (they sum to 1 for every row). This causes a rank-deficient design matrix for linear models (the famous "dummy variable trap"). Dropping one column (the reference category) removes the perfect multicollinearity. Tree-based models are not affected.

**Q2: What is target encoding and why is it risky?**
> Target encoding replaces a category with the mean of the target variable for that category. It can capture strong signals but risks data leakage: if you encode using the full training set, you are using the target of each row to encode itself. Use cross-validation-based target encoding (available in `category_encoders`) or encode only on the out-of-fold data.

**Q3: Why should cyclical features (like month, hour) be sin/cos encoded?**
> A linear encoding treats December (12) as far from January (1), but they are adjacent on the calendar. Sin/cos encoding maps values onto a circle: December and January are close, and the model can learn circular relationships.

**Q4: What is the risk of adding too many polynomial or interaction features?**
> The feature space grows as O(n^d) where n is the original number of features and d is the polynomial degree. For d=2 and n=100 features, you get ~5,000 features. This dramatically increases model complexity and overfitting risk. Apply regularization (Ridge, Lasso) or feature selection when using polynomial features.

**Q5: How do you avoid leaking future information in time-series feature engineering?**
> When engineering features that look at historical data (rolling averages, cumulative sums), ensure the window only includes data up to, but not including, the current timestamp. Use `.shift(1)` before rolling operations to ensure the current row's value is not included in its own feature.
