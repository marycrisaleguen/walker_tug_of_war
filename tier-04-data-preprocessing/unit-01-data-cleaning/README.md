# Unit 1 — Data Cleaning

## Topics

### 1. Understanding Your Data
- `df.info()`, `df.describe()`, `df.shape`
- Identifying column types: numeric, categorical, datetime, text
- Data validation: expected ranges, allowed categories, format checks

### 2. Handling Missing Values
- Types: MCAR (Missing Completely At Random), MAR, MNAR
- Strategies:
  - Drop rows: `df.dropna()`
  - Drop columns with too much missing: `df.dropna(axis=1, thresh=threshold)`
  - Impute with mean/median/mode: `SimpleImputer`
  - Impute with model predictions: `IterativeImputer`

### 3. Handling Outliers
- Detection: Z-score (>3σ), IQR method (< Q1−1.5·IQR or > Q3+1.5·IQR)
- Strategies: remove, cap/clip, transform (log), or leave as-is if meaningful

### 4. Duplicate Rows
- Detection: `df.duplicated()`
- Removal: `df.drop_duplicates()`
- Checking for near-duplicates with fuzzy matching

### 5. Data Type Corrections
- Converting strings to numbers: `pd.to_numeric(df["col"], errors="coerce")`
- Parsing dates: `pd.to_datetime(df["date"])`
- Categorical dtype: `df["col"].astype("category")`

### 6. Inconsistent Values
- Standardizing strings: `.str.strip()`, `.str.lower()`
- Correcting typos with dictionaries: `df["col"].replace({"NY": "New York"})`

---

## Exercises

### Exercise 1 — Detect and Handle Missing Values
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

np.random.seed(42)
df = pd.DataFrame({
    "age":    [25, np.nan, 35, 28, np.nan, 42],
    "income": [50000, 60000, np.nan, 45000, 75000, np.nan],
    "city":   ["NY", "LA", np.nan, "NY", "Chicago", "LA"]
})
print("Missing values:\n", df.isna().sum())

# Numeric imputation
num_imputer = SimpleImputer(strategy="median")
df[["age", "income"]] = num_imputer.fit_transform(df[["age", "income"]])

# Categorical imputation
cat_imputer = SimpleImputer(strategy="most_frequent")
df[["city"]] = cat_imputer.fit_transform(df[["city"]])

print("\nAfter imputation:\n", df)
```

### Exercise 2 — Outlier Detection
```python
import pandas as pd
import numpy as np

np.random.seed(0)
prices = pd.Series(np.concatenate([
    np.random.normal(100, 15, 97),
    [500, 600, -50]  # outliers
]))

# IQR method
Q1, Q3 = prices.quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers = prices[(prices < lower) | (prices > upper)]
print(f"IQR: {IQR:.2f}, Lower: {lower:.2f}, Upper: {upper:.2f}")
print("Outliers:\n", outliers)

# Clip to bounds
prices_clipped = prices.clip(lower, upper)
print("Clipped stats:", prices_clipped.describe())
```

### Exercise 3 — Duplicate Removal and String Cleaning
```python
import pandas as pd

df = pd.DataFrame({
    "name":  ["Alice", "bob", " Charlie", "Alice", "BOB", "Diana"],
    "score": [90, 85, 78, 90, 85, 92]
})
# Normalize names
df["name"] = df["name"].str.strip().str.title()
print("Before dedup:\n", df)

df_clean = df.drop_duplicates()
print("After dedup:\n", df_clean)
```

---

## Questions & Answers

**Q1: Should I impute missing values before or after splitting into train/test?**
> Always fit the imputer on the training set only, then use `transform()` on both train and test. If you fit on the combined dataset, you "leak" test statistics into training — this overly optimistic scenario is called data leakage and leads to unrealistically good validation scores.

**Q2: When should I drop rows with missing values vs impute?**
> Drop rows when: the fraction missing is small (<1–2%) and missing is random. Impute when: there are many missing values, or the pattern of missingness carries signal you do not want to lose. Dropping too many rows can introduce bias if data is not missing completely at random.

**Q3: What is the difference between MCAR, MAR, and MNAR?**
> - **MCAR** (Missing Completely At Random): missingness is unrelated to any data. Safe to drop or impute simply.
> - **MAR** (Missing At Random): missingness depends on observed variables but not the missing variable itself. Imputation models can handle this.
> - **MNAR** (Missing Not At Random): missingness depends on the missing value itself (e.g., high earners skip income field). Imputation is biased; special models or domain knowledge are needed.

**Q4: Why is the IQR method robust for outlier detection?**
> The IQR (Q3−Q1) is based on quantiles and is unaffected by extreme values. The mean and standard deviation are sensitive to outliers — a few very large values inflate σ, making the Z-score method less reliable when the data is already contaminated.

**Q5: Why should I not call `df["col"] = df["col"].str.lower()` on a slice?**
> Operating on a slice may cause a `SettingWithCopyWarning` because you might be modifying a copy rather than the original. Always work on the full DataFrame or use `.loc` to ensure modifications are made to the original object.
