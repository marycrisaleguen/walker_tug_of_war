# Unit 2 — Pandas

## Topics

### 1. Series and DataFrames
- `pd.Series`: 1-D labeled array
- `pd.DataFrame`: 2-D labeled table (columns of Series)
- Creating from dicts, lists, NumPy arrays, CSV files

### 2. Reading and Writing Data
- `pd.read_csv()`, `pd.read_excel()`, `pd.read_json()`
- `df.to_csv()`, `df.to_parquet()`
- Key parameters: `index_col`, `parse_dates`, `dtype`

### 3. Selecting and Filtering
- Column selection: `df["col"]`, `df[["col1", "col2"]]`
- Row selection: `df.loc[label]`, `df.iloc[index]`
- Boolean filtering: `df[df["age"] > 30]`
- `query()` method: `df.query("age > 30 and salary > 50000")`

### 4. Handling Missing Data
- Detection: `df.isna()`, `df.isnull()`, `df.info()`
- Dropping: `df.dropna()`
- Filling: `df.fillna(value)`, `df.fillna(method="ffill")`

### 5. GroupBy and Aggregations
- `df.groupby("col").agg({"val": ["mean", "std"]})`
- `pivot_table()`, `crosstab()`
- Sorting: `df.sort_values("col", ascending=False)`

### 6. Merging and Reshaping
- `pd.merge(df1, df2, on="key", how="inner/left/right/outer")`
- `pd.concat([df1, df2], axis=0)`
- `df.melt()` (wide → long), `df.pivot()` (long → wide)

### 7. Apply and Vectorized Operations
- `df["col"].apply(func)` — row/column-wise function application
- String methods: `df["col"].str.lower()`, `.str.contains()`
- DateTime methods: `df["date"].dt.year`, `.dt.dayofweek`

---

## Exercises

### Exercise 1 — DataFrame Basics
```python
import pandas as pd

data = {
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age":  [25, 30, 35, 28, 22],
    "salary": [70000, 85000, 90000, 75000, 60000],
    "dept": ["Eng", "HR", "Eng", "Marketing", "Eng"]
}
df = pd.DataFrame(data)
print(df.head())
print(df.dtypes)
print(df.describe())
```

### Exercise 2 — Filtering and Selection
Using the DataFrame above:
1. Select employees in Engineering
2. Select name and salary of employees earning more than $72,000
3. Find the employee with the highest salary

```python
eng = df[df["dept"] == "Eng"]
high_earners = df.loc[df["salary"] > 72000, ["name", "salary"]]
top_earner = df.loc[df["salary"].idxmax()]
print(eng); print(high_earners); print(top_earner)
```

### Exercise 3 — GroupBy
```python
# Average salary and headcount per department
summary = df.groupby("dept").agg(
    avg_salary=("salary", "mean"),
    headcount=("name", "count")
).reset_index()
print(summary)
```

### Exercise 4 — Missing Data
```python
import pandas as pd
import numpy as np

df2 = pd.DataFrame({
    "A": [1, 2, np.nan, 4, 5],
    "B": [np.nan, 2, 3, np.nan, 5],
    "C": [1, 2, 3, 4, 5]
})
print("Missing values:\n", df2.isna().sum())
df_filled = df2.fillna(df2.mean())
print("After mean-fill:\n", df_filled)
```

---

## Questions & Answers

**Q1: What is the difference between `loc` and `iloc`?**
> `loc` is label-based: you index by row/column labels (names). `iloc` is integer position-based: you index by 0-based integer positions. If the DataFrame has a default integer index they look the same, but they diverge when the index has string labels or non-contiguous integers.

**Q2: Why is `df["col"]` sometimes called a "view" and sometimes a "copy"?**
> Pandas may return a view (pointing to original data) or a copy depending on the operation. This is called the "SettingWithCopyWarning". Best practice: use `.loc[rows, cols] = value` to set values. Avoid chained indexing `df[cond]["col"] = val` as it may not modify the original.

**Q3: When should I use `apply()` vs vectorized methods?**
> Vectorized pandas/NumPy methods (`.str`, `.dt`, arithmetic operators) are much faster because they operate at the C level. Use `apply()` only when no vectorized alternative exists. For custom row-level logic, `apply(func, axis=1)` can be convenient but slow for large DataFrames.

**Q4: What is the difference between `merge` and `concat`?**
> `concat` stacks DataFrames along an axis (vertically or horizontally). `merge` performs SQL-style joins, combining rows from two DataFrames based on matching key values. Use `concat` to stack data from the same schema; use `merge` to join data from different sources on a common key.

**Q5: How should I handle a mix of numerical and categorical columns for ML?**
> Read the data with pandas, separate numerical columns (use directly or scale them) and categorical columns (encode with `pd.get_dummies()`, `OrdinalEncoder`, or `OneHotEncoder`). Then convert to a NumPy array for sklearn with `.values` or `.to_numpy()`.
