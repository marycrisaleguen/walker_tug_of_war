# Unit 1 — NumPy

## Topics

### 1. Creating Arrays
- `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`
- `np.random.randn()`, `np.random.randint()`
- Array attributes: `shape`, `dtype`, `ndim`, `size`

### 2. Indexing and Slicing
- 1-D: `a[0]`, `a[-1]`, `a[1:4]`
- 2-D: `A[row, col]`, `A[1:3, :]`, `A[:, 2]`
- Boolean masking: `a[a > 5]`
- Fancy indexing: `a[[0, 2, 4]]`

### 3. Array Operations
- Element-wise arithmetic: `+`, `-`, `*`, `/`, `**`
- Aggregations: `sum`, `mean`, `std`, `min`, `max`, `argmin`, `argmax`
- Axis parameter: `np.sum(A, axis=0)` (column sums), `axis=1` (row sums)

### 4. Broadcasting
- Rules for operating on arrays of different shapes
- Dimensions are compared from the right; sizes must be equal or one of them must be 1
- Example: adding a (3,) vector to a (4,3) matrix

### 5. Linear Algebra with NumPy
- `np.dot()`, `@` operator: matrix multiplication
- `np.linalg.inv()`, `np.linalg.solve()`
- `np.linalg.norm()`, `np.linalg.eig()`

### 6. Performance Tips
- Avoid Python loops — use vectorized operations
- `np.vectorize()` as a convenience wrapper (not as fast as true ufuncs)
- Memory layout: C order vs Fortran order

---

## Exercises

### Exercise 1 — Array Creation and Inspection
```python
import numpy as np

# Create and inspect
a = np.arange(1, 13).reshape(3, 4)
print("Shape:", a.shape, "Dtype:", a.dtype, "ndim:", a.ndim)
print(a)

# Slicing
print("First row:", a[0])
print("Last column:", a[:, -1])
print("Sub-matrix (rows 0-1, cols 1-3):\n", a[:2, 1:])
```

### Exercise 2 — Broadcasting
```python
import numpy as np

# Mean-center each column (feature normalization)
X = np.array([[1, 200, 0.5],
              [2, 150, 1.2],
              [3, 180, 0.8]])

col_means = X.mean(axis=0)   # shape (3,)
X_centered = X - col_means    # broadcasting: (3,3) - (3,) -> (3,3)
print("Column means:", col_means)
print("Centered:\n", X_centered)
```

### Exercise 3 — Boolean Masking
```python
import numpy as np

np.random.seed(42)
data = np.random.randn(1000)

above_2 = data[data > 2]
print(f"Values above 2.0: {len(above_2)} ({len(above_2)/len(data)*100:.2f}%)")
print(f"Expected for N(0,1): ~2.28%")

# Replace negative values with 0 (ReLU!)
relu = data.copy()
relu[relu < 0] = 0
print(f"After ReLU — negative values: {(relu < 0).sum()}")
```

### Exercise 4 — Vectorized vs Loop Performance
```python
import numpy as np
import time

n = 10_000_000
a = np.random.randn(n)
b = np.random.randn(n)

# Loop
t0 = time.time()
result_loop = sum(a[i]*b[i] for i in range(n))
print(f"Loop: {time.time()-t0:.3f}s")

# Vectorized
t0 = time.time()
result_vec = np.dot(a, b)
print(f"Vectorized: {time.time()-t0:.3f}s")
print(f"Results match: {abs(result_loop - result_vec) < 1e-6}")
```

---

## Questions & Answers

**Q1: What is the difference between `np.array` and a Python list?**
> NumPy arrays are typed, contiguous blocks of memory that support efficient vectorized operations. Python lists are flexible containers of Python objects. Arrays are ~10–100× faster for numerical operations and are the foundation of the entire scientific Python ecosystem.

**Q2: What is broadcasting?**
> Broadcasting is NumPy's mechanism for performing arithmetic between arrays of different — but compatible — shapes. The smaller array is virtually "stretched" along its size-1 dimensions to match the larger array. No copies are made; this is a memory-efficient logical expansion.

**Q3: What is the difference between `a.reshape(-1)` and `a.ravel()`?**
> Both flatten an array to 1-D. `ravel()` returns a view when possible (no memory copy), while `reshape(-1)` also returns a view when possible. In practice they behave identically for most use cases. `a.flatten()` always returns a copy.

**Q4: What does `axis=0` mean in aggregation functions?**
> `axis=0` means "collapse along the rows," resulting in column-wise statistics. For a (m×n) matrix, `np.sum(A, axis=0)` returns an (n,) array of column sums. `axis=1` collapses along columns, giving row-wise statistics.

**Q5: When should I copy an array vs use a view?**
> Slices return views — modifying them modifies the original. Use `arr.copy()` when you want an independent array. This is important in ML preprocessing: if you slice a feature matrix and modify the slice, you may corrupt your data.
