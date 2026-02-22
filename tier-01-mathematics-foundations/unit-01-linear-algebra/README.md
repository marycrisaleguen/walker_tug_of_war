# Unit 1 — Linear Algebra

## Topics

### 1. Vectors
- Definition: a 1-D array of numbers representing magnitude and direction
- Notation: **v** = [v₁, v₂, …, vₙ]ᵀ
- Operations: addition, scalar multiplication, dot product, norm (L1, L2)
- Geometric interpretation: arrows in n-dimensional space

### 2. Matrices
- Definition: a 2-D array of numbers (m rows × n columns)
- Operations: addition, scalar multiplication, matrix multiplication, transpose
- Special matrices: identity (I), diagonal, symmetric, orthogonal

### 3. Matrix Operations in ML
- Weight matrices in neural networks
- Data matrix X (n_samples × n_features)
- Predictions: **ŷ** = X**w**

### 4. Eigenvalues and Eigenvectors
- Definition: A**v** = λ**v**
- Geometric intuition: directions that only stretch under transformation
- Application: PCA, spectral decomposition

### 5. Systems of Linear Equations
- Gaussian elimination
- Matrix inverse: A⁻¹
- Least squares solution: **w** = (XᵀX)⁻¹Xᵀ**y**

---

## Exercises

### Exercise 1 — Vector Operations
Given **a** = [3, -1, 2] and **b** = [0, 4, -1]:

1. Compute **a** + **b**
2. Compute 2**a** − **b**
3. Compute the dot product **a** · **b**
4. Compute the L2 norm of **a** (‖**a**‖₂)

```python
import numpy as np

a = np.array([3, -1, 2])
b = np.array([0, 4, -1])

print("a + b =", a + b)
print("2a - b =", 2*a - b)
print("dot product =", np.dot(a, b))
print("L2 norm of a =", np.linalg.norm(a))
```

### Exercise 2 — Matrix Multiplication
Given:
```
A = [[1, 2],    B = [[5, 6],
     [3, 4]]         [7, 8]]
```
1. Compute A × B by hand (verify with code)
2. Compute Aᵀ
3. Is A × B == B × A? Explain why.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A @ B =\n", A @ B)
print("B @ A =\n", B @ A)
print("A.T =\n", A.T)
```

### Exercise 3 — Eigenvalues
For the matrix `M = [[4, 1], [2, 3]]`:

1. Compute eigenvalues and eigenvectors using NumPy
2. Verify: M @ v ≈ λ * v for each eigenpair

```python
import numpy as np

M = np.array([[4, 1], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(M)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)
```

### Exercise 4 — Least Squares
Given data points (x, y): (1,2), (2,4), (3,5), (4,4), (5,5):

Solve the normal equation **w** = (XᵀX)⁻¹Xᵀ**y** to find the best-fit line y = w₀ + w₁x.

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
X = np.column_stack([np.ones(5), x])  # add bias column

w = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"Intercept: {w[0]:.4f}, Slope: {w[1]:.4f}")
```

---

## Questions & Answers

**Q1: What is the difference between a row vector and a column vector?**
> A row vector has shape (1, n) and a column vector has shape (n, 1). By convention in ML, data vectors are usually column vectors. In NumPy, a 1-D array has shape (n,) — neither row nor column — but behaves like a column vector in most operations.

**Q2: When is matrix multiplication defined?**
> A (m×n) matrix can be multiplied by a (n×p) matrix. The inner dimensions must match. The result is an (m×p) matrix.

**Q3: What does the dot product measure?**
> The dot product **a**·**b** = ‖**a**‖‖**b**‖cos(θ) measures the alignment between two vectors. It is maximized when they point in the same direction (θ=0), zero when orthogonal (θ=90°), and negative when opposite.

**Q4: Why are eigenvalues important in ML?**
> Eigenvalues appear in PCA (they tell you how much variance each principal component explains), in spectral graph theory, in analyzing the convergence of optimization algorithms, and in understanding how transformations stretch or compress space.

**Q5: What is the significance of (XᵀX)⁻¹Xᵀy in linear regression?**
> This is the closed-form (analytical) solution for ordinary least squares regression. It finds the weight vector **w** that minimizes the sum of squared residuals. It requires (XᵀX) to be invertible, which fails when features are perfectly collinear or when n < p.
