# Unit 2 — Calculus

## Topics

### 1. Functions and Limits
- Definition of a limit: lim(x→a) f(x)
- Continuity
- Relevance: loss functions are continuous real-valued functions of model parameters

### 2. Derivatives
- Definition: f'(x) = lim(h→0) [f(x+h) − f(x)] / h
- Interpretation: instantaneous rate of change (slope of tangent line)
- Common rules: power rule, product rule, quotient rule, chain rule

### 3. Partial Derivatives
- Derivative of a multivariate function with respect to one variable, holding others constant
- Notation: ∂f/∂x
- The gradient ∇f collects all partial derivatives into a vector

### 4. The Gradient
- ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, …, ∂f/∂xₙ]ᵀ
- Points in the direction of steepest ascent
- Negative gradient points toward the local minimum → gradient descent

### 5. Chain Rule
- If z = f(g(x)), then dz/dx = (dz/dg)(dg/dx)
- Fundamental for backpropagation in neural networks

### 6. Optimization
- Critical points: where f'(x) = 0 or ∇f = 0
- Second derivative test to classify minima/maxima/saddle points
- Gradient descent update: **w** ← **w** − α ∇L(**w**)

---

## Exercises

### Exercise 1 — Basic Derivatives
Differentiate each function by hand, then verify numerically:

1. f(x) = 3x² − 5x + 2
2. g(x) = x³ · e^x
3. h(x) = ln(x² + 1)
4. k(x) = 1 / (1 + e^{−x})  ← this is the sigmoid function!

```python
import sympy as sp

x = sp.Symbol('x')
functions = [
    3*x**2 - 5*x + 2,
    x**3 * sp.exp(x),
    sp.ln(x**2 + 1),
    1 / (1 + sp.exp(-x))
]
for f in functions:
    print(f"f(x) = {f}")
    print(f"f'(x) = {sp.diff(f, x)}\n")
```

### Exercise 2 — Gradient of the MSE Loss
The Mean Squared Error loss for linear regression is:
L(**w**) = (1/n) ‖**y** − X**w**‖²

Derive ∇_**w** L(**w**) = (−2/n) Xᵀ(**y** − X**w**)

Then implement a gradient descent loop:

```python
import numpy as np

np.random.seed(42)
n, p = 100, 2
X = np.random.randn(n, p)
true_w = np.array([3.0, -1.5])
y = X @ true_w + 0.1 * np.random.randn(n)

w = np.zeros(p)
lr = 0.01
for step in range(500):
    grad = (-2/n) * X.T @ (y - X @ w)
    w -= lr * grad
    if step % 100 == 0:
        loss = np.mean((y - X @ w)**2)
        print(f"Step {step}: loss={loss:.6f}, w={w}")
print(f"\nTrue w: {true_w}")
```

### Exercise 3 — Chain Rule
Given f(u) = u², u(x) = 3x + 1, find df/dx at x = 2 using the chain rule. Verify numerically with a finite difference approximation: (f(x+h) − f(x)) / h for small h.

```python
def f(x):
    u = 3*x + 1
    return u**2

x0, h = 2.0, 1e-5
analytical = 2*(3*x0+1)*3  # chain rule: df/du * du/dx
numerical  = (f(x0+h) - f(x0)) / h
print(f"Analytical: {analytical}, Numerical: {numerical:.6f}")
```

---

## Questions & Answers

**Q1: What is the gradient and why does it matter for ML?**
> The gradient is the vector of partial derivatives of a scalar function with respect to all its inputs. In ML, the loss function L(**w**) maps model parameters to a scalar loss value. The gradient ∇L(**w**) tells us how steeply the loss increases in each parameter direction. By moving opposite to the gradient (gradient descent), we reduce the loss.

**Q2: Why is the chain rule critical for neural networks?**
> Neural networks compose many functions layer by layer: L = f_n(f_{n-1}(…f_1(x))). To compute ∂L/∂w_i for a weight deep in the network, the chain rule lets us multiply local gradients layer by layer — this is exactly what backpropagation does.

**Q3: What is the difference between a local minimum and a global minimum?**
> A local minimum is a point where the function value is lower than nearby points, but not necessarily the lowest overall. A global minimum is the lowest point everywhere. Gradient descent is only guaranteed to find a local minimum, not a global one. For convex functions (like MSE), every local minimum is also the global minimum.

**Q4: What is the learning rate and how does it affect gradient descent?**
> The learning rate α controls the step size: **w** ← **w** − α ∇L. If α is too large, the updates overshoot and the loss may diverge. If α is too small, convergence is slow. Choosing a good learning rate — or using adaptive optimizers like Adam — is one of the key hyperparameter decisions in training ML models.

**Q5: What is a saddle point?**
> A saddle point is a critical point (∇f = 0) that is neither a local minimum nor a maximum — it is a minimum in some directions and a maximum in others. Saddle points are common in high-dimensional loss landscapes and can slow down optimization.
