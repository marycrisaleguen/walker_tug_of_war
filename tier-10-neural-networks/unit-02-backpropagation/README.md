# Unit 2 — Backpropagation

## Topics

### 1. The Learning Problem
- Given loss L(**w**), find **w** that minimizes L
- Gradient descent requires ∂L/∂wᵢⱼ for every weight in the network
- Naive finite differences: O(n) forward passes for n parameters — infeasible

### 2. Backpropagation Algorithm
- Efficiently computes all gradients in a single backward pass using the chain rule
- Runs in O(same time as a forward pass)
- Two phases: Forward pass (compute activations), Backward pass (compute gradients)

### 3. Chain Rule Through a Network
- For output layer: δ_L = ∂L/∂z_L (gradient of loss w.r.t. pre-activation output)
- For hidden layers: δₗ = (W_{l+1}ᵀ δ_{l+1}) ⊙ f'(zₗ)
- Weight gradients: ∂L/∂Wₗ = δₗ hₗ₋₁ᵀ
- Bias gradients: ∂L/∂bₗ = δₗ

### 4. Vanishing and Exploding Gradients
- **Vanishing**: gradients shrink exponentially in deep networks (sigmoid/tanh saturate)
  - Fix: use ReLU activations, batch normalization, residual connections
- **Exploding**: gradients grow exponentially (common in RNNs)
  - Fix: gradient clipping, careful initialization

### 5. Weight Initialization
- Zero initialization: all neurons learn the same thing (symmetry problem)
- **Xavier/Glorot**: W ~ Uniform(−√(6/(n_in+n_out)), +√(6/(n_in+n_out))) — for tanh/sigmoid
- **He/Kaiming**: W ~ N(0, √(2/n_in)) — for ReLU activations

### 6. Computational Graphs
- Modern frameworks (PyTorch, TensorFlow) build a computation graph
- Backprop is automatic differentiation on the graph
- `loss.backward()` in PyTorch computes all gradients

---

## Exercises

### Exercise 1 — Backprop by Hand (1-Layer Network)
```python
import numpy as np

def sigmoid(z): return 1/(1+np.exp(-z))
def sigmoid_prime(z): s = sigmoid(z); return s*(1-s)

np.random.seed(42)
# Network: 2 inputs -> 3 hidden -> 1 output
X = np.array([[1.0, 2.0]])
y = np.array([[1.0]])

W1 = np.random.randn(3, 2) * 0.1; b1 = np.zeros((3, 1))
W2 = np.random.randn(1, 3) * 0.1; b2 = np.zeros((1, 1))

# Forward pass
z1 = W1 @ X.T + b1; a1 = sigmoid(z1)
z2 = W2 @ a1 + b2; a2 = sigmoid(z2)
loss = ((y.T - a2)**2).mean()
print(f"Forward: loss={loss:.4f}, prediction={a2.ravel()[0]:.4f}")

# Backward pass
dL_da2 = 2*(a2 - y.T)/y.size
dL_dz2 = dL_da2 * sigmoid_prime(z2)
dL_dW2 = dL_dz2 @ a1.T
dL_db2 = dL_dz2

dL_da1 = W2.T @ dL_dz2
dL_dz1 = dL_da1 * sigmoid_prime(z1)
dL_dW1 = dL_dz1 @ X
dL_db1 = dL_dz1

print(f"dL/dW2 shape: {dL_dW2.shape}, dL/dW1 shape: {dL_dW1.shape}")
```

### Exercise 2 — PyTorch Autograd
```python
import torch
import torch.nn as nn

# Define a simple 2-layer net and verify gradients
torch.manual_seed(42)
X = torch.randn(10, 4)
y = torch.randint(0, 3, (10,))

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for step in range(5):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    # Inspect gradient of first layer weights
    grad_norm = model[0].weight.grad.norm().item()
    optimizer.step()
    print(f"Step {step}: loss={loss.item():.4f}, grad_norm={grad_norm:.6f}")
```

---

## Questions & Answers

**Q1: Why is backpropagation efficient compared to finite differences?**
> Finite differences require computing the loss twice for each parameter (f(w+ε) and f(w−ε)), giving O(p) forward passes for p parameters. Backpropagation computes all p gradients in one forward + one backward pass — O(1) instead of O(p). For a network with millions of parameters, this is the difference between feasible and infeasible.

**Q2: What is the "symmetry problem" with zero initialization?**
> If all weights are initialized to zero, every neuron in a layer computes the same output and receives the same gradient — they are symmetric. All neurons update identically and remain identical forever. They effectively learn the same feature, wasting the capacity of the layer. Random initialization breaks this symmetry.

**Q3: Why does ReLU help with vanishing gradients?**
> The derivative of ReLU is 1 for positive activations and 0 for negative (dead neurons). For the active units, the gradient flows through without shrinking (unlike sigmoid/tanh, which saturate at ±1 and have near-zero derivatives there). This allows gradients to flow back through many layers without exponential decay.

**Q4: What is gradient clipping?**
> If the gradient norm exceeds a threshold, all gradient values are scaled down proportionally: g ← g * (threshold / ‖g‖). This prevents exploding gradients in RNNs and very deep networks without completely stopping learning. `torch.nn.utils.clip_grad_norm_()` in PyTorch.

**Q5: What is automatic differentiation (autograd)?**
> Autograd is an exact method for computing derivatives by tracking all mathematical operations performed on tensors in a computation graph. It is different from symbolic differentiation (manipulates expressions symbolically) and numerical differentiation (finite differences). It combines the speed of numerical computation with the exactness of symbolic derivatives. PyTorch and TensorFlow implement autograd.
