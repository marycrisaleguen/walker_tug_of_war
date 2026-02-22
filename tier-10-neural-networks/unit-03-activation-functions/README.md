# Unit 3 — Activation Functions

## Topics

### 1. Role of Activation Functions
- Introduce non-linearity — the key to approximating complex functions
- Every hidden layer's output is passed through an activation function
- Choice of activation affects gradient flow, convergence speed, and final performance

### 2. Sigmoid
- σ(x) = 1/(1+e⁻ˣ)  →  output ∈ (0,1)
- Derivative: σ'(x) = σ(x)(1−σ(x)) ≤ 0.25
- Problems: saturates for large |x| → near-zero gradients (vanishing gradient)
- Use: output layer for binary classification

### 3. Tanh
- tanh(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)  →  output ∈ (−1,1)
- Derivative: 1 − tanh(x)² ≤ 1
- Zero-centered (advantage over sigmoid) but still saturates
- Often preferred over sigmoid in hidden layers when non-ReLU needed

### 4. ReLU (Rectified Linear Unit)
- f(x) = max(0, x)
- Derivative: 1 if x>0, 0 if x<0 (undefined at 0, set to 0)
- Fast to compute; does not saturate for positive x; sparse activations
- Problem: Dying ReLU — neurons can get stuck at 0 permanently

### 5. Leaky ReLU & Variants
- Leaky ReLU: f(x) = x if x>0, else αx (α ≈ 0.01) — allows small negative gradient
- **ELU** (Exponential LU): smooth, negative values push mean activations toward 0
- **SELU**: self-normalizing variant of ELU; maintains unit mean/variance across layers
- **GELU**: used in Transformers (BERT, GPT); smooth approximation of ReLU

### 6. Softmax
- Used in the output layer for multi-class classification
- softmax(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ  →  outputs sum to 1 (probability distribution)
- Numerically stable: subtract max(z) before exponentiating

### 7. Choosing Activation Functions
| Layer | Recommended |
|-------|-------------|
| Hidden (default) | ReLU or GELU |
| Hidden (RNNs) | Tanh |
| Binary output | Sigmoid |
| Multi-class output | Softmax |
| Regression output | None (linear) |

---

## Exercises

### Exercise 1 — Visualize Activation Functions
```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):      return np.maximum(0, x)
def leaky_relu(x, a=0.1): return np.where(x > 0, x, a*x)
def sigmoid(x):   return 1/(1+np.exp(-x))
def tanh(x):      return np.tanh(x)

x = np.linspace(-4, 4, 300)
funcs = {"Sigmoid": sigmoid, "Tanh": tanh, "ReLU": relu, "Leaky ReLU": leaky_relu}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (name, f) in zip(axes.ravel(), funcs.items()):
    ax.plot(x, f(x), label=name, linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5); ax.axvline(0, color='black', linewidth=0.5)
    ax.set_title(name); ax.set_ylim(-1.5, 3.5); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

### Exercise 2 — Effect of Activation on Gradient Flow
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def test_gradients(activation, n_layers=10):
    torch.manual_seed(42)
    layers = [nn.Linear(10, 10)]
    for _ in range(n_layers - 1):
        layers += [activation, nn.Linear(10, 10)]
    model = nn.Sequential(*layers)

    x = torch.randn(1, 10)
    y = model(x).sum()
    y.backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    return grad_norms

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, act) in zip(axes, [("Sigmoid", nn.Sigmoid()), ("ReLU", nn.ReLU())]):
    norms = test_gradients(act)
    ax.bar(range(len(norms)), norms)
    ax.set_title(f"{name}: Gradient Norms per Layer")
    ax.set_xlabel("Parameter Index"); ax.set_ylabel("Gradient Norm")
plt.tight_layout(); plt.show()
```

### Exercise 3 — Softmax Temperature
```python
import numpy as np
import matplotlib.pyplot as plt

def softmax(z, T=1.0):
    z = np.array(z) / T
    e = np.exp(z - z.max())
    return e / e.sum()

logits = [2.0, 1.0, 0.5, -0.5]
temps = [0.1, 1.0, 5.0]
print("Temperature effect on Softmax probabilities:")
for T in temps:
    probs = softmax(logits, T)
    print(f"  T={T}: {probs.round(3)}")
```

---

## Questions & Answers

**Q1: Why is ReLU the default choice for hidden layers?**
> ReLU has several advantages: (1) No saturation for positive inputs → no vanishing gradient on the positive side. (2) Computationally cheap (just a max operation). (3) Induces sparse activations (≈50% of neurons output 0 on any input), which helps learning efficiency. (4) Empirically works very well for deep networks.

**Q2: What is the Dying ReLU problem?**
> If a neuron's pre-activation value becomes consistently negative (e.g., after a large negative gradient update), it will always output 0 and have zero gradient — it "dies" and never recovers. Fix: use Leaky ReLU, ELU, or careful learning rate management. Batch normalization also helps prevent this.

**Q3: Why do we subtract max(z) before computing softmax?**
> Without this, computing e^z for large z values causes numerical overflow (inf). Subtracting max(z) shifts all values to ≤ 0, so e^z ∈ (0,1]. The result is mathematically identical: softmax(z) = softmax(z − c) for any constant c.

**Q4: What does "softmax temperature" do?**
> Dividing logits by temperature T before softmax: as T→0, the distribution becomes one-hot (sharp, confident). As T→∞, the distribution becomes uniform (uncertain). Temperature scaling is used in: (1) training language models with "temperature sampling" for diversity; (2) knowledge distillation, where a teacher's soft labels use high temperature.

**Q5: When should I use GELU instead of ReLU?**
> GELU (Gaussian Error Linear Unit) is used in Transformer-based models (BERT, GPT). It is a smooth approximation to ReLU with slight negative values for negative inputs, allowing a small gradient to flow. In practice, GELU marginally outperforms ReLU for language models and some vision models. For most feedforward networks, ReLU or its variants work well.
