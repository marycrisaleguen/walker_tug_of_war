# Unit 1 — Perceptron & Multi-Layer Perceptron (MLP)

## Topics

### 1. Biological Inspiration
- Neuron: receives signals (dendrites), integrates them (cell body), fires if threshold exceeded (axon)
- Artificial neuron: weighted sum of inputs → activation function → output

### 2. The Perceptron
- Simplest neural network: single neuron
- Output: f(**w**ᵀ**x** + b), where f is a step function (0 or 1)
- Perceptron learning rule: **w** ← **w** + α(y − ŷ)**x**
- Converges only if data is linearly separable (Perceptron convergence theorem)

### 3. Multi-Layer Perceptron (MLP)
- Multiple layers of neurons: input → hidden → output
- Hidden layers allow non-linear transformations
- Universal approximation theorem: a single hidden layer with enough neurons can approximate any continuous function

### 4. Layer Types
- **Input layer**: receives raw features (no computation)
- **Hidden layers**: intermediate representations
- **Output layer**: final predictions (sigmoid for binary, softmax for multi-class, linear for regression)

### 5. Forward Pass
- Input propagates through layers: h = activation(W·x + b)
- Each layer transforms the representation
- Matrix form for a 2-layer MLP: ŷ = W₂ · activation(W₁ · x + b₁) + b₂

### 6. Non-Linearity: Why Hidden Layers Need Activation Functions
- Without activation functions, multiple layers = one linear layer (composing linear functions is linear)
- Activation functions introduce non-linearity, enabling complex decision boundaries

---

## Exercises

### Exercise 1 — Single Neuron from Scratch
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Single neuron: XOR is not linearly separable
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0, 0, 0, 1])  # AND gate (linearly separable)

# Initialize weights
np.random.seed(42)
w = np.random.randn(2) * 0.1
b = 0.0
lr = 0.5

for epoch in range(10000):
    z = X @ w + b
    y_hat = sigmoid(z)
    loss = -np.mean(y_and * np.log(y_hat+1e-9) + (1-y_and)*np.log(1-y_hat+1e-9))
    grad_w = X.T @ (y_hat - y_and) / len(y_and)
    grad_b = np.mean(y_hat - y_and)
    w -= lr * grad_w
    b -= lr * grad_b
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}")

print(f"\nPredictions: {sigmoid(X @ w + b).round(3)}")
print(f"Expected:    {y_and}")
```

### Exercise 2 — MLP with sklearn
```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
X_tr = StandardScaler().fit_transform(X_tr)
X_te = StandardScaler().fit(X_tr).transform(X_te)

mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu',
                    max_iter=1000, random_state=42)
mlp.fit(X_tr, y_tr)
print(f"Train accuracy: {accuracy_score(y_tr, mlp.predict(X_tr)):.4f}")
print(f"Test  accuracy: {accuracy_score(y_te, mlp.predict(X_te)):.4f}")
```

---

## Questions & Answers

**Q1: What is the Universal Approximation Theorem?**
> A feedforward neural network with a single hidden layer containing a sufficient number of neurons with a non-linear activation function can approximate any continuous function on a compact subset of ℝⁿ to arbitrary precision. This guarantees that a neural network can, in principle, represent any pattern — but it says nothing about whether gradient descent will find those weights.

**Q2: Why can't a perceptron learn XOR?**
> XOR is not linearly separable — no single line can separate the True and False outputs. A perceptron's decision boundary is always a hyperplane. Adding a hidden layer allows the MLP to combine multiple hyperplanes, creating non-linear decision regions that can solve XOR.

**Q3: What is the difference between a weight and a bias?**
> Weights (w) scale the inputs, controlling the importance of each feature. The bias (b) shifts the activation threshold, allowing the decision boundary to be positioned away from the origin. Without bias, the decision boundary must pass through the origin, severely limiting expressiveness.

**Q4: Why are deep networks (many hidden layers) better than wide networks (one wide layer)?**
> Both can approximate arbitrary functions (Universal Approximation Theorem). However, deep networks are more parameter-efficient for many functions (hierarchical feature learning) and often generalize better. They can represent functions that require exponential width with logarithmic depth. In practice, depth enables learning of compositional representations (edges → shapes → objects).

**Q5: What happens during the forward pass?**
> The input features are multiplied by the weight matrix of the first layer, a bias is added, and the result is passed through an activation function to produce the first hidden layer activations. This is repeated for each subsequent layer until the output layer produces the final prediction. Mathematically: h₁ = f(W₁x + b₁), h₂ = f(W₂h₁ + b₂), ..., ŷ = W_L h_{L-1} + b_L.
