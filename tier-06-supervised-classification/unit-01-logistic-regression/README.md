# Unit 1 — Logistic Regression

## Topics

### 1. The Classification Problem
- Output is a discrete class label (binary: 0/1, or multi-class)
- Goal: model P(y=1 | **x**) — probability of belonging to the positive class

### 2. The Sigmoid Function
- σ(z) = 1 / (1 + e^{−z}) maps any real value to (0, 1)
- Model: P(y=1 | **x**) = σ(**w**ᵀ**x** + b)
- Decision boundary: predict 1 if P > 0.5, else 0

### 3. Log-Loss (Binary Cross-Entropy)
- L(**w**) = −(1/n) Σ [yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]
- Derived from maximum likelihood with Bernoulli distribution
- No closed-form solution — optimize with gradient descent

### 4. Multi-Class Logistic Regression
- One-vs-Rest (OvR): train k binary classifiers, pick highest score
- Softmax regression: output probability for each class, Σpᵢ = 1
- `multi_class="multinomial"` with `solver="lbfgs"` in sklearn

### 5. Decision Boundary
- Linear: **w**ᵀ**x** + b = 0 defines a hyperplane
- Can make non-linear boundaries by adding polynomial features

### 6. Regularization for Logistic Regression
- L2 regularization: parameter `C = 1/λ` in sklearn (larger C = less regularization)
- Lasso for sparse logistic regression: `penalty="l1"`, `solver="liblinear"`

---

## Exercises

### Exercise 1 — Logistic Regression from Scratch
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y, y_hat):
    eps = 1e-9
    return -np.mean(y * np.log(y_hat + eps) + (1-y) * np.log(1 - y_hat + eps))

np.random.seed(42)
n, p = 200, 2
X = np.random.randn(n, p)
true_w = np.array([2.0, -1.5])
y = (sigmoid(X @ true_w) > 0.5).astype(float)

# Gradient descent
w = np.zeros(p)
lr = 0.1
for step in range(1000):
    y_hat = sigmoid(X @ w)
    grad = X.T @ (y_hat - y) / n
    w -= lr * grad
    if step % 200 == 0:
        print(f"Step {step}: loss={log_loss(y, y_hat):.4f}, w={w}")
print(f"\nTrue w: {true_w}")
```

### Exercise 2 — sklearn Logistic Regression
```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix"); plt.tight_layout(); plt.show()
```

---

## Questions & Answers

**Q1: Why can't we use MSE as the loss for logistic regression?**
> MSE with a sigmoid output creates a non-convex loss surface with many local minima, making optimization unreliable. The log-loss (cross-entropy) is convex when the model is the sigmoid of a linear function, guaranteeing that gradient descent finds the global minimum.

**Q2: What is the difference between the probability output and the class prediction?**
> The model outputs P(y=1 | **x**) ∈ (0,1) — a probability. To get a class label, we apply a threshold (default: 0.5). You can adjust the threshold to trade off precision vs recall. Always evaluate the raw probabilities when diagnosing calibration.

**Q3: What is the `C` parameter in sklearn's LogisticRegression?**
> C is the inverse of the regularization strength: C = 1/λ. Smaller C → stronger regularization → simpler model (risk of underfitting). Larger C → weaker regularization → more complex model (risk of overfitting). Default C=1.0. Tune with cross-validation.

**Q4: Can logistic regression handle non-linearly separable data?**
> Not without feature engineering. The decision boundary is always a hyperplane. To handle non-linear boundaries, add polynomial or interaction features, or switch to algorithms like kernel SVM, decision trees, or neural networks.

**Q5: What is the log-odds (logit)?**
> log(P/(1-P)) is the log-odds, also called the logit. The logistic regression model is linear in the log-odds: log(P/(1-P)) = **w**ᵀ**x** + b. The sigmoid is the inverse of the logit function.
