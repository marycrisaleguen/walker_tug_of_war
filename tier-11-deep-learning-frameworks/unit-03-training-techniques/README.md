# Unit 3 — Training Techniques

## Topics

### 1. Optimizers
- **SGD (Stochastic Gradient Descent)**: update **w** ← **w** − α·g; simple, requires tuning
- **Momentum SGD**: accumulate velocity v ← γv + αg; builds up in consistent directions
- **RMSProp**: adapts per-parameter learning rate using exponential moving average of squared gradients
- **Adam** (Adaptive Moment Estimation): combines momentum + RMSProp; default choice
  - m ← β₁m + (1−β₁)g  (first moment / momentum)
  - v ← β₂v + (1−β₂)g²  (second moment / RMSProp)
  - **w** ← **w** − α · m̂ / (√v̂ + ε)

### 2. Learning Rate Schedules
- **Step decay**: reduce by factor every N epochs
- **Cosine annealing**: smoothly reduce lr following a cosine curve
- **Warm-up + decay**: start with small lr, ramp up, then decay — common for Transformers
- **ReduceLROnPlateau**: reduce lr when validation metric stagnates
- **Cyclical Learning Rates**: oscillate lr between min and max values

### 3. Batch Size
- Small batch: noisy gradients, can escape local minima, slower per step
- Large batch: smoother gradients, faster per step, but tends to find sharp minima (worse generalization)
- Rule of thumb: start with 32–256; scale learning rate linearly with batch size

### 4. Regularization Techniques for Neural Networks
- **Dropout**: randomly zero activations during training
- **Weight decay (L2 regularization)**: penalize large weights — implemented as `weight_decay` in Adam
- **Early stopping**: halt training when validation loss stops improving
- **Batch normalization**: normalizes activations — acts as mild regularization
- **Data augmentation**: expand the effective training set (covered in Tier 12)

### 5. Gradient Clipping
- Cap gradient norm to prevent exploding gradients
- `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- `tf.keras.optimizers.Adam(clipnorm=1.0)`

### 6. Learning Rate Finder
- Run one epoch, incrementally increasing lr from very small to large
- Plot loss vs lr; pick lr just before loss starts exploding
- `lr_finder` libraries or manual implementation

---

## Exercises

### Exercise 1 — Compare Optimizers
```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_tr_s = torch.FloatTensor(scaler.fit_transform(X_tr))
y_tr_t  = torch.FloatTensor(y_tr).unsqueeze(1)

def make_model():
    return nn.Sequential(nn.Linear(30, 64), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(64, 1), nn.Sigmoid())

def train(optimizer_class, lr=1e-3, epochs=100):
    model = make_model()
    opt   = optimizer_class(model.parameters(), lr=lr)
    crit  = nn.BCELoss()
    losses = []
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(X_tr_s), y_tr_t)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return losses

plt.figure(figsize=(8, 4))
for name, cls in [("SGD", optim.SGD), ("Adam", optim.Adam), ("RMSprop", optim.RMSprop)]:
    plt.plot(train(cls), label=name)
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Optimizer Comparison"); plt.legend(); plt.show()
```

### Exercise 2 — Cosine Annealing Schedule
```python
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Visualize learning rate schedule
model = torch.nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

lrs = []
for _ in range(100):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.plot(lrs); plt.xlabel("Epoch"); plt.ylabel("Learning Rate")
plt.title("Cosine Annealing Schedule"); plt.show()
```

---

## Questions & Answers

**Q1: Why is Adam the default optimizer for most neural networks?**
> Adam adapts the learning rate for each parameter independently based on the first and second moments of the gradients. This makes it robust to different gradient magnitudes across layers (common in deep networks) and requires minimal tuning compared to SGD. The default hyperparameters (β₁=0.9, β₂=0.999, ε=1e-8) work well for most problems.

**Q2: Why does SGD with momentum sometimes outperform Adam?**
> Adam can converge quickly early in training but sometimes to a sharper local minimum. SGD with momentum, while slower to converge, often finds flatter minima that generalize better. This is why many state-of-the-art image classification models use SGD + momentum + cosine annealing after initial warm-up, not Adam.

**Q3: What is weight decay and how does it relate to L2 regularization?**
> Weight decay multiplies all weights by (1 − α·λ) at each step. For SGD, weight decay and L2 regularization are exactly equivalent. For adaptive optimizers like Adam, they are different: L2 adds λ‖w‖² to the loss (which Adam then adapts), while AdamW (weight decay) decouples the regularization from the adaptive scaling. AdamW is the preferred variant.

**Q4: What batch size should I start with?**
> 32 is a good default. For computer vision, 64–256 is common. For large language models, much larger (1024–8192 with gradient accumulation). If you increase the batch size by N, you should also linearly scale the learning rate by N (linear scaling rule). Always monitor validation performance — sometimes smaller batches generalize better.

**Q5: How does the learning rate finder work?**
> Start with a very small lr (e.g., 1e-7). Train one batch, record loss, increase lr by a constant factor, repeat. Plot loss vs lr. The optimal lr is just before loss starts to explode — typically around 10× smaller than the minimum loss point. This removes much of the guesswork in lr selection.
