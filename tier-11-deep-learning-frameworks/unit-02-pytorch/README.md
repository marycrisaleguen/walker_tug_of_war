# Unit 2 — PyTorch

## Topics

### 1. PyTorch Tensors
- `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.randn()`
- Tensor operations mirror NumPy but support GPU (`.cuda()` / `.to(device)`)
- Automatic differentiation: set `requires_grad=True` to track operations

### 2. Autograd
- `tensor.backward()`: compute gradients with respect to all `requires_grad=True` tensors
- `tensor.grad`: access stored gradient
- `torch.no_grad()`: disable autograd (saves memory during inference)
- `optimizer.zero_grad()`: clear previous gradients before each backward pass

### 3. Defining Models with `nn.Module`
- Subclass `nn.Module`, define layers in `__init__`, implement `forward()`
- Use `nn.Linear`, `nn.Conv2d`, `nn.LSTM`, `nn.Embedding`, etc.
- `model.parameters()` returns all learnable parameters

### 4. Training Loop (Manual)
```python
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
```

### 5. PyTorch Data Utilities
- `Dataset`: abstract class — implement `__len__` and `__getitem__`
- `DataLoader`: batches data, shuffles, supports multi-process loading
- `torchvision.datasets`: MNIST, CIFAR-10, ImageNet, etc.

### 6. Device Management (CPU vs GPU)
- `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Move model: `model.to(device)`
- Move data: `X.to(device)`, `y.to(device)`

---

## Exercises

### Exercise 1 — MLP from scratch in PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

X_train_t = torch.FloatTensor(X_tr_s)
y_train_t  = torch.FloatTensor(y_tr).unsqueeze(1)
X_test_t   = torch.FloatTensor(X_te_s)
y_test_t   = torch.FloatTensor(y_te).unsqueeze(1)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = MLP(X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_t)
    loss  = criterion(preds, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test_t)
            val_acc = ((val_preds > 0.5).float() == y_test_t).float().mean()
        print(f"Epoch {epoch+1}: loss={loss.item():.4f}, val_acc={val_acc:.4f}")
```

### Exercise 2 — Custom Dataset and DataLoader
```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
X_tr_s = StandardScaler().fit_transform(X_tr)

train_ds = TabularDataset(X_tr_s, y_tr)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

for X_batch, y_batch in train_loader:
    print(f"Batch shapes: X={X_batch.shape}, y={y_batch.shape}")
    break
```

---

## Questions & Answers

**Q1: What is the difference between PyTorch and TensorFlow/Keras?**
> PyTorch uses a dynamic computation graph (define-by-run) — the graph is built at runtime, making debugging natural (use Python's print/pdb). TF 2.x with eager mode also supports dynamic graphs, but Keras's high-level API hides the training loop. PyTorch requires writing the training loop explicitly, giving more control. PyTorch dominates research; TF/Keras is popular in production and industry.

**Q2: Why must I call `optimizer.zero_grad()` before each backward pass?**
> PyTorch accumulates gradients in `.grad` by default (useful for certain tricks like gradient accumulation). If you don't zero them, gradients from previous batches will add to the current batch's gradients, effectively computing the wrong update. Always call `zero_grad()` at the start of each training step.

**Q3: What is the difference between `model.train()` and `model.eval()`?**
> `model.train()` enables training-mode behaviors: Dropout randomly drops neurons, BatchNorm uses batch statistics. `model.eval()` disables Dropout and uses running statistics in BatchNorm. Always switch to `.eval()` before inference and back to `.train()` before the next training step.

**Q4: What is `torch.no_grad()` and when should I use it?**
> `torch.no_grad()` disables gradient tracking within its context. This saves memory and speeds up computation. Use it whenever you don't need gradients: during validation loops, test inference, and when computing metrics after training.

**Q5: What is gradient accumulation and when is it useful?**
> Gradient accumulation computes gradients over multiple mini-batches before calling `optimizer.step()`, effectively simulating a larger batch size. Useful when GPU memory limits you to small batches but you want the benefits of large-batch training. Every N steps: accumulate gradients without zeroing, then call `optimizer.step()` and `optimizer.zero_grad()`.
