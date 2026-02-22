# Unit 3 — Matplotlib & Seaborn

## Topics

### 1. Matplotlib Basics
- `plt.figure()`, `plt.subplot()`
- `plt.plot()`, `plt.scatter()`, `plt.bar()`, `plt.hist()`
- Axes labels: `plt.xlabel()`, `plt.ylabel()`, `plt.title()`
- `plt.legend()`, `plt.grid()`, `plt.tight_layout()`
- Saving: `plt.savefig("plot.png", dpi=150)`

### 2. Figure and Axes API (OO style)
- `fig, ax = plt.subplots(nrows, ncols, figsize=(w, h))`
- `ax.plot()`, `ax.set_xlabel()`, `ax.set_title()`
- Preferred over `plt.xxx` for production-quality code

### 3. Common Plot Types in ML
- **Line plot**: training/validation loss curves
- **Scatter plot**: feature relationships, decision boundaries
- **Histogram**: feature distributions
- **Heatmap** (Seaborn): correlation matrix, confusion matrix
- **Box plot**: distribution by category
- **Pair plot** (Seaborn): relationships between all feature pairs

### 4. Seaborn
- High-level interface built on Matplotlib
- `sns.histplot()`, `sns.boxplot()`, `sns.scatterplot()`, `sns.heatmap()`
- `sns.pairplot()` for exploratory data analysis
- Built-in themes: `sns.set_theme()`

### 5. Visualization Best Practices
- Label every axis with units
- Use color deliberately (colorblind-friendly palettes)
- Keep it simple: remove chartjunk
- Always save at high resolution for reports

---

## Exercises

### Exercise 1 — Line Plot (Loss Curve)
```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
epochs = np.arange(1, 51)
train_loss = 1.0 * np.exp(-0.1 * epochs) + 0.05*np.random.randn(50)
val_loss   = 1.0 * np.exp(-0.08 * epochs) + 0.08*np.random.randn(50)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, train_loss, label="Train Loss")
ax.plot(epochs, val_loss,   label="Val Loss", linestyle="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training vs Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/loss_curve.png", dpi=150)
plt.show()
```

### Exercise 2 — Histogram and Scatter
```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
heights = np.random.normal(170, 10, 500)
weights = heights * 0.5 + np.random.normal(0, 5, 500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.hist(heights, bins=30, edgecolor='black', color='steelblue')
ax1.set_title("Height Distribution")
ax1.set_xlabel("Height (cm)")
ax1.set_ylabel("Count")

ax2.scatter(heights, weights, alpha=0.4, s=10, color='coral')
ax2.set_title("Height vs Weight")
ax2.set_xlabel("Height (cm)")
ax2.set_ylabel("Weight (kg)")

plt.tight_layout()
plt.show()
```

### Exercise 3 — Seaborn Heatmap (Correlation Matrix)
```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
df = pd.DataFrame(np.random.randn(100, 5),
                  columns=["Feature A","Feature B","Feature C","Feature D","Target"])
df["Target"] = df["Feature A"] * 2 + df["Feature C"] + np.random.randn(100)*0.5

corr = df.corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
```

---

## Questions & Answers

**Q1: What is the difference between `plt.plot()` (state-machine API) and the OO API?**
> The state-machine API (`plt.plot()`) operates on the "current" figure and axes implicitly. The OO API (`fig, ax = plt.subplots(); ax.plot()`) gives you explicit references to figure and axes objects. The OO API is preferred when creating multiple subplots, writing reusable functions, or building complex visualizations.

**Q2: Why should I use `plt.tight_layout()`?**
> `tight_layout()` automatically adjusts subplot parameters to prevent labels from overlapping. It is a good habit to call it before `savefig()` or `show()` to avoid cut-off titles or axis labels.

**Q3: What colormap should I use for a heatmap?**
> For diverging data centered at zero (e.g., correlation matrices), use `"coolwarm"` or `"RdBu_r"`. For sequential data (e.g., model accuracy), use `"viridis"` or `"plasma"` — these are perceptually uniform and colorblind-friendly. Avoid `"jet"` as it is misleading.

**Q4: When should I use Seaborn instead of Matplotlib directly?**
> Seaborn is best for statistical visualizations (distributions, relationships, categorical plots) with a beautiful default aesthetic. Matplotlib gives more control over every element. A common pattern is to create the figure with Matplotlib, add Seaborn plots, then customize with Matplotlib methods.

**Q5: How do I visualize a decision boundary for a 2-D classifier?**
> Create a meshgrid of (x, y) values over the input space, run `model.predict()` on all points, then plot the predictions as a filled contour plot (`plt.contourf()`). Overlay the actual data points as a scatter plot to see how well the boundary separates the classes.
