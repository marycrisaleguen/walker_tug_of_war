# Unit 3 — PCA & Dimensionality Reduction

## Topics

### 1. Motivation for Dimensionality Reduction
- Curse of dimensionality: models become less effective as features increase
- Visualization: reduce to 2D/3D for plotting
- Noise reduction: remove low-variance dimensions
- Compression and speed: fewer features → faster models

### 2. Principal Component Analysis (PCA)
- Find orthogonal axes (principal components) that capture maximum variance
- PC₁: direction of greatest variance; PC₂: orthogonal to PC₁, next greatest variance; …
- Derived from eigendecomposition of the covariance matrix (or SVD of the data matrix)

### 3. PCA Steps
1. Center the data: subtract mean of each feature
2. Compute covariance matrix Σ = (1/n) XᵀX
3. Eigendecompose: Σ = VΛVᵀ
4. Project: X_pca = X · V[:, :k]  (keep k eigenvectors with largest eigenvalues)

### 4. Explained Variance Ratio
- Each PC's eigenvalue = variance along that PC
- Explained variance ratio = λᵢ / Σλⱼ
- `pca.explained_variance_ratio_`
- Plot cumulative explained variance vs number of components

### 5. Choosing Number of Components
- Keep enough components to explain ~90–95% of variance
- Use the cumulative explained variance curve ("knee")
- Alternatively: choose k to satisfy a downstream task performance criterion

### 6. Limitations of PCA
- Linear: cannot capture non-linear structure
- Components are not interpretable (linear combinations of all features)
- Not ideal for sparse data (e.g., text bag-of-words → use Truncated SVD/LSA)

### 7. Non-Linear Dimensionality Reduction
- **t-SNE**: preserves local structure; excellent for 2D visualization; not reusable on new data
- **UMAP**: faster than t-SNE, preserves more global structure, can transform new data
- Both are mainly used for visualization, not as preprocessing for downstream models

---

## Exercises

### Exercise 1 — PCA on the Digits Dataset
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, y = load_digits(return_X_y=True)
X_s = StandardScaler().fit_transform(X)

pca_full = PCA().fit(X_s)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.plot(cumvar, marker='.')
plt.axhline(0.95, color='red', linestyle='--', label='95% variance')
plt.xlabel("Number of Components"); plt.ylabel("Cumulative Explained Variance")
plt.title("PCA: Explained Variance"); plt.legend(); plt.show()

k = np.searchsorted(cumvar, 0.95) + 1
print(f"Components needed for 95% variance: {k} (of {X_s.shape[1]} original)")
```

### Exercise 2 — PCA for 2D Visualization
```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, y = load_iris(return_X_y=True)
names = load_iris().target_names

X_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))

plt.figure(figsize=(7, 5))
for i, name in enumerate(names):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, s=40)
plt.xlabel("PC 1"); plt.ylabel("PC 2")
plt.title("Iris Dataset: PCA 2D Projection"); plt.legend(); plt.show()
```

### Exercise 3 — t-SNE Visualization
```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)
# Run on a subset for speed
X_sub, y_sub = X[:1000], y[:1000]

X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_sub)

plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_sub, cmap='tab10', s=10, alpha=0.8)
plt.colorbar(scatter, label="Digit")
plt.title("t-SNE Visualization of Digits Dataset"); plt.show()
```

---

## Questions & Answers

**Q1: Should I scale before PCA?**
> Yes, almost always. PCA is sensitive to feature scales: a feature with range [0, 1000] will dominate a feature with range [0, 1], and the first PC will mostly capture the high-scale feature's variance. Standardize first so all features contribute equally.

**Q2: What is the relationship between PCA and SVD?**
> PCA via eigendecomposition of the covariance matrix is equivalent to computing the SVD of the centered data matrix X = UΣVᵀ. The principal components are the right singular vectors V; the explained variance per component is (σᵢ²)/(n−1). sklearn's PCA internally uses SVD for numerical stability.

**Q3: Can I use PCA to remove noise?**
> Yes. PCA concentrates variance in the first few components. The later components correspond to low-variance directions that often capture noise. Reconstructing the data from only the first k components (inverse_transform) filters out the noise dimensions.

**Q4: What is the difference between PCA and t-SNE for visualization?**
> PCA is a linear method — it finds the directions of maximum global variance and projects linearly. t-SNE is non-linear and focuses on preserving local neighborhoods. PCA is deterministic, fast, and can transform new data. t-SNE is stochastic, slow, and cannot straightforwardly transform new data. Use PCA first to reduce to 50 dimensions, then t-SNE for visualization.

**Q5: Why shouldn't I use t-SNE projections as input features for a classifier?**
> t-SNE is non-deterministic and optimized for 2D visualization, not for preserving the information needed for classification. The 2D embedding may cluster visually but lose important discriminative information. Use PCA or autoencoders for feature extraction and dimensionality reduction before a classifier.
