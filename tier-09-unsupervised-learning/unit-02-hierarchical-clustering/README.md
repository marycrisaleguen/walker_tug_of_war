# Unit 2 — Hierarchical Clustering & DBSCAN

## Topics

### 1. Hierarchical Clustering
- Builds a tree (dendrogram) of nested cluster mergers
- No need to specify K in advance
- Two variants: Agglomerative (bottom-up) and Divisive (top-down)

### 2. Agglomerative Clustering Algorithm
1. Start: each point is its own cluster
2. Compute pairwise distances
3. Merge the two closest clusters
4. Repeat until one cluster remains
5. Cut the dendrogram at a desired number of clusters

### 3. Linkage Methods
- **Single linkage** (min): distance = min distance between any two points
- **Complete linkage** (max): distance = max distance between any two points
- **Average linkage**: distance = mean of all pairwise distances
- **Ward's linkage**: merges clusters minimizing variance increase — most common

### 4. Choosing Number of Clusters from Dendrogram
- Look for the largest vertical "gap" in the dendrogram
- Cut at the height that maximizes the gap
- Or use a pre-specified number: `n_clusters=K`

### 5. DBSCAN (Density-Based Spatial Clustering)
- Groups points that are closely packed; marks sparse outliers as noise
- Parameters: ε (neighborhood radius), min_samples (core point threshold)
- Advantages: finds arbitrary shapes, handles outliers, no K to specify
- Disadvantage: sensitive to ε and min_samples; struggles with varying densities

### 6. Core, Border, and Noise Points
- **Core point**: has ≥ min_samples points within radius ε
- **Border point**: within ε of a core point but not a core point itself
- **Noise point**: not within ε of any core point (labeled −1)

---

## Exercises

### Exercise 1 — Dendrogram Visualization
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=30, centers=3, cluster_std=0.5, random_state=42)
Z = linkage(X, method='ward')

plt.figure(figsize=(12, 5))
dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
plt.title("Ward Linkage Dendrogram")
plt.xlabel("Sample Index"); plt.ylabel("Distance")
plt.tight_layout(); plt.show()
```

### Exercise 2 — Agglomerative Clustering vs K-Means
```python
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

X, y_true = make_moons(n_samples=200, noise=0.05, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, model, title in [
    (axes[0], KMeans(n_clusters=2, n_init=10, random_state=42),            "K-Means"),
    (axes[1], AgglomerativeClustering(n_clusters=2, linkage='ward'),       "Ward Linkage"),
    (axes[2], AgglomerativeClustering(n_clusters=2, linkage='single'),     "Single Linkage"),
]:
    labels = model.fit_predict(X)
    ax.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', s=20)
    ax.set_title(f"{title}\nARI={adjusted_rand_score(y_true, labels):.3f}")
plt.tight_layout(); plt.show()
```

### Exercise 3 — DBSCAN on Non-Spherical Data
```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, y_true = make_moons(n_samples=300, noise=0.05, random_state=42)
X_s = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.3, min_samples=5)
labels = db.fit_predict(X_s)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
print(f"Clusters found: {n_clusters}, Noise points: {n_noise}")

plt.scatter(X_s[:,0], X_s[:,1], c=labels, cmap='tab10', s=20)
plt.title(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")
plt.show()
```

---

## Questions & Answers

**Q1: What is the main advantage of hierarchical clustering over K-Means?**
> You do not need to specify K in advance. The dendrogram shows the full hierarchy of mergers, letting you choose K by inspection. It also naturally captures nested structure. Drawback: O(n²) memory and O(n² log n) compute — does not scale to large datasets.

**Q2: When should I use DBSCAN instead of K-Means?**
> Use DBSCAN when: (1) clusters have arbitrary, non-spherical shapes (moons, rings), (2) you want automatic outlier detection (noise points), (3) you do not know K in advance. Use K-Means when clusters are roughly spherical and similar in size.

**Q3: How do I choose ε (epsilon) for DBSCAN?**
> Plot the sorted k-NN distance graph (k = min_samples − 1). Find the "knee" — the point where distance starts to increase rapidly. That distance is a good starting ε. Alternatively, use domain knowledge about the expected cluster density.

**Q4: What is the Adjusted Rand Index (ARI)?**
> ARI measures agreement between two clusterings (or a clustering vs ground truth labels). ARI = 0 for random clustering, ARI = 1 for perfect agreement, ARI can be negative for clusterings worse than random. It corrects for chance, making it more reliable than the raw Rand Index for comparing clusterings.

**Q5: Can hierarchical clustering be applied to non-numeric data?**
> Yes, as long as you can define a pairwise distance/dissimilarity between samples. For text, you can use cosine similarity. For categorical data, Hamming distance. The linkage matrix operates on pairwise distances, so any valid distance metric works with `scipy.cluster.hierarchy.linkage`.
