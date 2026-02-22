# Unit 1 — K-Means Clustering

## Topics

### 1. Clustering Problem
- Find natural groups in unlabeled data
- No target variable — purely unsupervised
- Applications: customer segmentation, document clustering, image compression, anomaly detection

### 2. K-Means Algorithm
1. Choose K centroids randomly (or with K-Means++ initialization)
2. Assign each point to the nearest centroid (Euclidean distance)
3. Recompute centroids as the mean of assigned points
4. Repeat until centroids stop moving (convergence)

### 3. Objective Function
- Minimize within-cluster sum of squares (WCSS / inertia):
  J = Σₖ Σ_{x∈Cₖ} ‖x − μₖ‖²
- K-Means finds a local minimum (not guaranteed global)
- Run multiple times with different initializations (`n_init=10`)

### 4. Choosing K: Elbow Method
- Plot inertia vs K; look for "elbow" where inertia stops decreasing rapidly
- No sharp elbow: try silhouette score instead

### 5. Silhouette Score
- For each point: silhouette = (b − a) / max(a, b)
  - a = mean distance to same-cluster points
  - b = mean distance to nearest-cluster points
- Range [−1, 1]; higher is better; average across all points

### 6. Limitations of K-Means
- Assumes spherical (isotropic), equal-size clusters
- Sensitive to initialization and outliers
- Must specify K in advance
- Uses Euclidean distance — poor for high dimensions or non-numeric data

### 7. K-Means++
- Smart initialization: first centroid random; subsequent centroids chosen with probability proportional to distance² from existing centroids
- Faster convergence, better solutions
- Default in sklearn (`init="k-means++"`)

---

## Exercises

### Exercise 1 — K-Means on 2-D Data
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

km = KMeans(n_clusters=4, n_init=10, random_state=42)
labels = km.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=30, alpha=0.7)
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            c='red', marker='*', s=200, label='Centroids')
plt.title("K-Means Clustering (K=4)"); plt.legend(); plt.show()
print(f"Inertia: {km.inertia_:.2f}")
```

### Exercise 2 — Elbow Method + Silhouette
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=500, centers=5, cluster_std=1.0, random_state=0)

inertias, silhouettes = [], []
K_range = range(2, 12)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, km.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(K_range, inertias, marker='o'); ax1.set_title("Elbow Method"); ax1.set_xlabel("K")
ax2.plot(K_range, silhouettes, marker='o', color='green')
ax2.set_title("Silhouette Score"); ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette")
plt.tight_layout(); plt.show()
print(f"Best K by silhouette: {K_range[np.argmax(silhouettes)]}")
```

---

## Questions & Answers

**Q1: Why is K-Means sensitive to initialization?**
> K-Means can converge to different local minima depending on where the centroids start. A bad initialization may result in poor cluster assignments. K-Means++ addresses this by spreading initial centroids out. `n_init=10` runs the algorithm 10 times and keeps the best result.

**Q2: What are the main failure modes of K-Means?**
> (1) Non-spherical clusters (elongated, curved). (2) Very different cluster sizes. (3) Clusters with very different densities. In these cases, use DBSCAN (density-based) or Gaussian Mixture Models instead.

**Q3: Is the elbow method always reliable?**
> No. For well-separated clusters the elbow is clear. For overlapping or complex clusters, there is no obvious elbow. Silhouette score provides a more quantitative measure. In practice, use domain knowledge to guide K selection.

**Q4: Can K-Means handle categorical data?**
> Not directly — K-Means uses Euclidean distance and computes means. For categorical data, use K-Modes (uses modes instead of means) or K-Prototypes (for mixed data). Alternatively, encode categorical variables as numeric and use Euclidean distance with caution.

**Q5: How does K-Means relate to the EM algorithm?**
> K-Means is a special case of the Expectation-Maximization (EM) algorithm applied to a Gaussian Mixture Model with equal, spherical covariances. The "E step" assigns points to clusters (hard assignment in K-Means vs soft probabilistic in GMM). The "M step" updates centroids (means in K-Means vs means + covariances + weights in GMM).
