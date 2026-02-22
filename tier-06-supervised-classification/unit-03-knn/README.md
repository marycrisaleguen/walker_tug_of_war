# Unit 3 — K-Nearest Neighbors (KNN)

## Topics

### 1. Intuition
- "Tell me who your neighbors are, and I'll tell you who you are"
- Classify a new point by majority vote of its k nearest training examples
- For regression: average of k nearest neighbors' target values

### 2. Distance Metrics
- **Euclidean**: d = √(Σ(xᵢ − yᵢ)²) — most common, assumes isotropic space
- **Manhattan**: d = Σ|xᵢ − yᵢ| — less sensitive to outliers
- **Minkowski**: generalizes both (p=1: Manhattan, p=2: Euclidean)
- **Cosine similarity**: used for text and high-dimensional data

### 3. Choosing K
- Small K: flexible, low bias, high variance (overfitting)
- Large K: smooth, high bias, low variance (underfitting)
- Use cross-validation to select K; odd K avoids ties in binary classification

### 4. The Curse of Dimensionality
- In high dimensions, all points are equidistant from any query point
- KNN degrades as dimensionality increases
- Solution: dimensionality reduction (PCA, feature selection) before KNN

### 5. Practical Considerations
- **Must scale features** (KNN is distance-based)
- Prediction is O(n · d) per query — slow for large datasets
- Approximate nearest neighbor libraries (FAISS, Annoy) speed up large-scale KNN

---

## Exercises

### Exercise 1 — KNN Classifier
```python
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)
print(classification_report(y_test, knn.predict(X_test_s)))
```

### Exercise 2 — Selecting K with Cross-Validation
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_s = StandardScaler().fit_transform(X)

k_values = range(1, 31)
cv_scores = [cross_val_score(KNeighborsClassifier(k), X_s, y, cv=5).mean()
             for k in k_values]

plt.plot(k_values, cv_scores, marker='o')
plt.xlabel("K"); plt.ylabel("CV Accuracy")
plt.title("KNN: K vs Cross-Validation Accuracy")
plt.axvline(k_values[np.argmax(cv_scores)], color='red', linestyle='--',
            label=f"Best K={k_values[np.argmax(cv_scores)]}")
plt.legend(); plt.show()
```

### Exercise 3 — Effect of Scaling on KNN
```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
knn = KNeighborsClassifier(n_neighbors=5)

raw_acc   = cross_val_score(knn, X,                             y, cv=5).mean()
scaled_acc = cross_val_score(knn, StandardScaler().fit_transform(X), y, cv=5).mean()

print(f"Without scaling: {raw_acc:.4f}")
print(f"With scaling:    {scaled_acc:.4f}")
```

---

## Questions & Answers

**Q1: Is KNN a parametric or non-parametric model?**
> KNN is non-parametric — it makes no assumptions about the functional form of the decision boundary. The "model" is the entire training dataset stored in memory. At prediction time, it searches the training set for the nearest neighbors. This contrasts with parametric models (linear regression, logistic regression) that summarize the data into a fixed number of parameters.

**Q2: Why does KNN degrade in high dimensions?**
> In high-dimensional spaces, the volume of a hypersphere grows much slower than that of a hypercube. As a result, "nearest neighbors" are not meaningfully closer than "farthest neighbors" — distances become uniform. This means the neighborhood is no longer a local, representative sample of the target function.

**Q3: What is the effect of K on the decision boundary?**
> K=1 creates an extremely irregular "Voronoi" boundary that memorizes every training point. Large K smooths the boundary into nearly a linear one. The optimal K depends on how smooth the true decision boundary is in your problem.

**Q4: Can KNN be used for regression?**
> Yes. `KNeighborsRegressor` predicts the average of the k nearest neighbors' target values. This is useful for non-parametric function approximation but shares all of KNN's scalability and curse-of-dimensionality issues.

**Q5: What are the main drawbacks of KNN in production?**
> (1) Slow prediction: O(n·d) per query. (2) Large memory: stores all training data. (3) Sensitive to irrelevant features. (4) Requires feature scaling. For large-scale applications, approximate nearest neighbor search (FAISS, ScaNN) or indexing structures (KD-trees, Ball Trees) are used.
