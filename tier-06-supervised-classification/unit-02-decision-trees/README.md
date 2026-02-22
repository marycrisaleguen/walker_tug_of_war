# Unit 2 — Decision Trees

## Topics

### 1. Intuition
- A flowchart of yes/no questions that recursively partition the feature space
- Each internal node tests one feature; each leaf predicts a class (or value)
- Non-parametric: no assumption about data distribution

### 2. Splitting Criteria
- **Gini Impurity**: Gini(t) = 1 − Σ pᵢ²  (0 = pure, 0.5 = maximum impurity for binary)
- **Entropy**: H(t) = −Σ pᵢ log₂(pᵢ)
- **Information Gain**: IG = H(parent) − Σ (nchild/nparent) · H(child)
- For regression: variance reduction (MSE of split)

### 3. Building a Tree (CART Algorithm)
- At each node, search all features and all thresholds for the split with best criterion
- Recurse on left and right subsets until stopping criterion met
- Stopping: max depth, min samples per leaf, min impurity improvement

### 4. Overfitting in Trees
- A fully grown tree memorizes training data (each leaf = one sample)
- Control with: `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`

### 5. Pruning
- Post-pruning: grow full tree, then remove branches that don't improve validation performance
- sklearn: `ccp_alpha` (cost-complexity pruning parameter)

### 6. Feature Importance
- Mean decrease in impurity (MDI): sum of impurity reduction weighted by fraction of samples
- `model.feature_importances_`

---

## Exercises

### Exercise 1 — Build and Visualize a Decision Tree
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
print(f"Train accuracy: {accuracy_score(y_train, tree.predict(X_train)):.4f}")
print(f"Test  accuracy: {accuracy_score(y_test,  tree.predict(X_test)):.4f}")

fig, ax = plt.subplots(figsize=(18, 6))
plot_tree(tree, feature_names=load_iris().feature_names,
          class_names=load_iris().target_names, filled=True, ax=ax)
plt.tight_layout(); plt.show()
```

### Exercise 2 — Depth vs Accuracy (Overfitting)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

depths = range(1, 20)
train_acc, test_acc = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, t.predict(X_train)))
    test_acc.append(accuracy_score(y_test,  t.predict(X_test)))

plt.plot(depths, train_acc, label="Train")
plt.plot(depths, test_acc,  label="Test", linestyle='--')
plt.xlabel("Max Depth"); plt.ylabel("Accuracy")
plt.title("Decision Tree: Depth vs Accuracy"); plt.legend(); plt.show()
```

---

## Questions & Answers

**Q1: What is the difference between Gini Impurity and Entropy?**
> Both measure class impurity at a node. Gini is computationally cheaper (no logarithm). Entropy can be slightly more balanced in practice. In most problems, the two criteria produce nearly identical trees. sklearn's default is Gini.

**Q2: Why do decision trees tend to overfit?**
> A fully grown tree will partition the space until each leaf contains exactly one training sample, perfectly memorizing the training labels including noise. This is extreme overfitting (high variance). Limiting tree depth, requiring minimum samples per leaf, or pruning constrains the tree.

**Q3: Are decision trees sensitive to feature scaling?**
> No. The algorithm only looks at the ordering of feature values to find split thresholds. Multiplying a feature by 100 does not change which threshold is optimal — it just scales all thresholds by 100. This makes decision trees (and ensembles built on them) robust to scale differences.

**Q4: Can a decision tree handle missing values?**
> Standard CART (sklearn's implementation) does not handle missing values natively — you must impute first. Some implementations like XGBoost and LightGBM have built-in missing value handling.

**Q5: What does feature importance measure in a decision tree?**
> Feature importance is the sum of impurity reductions (weighted by the number of samples) at each node where the feature is used. It is normalized to sum to 1. A feature with high importance reduces impurity a lot across many training samples. Caution: MDI importance is biased toward high-cardinality features.
