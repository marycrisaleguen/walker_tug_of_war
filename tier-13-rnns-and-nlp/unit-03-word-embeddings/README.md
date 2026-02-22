# Unit 3 — Word Embeddings

## Topics

### 1. The Problem with BoW
- Sparse, high-dimensional vectors
- No notion of word similarity: "dog" and "puppy" are orthogonal
- No word order information

### 2. Word Embeddings
- Dense, low-dimensional (50–300d) vector representations
- Similar words have similar vectors: cosine_sim("king", "queen") > cosine_sim("king", "car")
- Learned from large corpora by predicting word contexts

### 3. Word2Vec
- **Skip-gram**: predict context words from center word
- **CBOW** (Continuous Bag of Words): predict center word from context words
- Objective: maximize P(context | center) = softmax over vocabulary
- Famous analogy: king − man + woman ≈ queen

### 4. GloVe (Global Vectors)
- Factorizes the co-occurrence matrix (how often do words appear together?)
- Combines global corpus statistics with local context information
- Often produces slightly smoother semantic space than Word2Vec

### 5. FastText
- Represents words as sum of character n-gram embeddings
- Handles morphologically rich languages and out-of-vocabulary words
- "playing" = "play" + "playing" + "ing" + "n-grams"

### 6. Embedding Layers in Neural Networks
- `tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=seq_len)`
- Learns embeddings jointly with the task (task-specific)
- Initialize with pretrained weights (GloVe, Word2Vec) and optionally fine-tune

### 7. Contextual Embeddings
- Word2Vec, GloVe: static — same vector for "bank" regardless of context
- Contextual: ELMo, BERT — different vectors for "bank" in "river bank" vs "bank account"
- Covered in depth in Tier 14 (Transformers)

---

## Exercises

### Exercise 1 — Train Word2Vec Embeddings
```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

sentences = [
    "machine learning is a subset of artificial intelligence",
    "deep learning is a type of machine learning",
    "neural networks are the foundation of deep learning",
    "natural language processing deals with text and speech",
    "computer vision processes images and videos",
    "king and queen are royalty words in embeddings",
    "man is to woman as king is to queen"
]

tokenized = [simple_preprocess(sent) for sent in sentences]
model = Word2Vec(tokenized, vector_size=50, window=3, min_count=1, epochs=200, seed=42)

# Word similarity
print("Most similar to 'machine':", model.wv.most_similar("machine", topn=5))
print("Most similar to 'learning':", model.wv.most_similar("learning", topn=5))
print("\nAnalogy king-man+woman:", model.wv.most_similar(
    positive=["king", "woman"], negative=["man"], topn=3))
```

### Exercise 2 — Use GloVe Embeddings in Keras
```python
import numpy as np
import tensorflow as tf

# Simulated GloVe loading (in practice, download glove.6B.50d.txt)
# This creates random embeddings as a placeholder
vocab_size, embed_dim = 10000, 50
np.random.seed(42)
embedding_matrix = np.random.randn(vocab_size, embed_dim).astype("float32")

# Build model with pretrained embedding layer (non-trainable)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim, weights=[embedding_matrix],
                               trainable=False, input_length=200),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

### Exercise 3 — Visualize Word Embeddings with PCA
```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Using model from Exercise 1
words = ["machine", "learning", "deep", "neural", "text", "image", "king", "queen"]
vectors = np.array([model.wv[w] for w in words])

pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

plt.figure(figsize=(8, 6))
plt.scatter(coords[:,0], coords[:,1])
for word, (x,y) in zip(words, coords):
    plt.annotate(word, (x, y), fontsize=12)
plt.title("Word Embeddings — PCA 2D Projection"); plt.grid(True, alpha=0.3); plt.show()
```

---

## Questions & Answers

**Q1: Why are word embeddings so much better than one-hot encoding?**
> One-hot vectors are orthogonal — "cat" and "dog" are equally different as "cat" and "algebra." Embeddings encode semantic relationships: similar words have similar vectors. The dense, low-dimensional representation also reduces memory and computation compared to 50,000-dimensional sparse one-hot vectors.

**Q2: How does Word2Vec learn embeddings?**
> Word2Vec trains a shallow neural network to predict context words from a target word (skip-gram) or vice versa (CBOW). The input is the one-hot word vector; the weight matrix of the hidden layer becomes the embedding matrix. After training, the task (word prediction) is discarded and only the embeddings are kept.

**Q3: What is the famous "king − man + woman = queen" analogy?**
> It demonstrates that word embeddings encode semantic relationships as geometric offsets. The vector from "man" to "woman" is roughly the same as from "king" to "queen" (gender offset). Other analogies: Paris − France + Germany ≈ Berlin (capital-country offset), "bigger" − "big" ≈ "taller" − "tall" (comparative offset).

**Q4: Should I train embeddings from scratch or use pretrained?**
> Use pretrained for small datasets — they encode rich semantic knowledge from billions of words of text. Train from scratch (or fine-tune pretrained) when you have large domain-specific data (medical, legal, code). With contextual models like BERT (Tier 14), fine-tuning the whole model on your task is standard.

**Q5: What is the difference between static and contextual embeddings?**
> Static embeddings (Word2Vec, GloVe) assign one fixed vector per word regardless of context. "Bank" has the same embedding in "river bank" and "bank account." Contextual embeddings (ELMo, BERT) produce different vectors for the same word in different contexts, enabling much richer semantic representations. This is a key advantage of Transformer-based models.
