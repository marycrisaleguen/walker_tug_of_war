# Unit 2 — Text Preprocessing

## Topics

### 1. The NLP Pipeline
- Raw text → tokenize → normalize → vectorize → model
- Every step affects model performance; garbage in, garbage out

### 2. Tokenization
- Splitting text into tokens (words, subwords, characters)
- Word tokenization: split on whitespace and punctuation
- Subword tokenization (BPE, WordPiece): handles unknown words; used in BERT, GPT
- Character tokenization: very fine-grained; handles typos but long sequences

### 3. Text Normalization
- Lowercasing
- Removing stopwords (the, is, at, …) — depends on task; sometimes hurts
- Removing punctuation, HTML tags, special characters
- **Stemming**: cut word to root form ("running" → "run"); fast but rough
- **Lemmatization**: reduce to dictionary base form ("better" → "good"); more accurate

### 4. Bag of Words (BoW)
- Represent document as vector of word frequencies
- Ignores word order: "dog bites man" = "man bites dog"
- Vocabulary: all unique words in training corpus
- High-dimensional but sparse; works surprisingly well for many tasks

### 5. TF-IDF (Term Frequency-Inverse Document Frequency)
- TF: frequency of word in the document
- IDF: log(N / df) where N = total docs, df = docs containing word
- TF-IDF: TF × IDF — high weight for words frequent in this doc but rare overall
- Downweights common words (like stopwords) automatically

### 6. N-grams
- Sequences of n consecutive words: unigrams (n=1), bigrams (n=2), trigrams (n=3)
- Capture some word order information ("not good" is a bigram different from "good")
- Use with BoW: `CountVectorizer(ngram_range=(1,2))`

---

## Exercises

### Exercise 1 — TF-IDF with Logistic Regression
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load only 4 categories for speed
categories = ['sci.space', 'rec.sport.hockey', 'talk.politics.guns', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers','footers','quotes'))
test  = fetch_20newsgroups(subset='test',  categories=categories, remove=('headers','footers','quotes'))

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=3)),
    ('clf',   LogisticRegression(max_iter=1000, C=1.0))
])
pipeline.fit(train.data, train.target)
y_pred = pipeline.predict(test.data)
print(classification_report(test.target, y_pred, target_names=categories))
```

### Exercise 2 — Custom Tokenization and Vocabulary
```python
import re
from collections import Counter

def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

corpus = [
    "Machine learning is amazing",
    "Deep learning is a subset of machine learning",
    "Natural language processing uses machine learning"
]

# Build vocabulary
all_tokens = [t for doc in corpus for t in tokenize(doc)]
vocab_freq = Counter(all_tokens)
vocab = {word: i for i, word in enumerate(sorted(vocab_freq))}
print("Vocabulary:", vocab)

# Bag-of-words encoding
def bow_encode(doc, vocab):
    vec = [0] * len(vocab)
    for token in tokenize(doc):
        if token in vocab:
            vec[vocab[token]] += 1
    return vec

for doc in corpus:
    print(bow_encode(doc, vocab))
```

---

## Questions & Answers

**Q1: When should I remove stopwords and when should I keep them?**
> Remove stopwords for document classification and information retrieval — they add noise and bloat the vocabulary. Keep them for tasks where word order and function words matter: sentiment analysis ("not good" vs "good"), named entity recognition, machine translation. Modern neural models with embeddings typically don't remove stopwords.

**Q2: What is the difference between stemming and lemmatization?**
> Stemming chops word endings with simple heuristic rules ("studies" → "studi") — fast but can produce non-words. Lemmatization uses a vocabulary and morphological analysis to return the dictionary base form ("studies" → "study", "better" → "good") — more accurate but slower. For fast pipelines, stem; for higher quality, lemmatize.

**Q3: Why does TF-IDF outperform raw term frequency?**
> Raw TF gives high weights to very common words (the, and, is) that appear everywhere and carry no discriminative power. IDF penalizes words that appear in many documents. The product TF-IDF gives high scores to words that are frequent in this document but rare across the corpus — exactly the words that characterize the document's topic.

**Q4: What is subword tokenization and why is it important?**
> Subword tokenization (BPE, WordPiece) splits rare words into subword units ("unhappiness" → "un", "happiness"). This handles out-of-vocabulary words gracefully and keeps the vocabulary size manageable. All modern language models (BERT, GPT) use subword tokenization — character-level models have too long sequences, word-level models can't handle rare/new words.

**Q5: What is the vocabulary size tradeoff?**
> Small vocabulary: many OOV (out-of-vocabulary) words; misses nuance. Large vocabulary: sparse vectors, more memory, rare words are hard to represent. Typical word-level vocab: 30,000–100,000. Subword vocab (BERT): ~30,000 WordPiece tokens covering essentially all text.
