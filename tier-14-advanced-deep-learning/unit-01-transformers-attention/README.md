# Unit 1 — Transformers & Attention

## Topics

### 1. Motivation: Problems with RNNs
- Sequential processing: cannot parallelize across timesteps
- Difficulty with very long sequences even with LSTM
- Attention mechanisms (added to RNNs in 2015) showed that direct connections between any two positions help

### 2. The Attention Mechanism
- For each output position, compute a weighted sum of all input positions
- Scores: e(query, key) = (Q · Kᵀ) / √d_k  (scaled dot-product attention)
- Weights: softmax(scores) — attend more to relevant positions
- Output: weights × Values

### 3. Self-Attention
- Query, Key, and Value all come from the same sequence
- Each word attends to every other word in the sequence
- Captures long-range dependencies in O(1) steps (vs O(n) for RNN)

### 4. Multi-Head Attention
- Run multiple attention heads in parallel with different learned projections
- Each head attends to different aspects (syntax, semantics, coreference)
- Concatenate and project: MultiHead(Q,K,V) = Concat(head₁,...,headₕ)Wᴼ

### 5. Positional Encoding
- Self-attention is permutation-invariant — order doesn't matter by default
- Add positional encodings to embed position information
- Sinusoidal: PE(pos,2i) = sin(pos/10000^{2i/d}), PE(pos,2i+1) = cos(...)
- Learned positional embeddings (alternative approach)

### 6. The Transformer Architecture
- **Encoder**: stack of N identical layers, each with multi-head attention + FFN + layer norm
- **Decoder**: similar stack, with masked self-attention + cross-attention to encoder
- "Attention Is All You Need" (Vaswani et al., 2017)

### 7. BERT and GPT
- **BERT** (Bidirectional Encoder): encoder only; pretrained on masked language modeling + next sentence prediction; fine-tuned for NLP tasks
- **GPT** (Generative Pretrained Transformer): decoder only; pretrained on language modeling (predict next token); used for generation
- Both are pretrained on massive corpora (Wikipedia, BookCorpus, web text)

---

## Exercises

### Exercise 1 — Scaled Dot-Product Attention from Scratch
```python
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Q: (seq_len, d_k)
    K: (seq_len, d_k)
    V: (seq_len, d_v)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V
    return output, weights

np.random.seed(42)
seq_len, d_k, d_v = 5, 8, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
print("Attention output shape:", output.shape)
print("Attention weights (sum per row = 1):", weights.sum(axis=-1))
```

### Exercise 2 — Fine-Tune BERT with Hugging Face Transformers
```python
# pip install transformers datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Load a small sentiment dataset (SST-2)
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=128, padding="max_length")

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

args = TrainingArguments(
    output_dir="/tmp/sst2_bert",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=tokenized["train"].select(range(2000)),
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics
)
trainer.train()
```

---

## Questions & Answers

**Q1: What problem does self-attention solve compared to RNNs?**
> RNNs process sequences step-by-step — the signal from a word 100 positions ago must pass through 100 hidden states before influencing the current step. Self-attention creates direct connections between any two positions, letting the model capture long-range dependencies in a single operation. It is also parallelizable unlike RNN's sequential computation.

**Q2: What is the query-key-value (QKV) analogy?**
> Think of a database: you send a **query** (what you're looking for), compare it against all **keys** (what's available), and retrieve the **values** (the actual content) weighted by similarity. In self-attention, each position generates its own Q, K, V from the same input sequence, allowing each word to "query" all other words for relevant information.

**Q3: Why scale by √d_k in attention?**
> For large d_k, the dot products grow in magnitude. Large values push softmax into regions with near-zero gradients (saturation). Scaling by √d_k keeps the magnitudes reasonable. This is equivalent to keeping the variance of the dot product at 1.

**Q4: What is the difference between BERT and GPT?**
> BERT uses only the encoder stack (bidirectional). It is pretrained by masking random words and predicting them, and is fine-tuned for understanding tasks (classification, QA). GPT uses only the decoder stack (causal/left-to-right). It is pretrained to predict the next token, making it naturally suited for generation tasks (text completion, dialogue).

**Q5: What is a "foundation model"?**
> A large pretrained model (BERT, GPT-3, LLaMA, Stable Diffusion) trained on massive data at great computational cost, designed to be fine-tuned or prompted for many downstream tasks without training from scratch. The term reflects that these models form the "foundation" of modern AI applications. They enable high performance on new tasks with very little task-specific data.
