# Unit 1 — RNN / LSTM / GRU

## Topics

### 1. The Sequential Problem
- Some data has inherent temporal or sequential order: text, time series, speech, video
- Feedforward networks process inputs independently — no memory of past inputs
- Recurrent networks pass a hidden state from one timestep to the next

### 2. Vanilla RNN
- hₜ = tanh(Wₕ hₜ₋₁ + Wₓ xₜ + b)
- Hidden state hₜ summarizes "memory" of all previous inputs
- Problem: vanishing/exploding gradients over long sequences — cannot capture long-range dependencies

### 3. Long Short-Term Memory (LSTM)
- Adds a cell state Cₜ alongside the hidden state hₜ
- Three gates control information flow:
  - **Forget gate**: what to erase from cell state — fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
  - **Input gate**: what new info to add — iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
  - **Output gate**: what to output — oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
- Cell state update: Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ tanh(Wc · [hₜ₋₁, xₜ] + bc)
- Hidden state: hₜ = oₜ ⊙ tanh(Cₜ)

### 4. Gated Recurrent Unit (GRU)
- Simplified LSTM: two gates (reset and update), no separate cell state
- Fewer parameters than LSTM, often comparable performance
- Update gate: zₜ = σ(Wz · [hₜ₋₁, xₜ])
- Reset gate: rₜ = σ(Wr · [hₜ₋₁, xₜ])
- New hidden: hₜ = (1−zₜ) ⊙ hₜ₋₁ + zₜ ⊙ tanh(Wh · [rₜ ⊙ hₜ₋₁, xₜ])

### 5. Sequence-to-Sequence Models
- Encoder: processes input sequence, outputs context vector
- Decoder: generates output sequence from context vector
- Applications: machine translation, summarization, speech recognition

### 6. Bidirectional RNNs
- Process sequence both forward and backward
- Output at each step: concatenation of forward and backward hidden states
- Useful when full sequence is available (NLP classification, named entity recognition)

---

## Exercises

### Exercise 1 — LSTM for Sentiment Classification
```python
import tensorflow as tf
import numpy as np

# IMDB movie review sentiment dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# Pad sequences to same length
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=200)
X_test  = tf.keras.preprocessing.sequence.pad_sequences(X_test,  maxlen=200)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 64, input_length=200),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train, epochs=5, batch_size=64,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=1
)
print(f"Test accuracy: {model.evaluate(X_test, y_test, verbose=0)[1]:.4f}")
```

### Exercise 2 — Time Series Prediction with LSTM
```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate sine wave
t = np.linspace(0, 20*np.pi, 1000)
series = np.sin(t) + 0.1*np.random.randn(1000)

# Create sequences
def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X)[..., np.newaxis], np.array(y)

seq_len = 50
X, y = make_sequences(series, seq_len)
split = int(0.8 * len(X))
X_tr, y_tr = X[:split], y[:split]
X_te, y_te = X[split:], y[split:]

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(seq_len, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_tr, y_tr, epochs=10, batch_size=32, verbose=0)

y_pred = model.predict(X_te, verbose=0)
plt.plot(y_te[:200], label="True"); plt.plot(y_pred[:200], label="Predicted", linestyle='--')
plt.title("LSTM Time Series Prediction"); plt.legend(); plt.show()
```

---

## Questions & Answers

**Q1: Why do vanilla RNNs struggle with long-range dependencies?**
> During backpropagation through time (BPTT), gradients are multiplied at each timestep by the recurrent weight matrix W. If ‖W‖ < 1, gradients vanish exponentially with sequence length. If ‖W‖ > 1, they explode. In practice, vanishing dominates: gradients from events 50+ timesteps ago become negligible, making it impossible to learn long-range dependencies.

**Q2: How does the LSTM cell state solve the vanishing gradient problem?**
> The cell state update Cₜ = f ⊙ Cₜ₋₁ + i ⊙ C̃ₜ is primarily additive. The gradient of Cₜ w.r.t. Cₜ₋₁ is the forget gate fₜ — close to 1 when the forget gate is open. This near-linear update path allows gradients to flow backward over many timesteps without shrinking.

**Q3: When should I use GRU vs LSTM?**
> GRU has fewer parameters and trains faster. LSTM has slightly more capacity for very complex long-term dependencies. For most NLP and time series tasks, they perform comparably. Start with GRU; switch to LSTM if GRU underfits. Transformers (Tier 14) have largely replaced both for NLP.

**Q4: What does `return_sequences=True` do in Keras?**
> By default, an LSTM layer returns only the last hidden state (shape: batch × units). With `return_sequences=True`, it returns the hidden state at every timestep (shape: batch × timesteps × units). Use `return_sequences=True` for stacked LSTMs (all but the last layer) and for sequence-labeling tasks.

**Q5: What is the difference between many-to-one, many-to-many, and sequence-to-sequence models?**
> - **Many-to-one**: process whole input sequence, output one label (sentiment classification)
> - **Many-to-many (same length)**: output label at each timestep (named entity recognition)
> - **Seq2Seq**: input and output sequences can have different lengths (machine translation); uses encoder-decoder architecture
