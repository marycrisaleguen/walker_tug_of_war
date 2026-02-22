# Unit 1 — TensorFlow & Keras

## Topics

### 1. TensorFlow Basics
- Tensors: n-dimensional arrays, analogous to NumPy arrays but GPU-accelerated
- `tf.constant()`, `tf.Variable()`, `tf.zeros()`, `tf.ones()`
- Eager execution (default in TF 2.x): operations execute immediately
- `@tf.function`: compile to static graph for speed

### 2. Keras Sequential API
- Stack layers linearly: `model = tf.keras.Sequential([...])`
- Built-in layers: `Dense`, `Dropout`, `BatchNormalization`, `Flatten`
- `model.summary()` to inspect architecture
- Fastest way to build standard models

### 3. Keras Functional API
- Supports complex architectures: multi-input, multi-output, skip connections
- Define inputs explicitly: `inputs = tf.keras.Input(shape=(...))`
- Call layers as functions: `x = Dense(64)(inputs)`
- Build model: `model = tf.keras.Model(inputs=inputs, outputs=outputs)`

### 4. Training with `model.compile()` and `model.fit()`
- `compile(optimizer, loss, metrics)`
- Optimizers: `Adam`, `SGD`, `RMSprop`
- Losses: `binary_crossentropy`, `categorical_crossentropy`, `mse`
- `fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)`

### 5. Callbacks
- `ModelCheckpoint`: save best model during training
- `EarlyStopping`: stop when validation loss stops improving
- `ReduceLROnPlateau`: reduce learning rate when improvement stalls
- `TensorBoard`: real-time training visualization

### 6. Saving and Loading Models
- `model.save("model.h5")` / `tf.keras.models.load_model("model.h5")`
- SavedModel format: `model.save("my_model/")` (preferred for deployment)

---

## Exercises

### Exercise 1 — Binary Classification with Keras Sequential
```python
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_tr_s, y_tr,
    epochs=50, batch_size=32,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=0
)
test_loss, test_acc = model.evaluate(X_te_s, y_te, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
```

### Exercise 2 — Plot Training History
```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['loss'],     label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss', linestyle='--')
ax1.set_title("Loss"); ax1.legend()
ax2.plot(history.history['accuracy'],     label='Train Acc')
ax2.plot(history.history['val_accuracy'], label='Val Acc', linestyle='--')
ax2.set_title("Accuracy"); ax2.legend()
plt.tight_layout(); plt.show()
```

### Exercise 3 — Functional API: Multi-Input Model (Sketch)
```python
import tensorflow as tf

# Concatenate two feature groups as separate inputs
input_a = tf.keras.Input(shape=(15,), name="group_a")
input_b = tf.keras.Input(shape=(15,), name="group_b")

x_a = tf.keras.layers.Dense(16, activation='relu')(input_a)
x_b = tf.keras.layers.Dense(16, activation='relu')(input_b)
merged = tf.keras.layers.concatenate([x_a, x_b])
output = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
```

---

## Questions & Answers

**Q1: What is the difference between the Sequential API and the Functional API?**
> Sequential is for simple, linear stacks of layers. Functional API supports arbitrary graphs: shared layers, multiple inputs/outputs, skip connections (ResNet), and branches. If your architecture is a straight pipeline, Sequential is cleaner. For anything more complex, use the Functional API.

**Q2: What does `Dropout` do during training vs inference?**
> During training, Dropout randomly zeroes a fraction of neurons' outputs (rate=0.3 → 30% dropped). This acts as an ensemble of many sub-networks, reducing overfitting. During inference (`model.evaluate()`, `model.predict()`), Dropout is automatically disabled and all neurons are active.

**Q3: What is the purpose of Batch Normalization?**
> BatchNorm normalizes the activations of each layer to have zero mean and unit variance across the batch, then applies learnable scale and shift parameters. This (1) stabilizes training, (2) allows higher learning rates, (3) reduces sensitivity to initialization, and (4) acts as mild regularization.

**Q4: When should I use `EarlyStopping`?**
> When training for a fixed number of epochs risks over-training. Set `patience=10–20` to allow temporary validation loss increases (noise) before actually stopping. Use `restore_best_weights=True` to automatically reload the best checkpoint. Early stopping is the most practical regularization technique for neural networks.

**Q5: What is the difference between `model.save()` HDF5 format and SavedModel format?**
> HDF5 (`.h5`) stores weights and architecture in a single file — convenient for simple models. SavedModel stores the full TensorFlow computation graph, making it more portable for deployment (TensorFlow Serving, TensorFlow Lite, TensorFlow.js). Prefer SavedModel for production deployments.
