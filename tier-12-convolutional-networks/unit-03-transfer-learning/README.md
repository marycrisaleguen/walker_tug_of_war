# Unit 3 — Transfer Learning

## Topics

### 1. What is Transfer Learning?
- Use a model trained on a large dataset (ImageNet) as a starting point for a new task
- Leverages learned feature representations: edges, textures, shapes, object parts
- Dramatically reduces training data and compute requirements

### 2. Why Transfer Learning Works
- ImageNet features generalize: low-level features (edges, corners) transfer to any visual task
- High-level features are more task-specific but still useful as initialization
- The deeper the layer, the more task-specific the features become

### 3. Two Strategies

#### Feature Extraction
- Freeze all pretrained layers (no gradient updates)
- Add and train only a new classifier head on top
- Use when: your dataset is small; your domain is similar to ImageNet

#### Fine-Tuning
- Start with pretrained weights, unfreeze some (or all) layers
- Train with a **small learning rate** to avoid destroying pretrained features
- Use when: you have more data; your domain differs somewhat from ImageNet

### 4. Common Pretrained Models
| Model | Top-1 Accuracy (ImageNet) | Notes |
|-------|--------------------------|-------|
| VGG16 | 71.3% | Simple architecture, large size |
| ResNet50 | 75.9% | Residual connections, versatile |
| EfficientNetB0 | 77.1% | Best accuracy/param tradeoff |
| MobileNetV2 | 71.8% | Lightweight, mobile deployment |

### 5. Practical Tips
- Use the same preprocessing as the pretrained model (e.g., `tf.keras.applications.resnet50.preprocess_input`)
- Unfreeze layers from the top down (task-specific layers first)
- Use a learning rate 10–100× smaller than the original training lr for fine-tuning
- Monitor for overfitting — pretrained features + small dataset = high overfitting risk

---

## Exercises

### Exercise 1 — Feature Extraction with ResNet50
```python
import tensorflow as tf
import numpy as np

# Load CIFAR-10 and resize to match ResNet50 input
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Resize from 32x32 to 96x96 (ResNet minimum is 32, but larger is better)
X_train_r = tf.image.resize(X_train, [96, 96]).numpy().astype("float32")
X_test_r  = tf.image.resize(X_test,  [96, 96]).numpy().astype("float32")
X_train_p = tf.keras.applications.resnet50.preprocess_input(X_train_r)
X_test_p  = tf.keras.applications.resnet50.preprocess_input(X_test_r)

# Build model: frozen ResNet50 base + new head
base = tf.keras.applications.ResNet50(
    weights='imagenet', include_top=False,
    input_shape=(96, 96, 3), pooling='avg'
)
base.trainable = False  # freeze

inputs = tf.keras.Input(shape=(96, 96, 3))
x = base(inputs, training=False)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train only the new head
history = model.fit(
    X_train_p[:5000], y_train[:5000],  # small subset for demo
    epochs=10, batch_size=64, validation_split=0.2, verbose=1
)
print(f"Test accuracy: {model.evaluate(X_test_p, y_test, verbose=0)[1]:.4f}")
```

### Exercise 2 — Fine-Tuning (Unfreeze Top Layers)
```python
# Unfreeze the top 20 layers for fine-tuning (continuing from Exercise 1)
base.trainable = True
# Freeze all layers except the last 20
for layer in base.layers[:-20]:
    layer.trainable = False

# Recompile with a much smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # 100x smaller lr
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Fine-tune
model.fit(
    X_train_p[:5000], y_train[:5000],
    epochs=5, batch_size=32, validation_split=0.2, verbose=1
)
print(f"After fine-tuning: {model.evaluate(X_test_p, y_test, verbose=0)[1]:.4f}")
```

---

## Questions & Answers

**Q1: Why does feature extraction work even when the new task is very different from ImageNet?**
> Low-level features (edges, textures, color gradients) are domain-agnostic — useful for any image task. Even for medical images or satellite imagery, the feature extraction backbone learns useful low-level representations. The domain mismatch matters more for high-level features (specific objects, scenes).

**Q2: How do I decide how many layers to freeze?**
> General guideline: the more data you have and the more different your domain is from ImageNet, the more layers you should unfreeze (fine-tune). With very limited data, freeze everything and only train the new head. With more data and a different domain, fine-tune the top few layers. With lots of data, fine-tune the entire network.

**Q3: Why use a smaller learning rate for fine-tuning?**
> The pretrained weights already encode useful features. Large gradient updates would overwrite them, losing the transferred knowledge. A small lr (typically 1e-5 to 1e-4) makes gentle adjustments that adapt the features to the new task while preserving the broad structure learned on ImageNet.

**Q4: What is catastrophic forgetting?**
> When a network trained on task A is then trained on task B, performance on task A degrades dramatically — the new training "forgets" the old. Fine-tuning with a small lr, gradual layer unfreezing, and keeping the new training set large help mitigate catastrophic forgetting.

**Q5: Should I always normalize images using the pretrained model's expected statistics?**
> Yes. Pretrained models were trained with specific input normalization (e.g., ResNet expects ImageNet mean/std subtraction). Using a different normalization changes the activation distribution throughout the network, reducing the quality of the pretrained features. Always apply the preprocessing function provided by the model (e.g., `preprocess_input`).
