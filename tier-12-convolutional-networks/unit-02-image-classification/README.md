# Unit 2 — Image Classification

## Topics

### 1. Image Datasets and Formats
- Images as (H × W × C) tensors: height, width, channels (RGB=3, grayscale=1)
- Common formats: JPEG, PNG; pixel values typically normalized to [0,1] or [-1,1]
- Standard benchmarks: MNIST, CIFAR-10/100, ImageNet, COCO

### 2. Data Augmentation
- Artificially expand the training set by applying random transformations
- Common augmentations: horizontal flip, random crop, rotation, color jitter, cutout
- Prevents overfitting, improves generalization, especially for small datasets
- `tf.keras.layers.RandomFlip`, `RandomRotation`, `RandomZoom` (in Keras)
- `torchvision.transforms` (in PyTorch)

### 3. Batch Normalization
- Normalizes activations within each mini-batch: zero mean, unit variance per channel
- Then applies learnable scale (γ) and shift (β) parameters
- Placed after Conv layers, before activation (or after — both are used)
- Benefits: stabilizes training, allows higher lr, reduces need for careful initialization

### 4. Dropout in CNNs
- Standard dropout works poorly in CNNs (spatially correlated activations)
- **SpatialDropout2D**: drops entire feature maps randomly
- Use in fully connected layers as usual

### 5. Training Pipeline
- Load images → augment → normalize → batch → model → loss → optimize
- `tf.data.Dataset` or `torch.utils.data.DataLoader` for efficient pipelines

### 6. Architecture Best Practices
- Start small, add complexity if underfitting
- Increase channels as spatial dimensions decrease
- Use Batch Normalization throughout
- Global Average Pooling before classifier (instead of Flatten)

---

## Exercises

### Exercise 1 — CIFAR-10 Classification with Data Augmentation
```python
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

# Data augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

def build_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = augmentation(inputs)
    for filters in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=128,
                    validation_split=0.1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
                    verbose=1)
print(f"Test accuracy: {model.evaluate(X_test, y_test, verbose=0)[1]:.4f}")
```

---

## Questions & Answers

**Q1: Why is data augmentation so effective?**
> From the network's perspective, augmented images look like entirely new training examples. Horizontal flips, crops, and color jitter create variations the model must be invariant to, teaching the model that a cat flipped horizontally is still a cat. This is a form of implicit regularization that often provides larger accuracy gains than architectural improvements on small datasets.

**Q2: What is the difference between Batch Normalization before vs after activation?**
> Original ResNet (He et al., 2016) applies BN after Conv and before activation. Pre-activation ResNet (He et al., 2016b) applies BN then activation then Conv. Pre-activation is theoretically cleaner (normalization directly precedes the linear operation) and often performs slightly better in very deep residual networks. In practice, both work well.

**Q3: Why use Global Average Pooling instead of Flatten before the Dense layer?**
> Flatten from a (7,7,512) feature map creates 7×7×512 = 25,088 inputs to the Dense layer — many parameters, high overfitting risk. Global Average Pooling reduces each feature map to a single number (its mean), giving 512 inputs. This dramatically reduces parameters and acts as regularization. It also makes the model input-size agnostic.

**Q4: What is test-time augmentation (TTA)?**
> TTA applies multiple augmented versions of a test image (e.g., original + flipped + cropped), runs inference on each, and averages the predictions. It can boost accuracy by 1–2% at the cost of multiple forward passes. Common in competitions and when high accuracy is critical.

**Q5: How do I handle class imbalance in image classification?**
> Options: (1) Oversample minority classes (duplicate images or apply more augmentation to them). (2) Undersample majority class. (3) Class-weighted loss: `class_weight` parameter in Keras `model.fit()`. (4) Use focal loss (emphasizes hard/rare examples). For very severe imbalance, evaluate with F1 or AUC, not accuracy.
