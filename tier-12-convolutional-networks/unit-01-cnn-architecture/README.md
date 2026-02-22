# Unit 1 — CNN Architecture

## Topics

### 1. Why CNNs for Images?
- Fully connected layers for images are parameter-inefficient: a 224×224×3 image → 150,528 inputs
- CNNs exploit spatial locality and translational invariance via weight sharing
- A filter that detects an edge anywhere in the image uses the same weights → far fewer parameters

### 2. Convolutional Layer
- A filter (kernel) of shape (k×k×C_in) slides over the input
- Computes dot product between filter and each local patch
- One filter → one feature map; multiple filters → multiple feature maps
- Parameters: `out_channels × in_channels × k × k` + bias
- Hyperparameters: `kernel_size`, `stride`, `padding`, `out_channels`

### 3. Pooling Layers
- **Max Pooling**: take the maximum value in each pool region → selects dominant features
- **Average Pooling**: take the average → smoother downsampling
- Reduces spatial dimensions, adds translation invariance, reduces parameters
- `stride=2` halves the spatial dimensions

### 4. Padding
- **Valid**: no padding — output size = (n − k + 1) / s
- **Same**: pad so output size = input size (with stride=1)
- Formula: output_size = (input + 2p − k) / s + 1

### 5. Receptive Field
- The portion of the input that influences a given neuron
- Grows with depth: early layers detect local features; deep layers detect global patterns
- Stacking 2 layers of 3×3 convolutions = 5×5 receptive field with fewer parameters than one 5×5

### 6. Classic CNN Architectures
- **LeNet-5** (1998): Conv → Pool → Conv → Pool → FC → FC
- **AlexNet** (2012): first deep CNN on ImageNet; introduced ReLU, Dropout
- **VGG**: very deep with 3×3 convolutions
- **ResNet**: residual connections (skip connections) solve vanishing gradient in very deep networks

---

## Exercises

### Exercise 1 — Build a CNN for MNIST
```python
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=5, batch_size=64,
                    validation_split=0.1, verbose=1)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")
```

### Exercise 2 — Visualize Filters and Feature Maps
```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), _ = tf.keras.datasets.mnist.load_data()
X_sample = X_train[0:1].reshape(1, 28, 28, 1).astype("float32") / 255.0

# Extract first conv layer activations
activation_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[model.layers[0].output]  # first Conv2D
)
feature_maps = activation_model.predict(X_sample, verbose=0)[0]

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.ravel()):
    if i < feature_maps.shape[-1]:
        ax.imshow(feature_maps[:, :, i], cmap='viridis')
    ax.axis('off')
plt.suptitle("First Conv Layer Feature Maps"); plt.tight_layout(); plt.show()
```

---

## Questions & Answers

**Q1: Why do CNNs use weight sharing (shared filters)?**
> The same feature (e.g., a horizontal edge) can appear anywhere in an image. Rather than learning a separate detector for each position, a single filter learns to detect that feature and is applied at every position. This drastically reduces parameters: a 3×3 filter on 256-channel input has only 3×3×256+1 = 2305 parameters instead of millions for a fully connected approach.

**Q2: What is the difference between valid and same padding?**
> "Valid" padding applies no padding — the convolution shrinks the spatial dimensions by (k−1) pixels on each side. "Same" padding pads the input (usually with zeros) so the output has the same spatial dimensions as the input (for stride=1). Same padding is more convenient for designing architectures since dimensions are predictable.

**Q3: Why are skip (residual) connections important in deep networks?**
> In very deep networks (50+ layers), gradients can vanish during backpropagation. ResNet's skip connections add the input directly to the output of a block: h = F(x) + x. The gradient of h w.r.t. x is at least 1 (from the identity shortcut), preventing vanishing. This allows training of 100+ layer networks.

**Q4: What is the receptive field and why does it matter?**
> The receptive field is the region of the input that a neuron "sees." Early layers have small receptive fields (local features: edges, corners). Deeper layers have larger receptive fields (global features: objects, scenes). For tasks requiring global context (whole-image classification), you need the final layers to have receptive fields covering most of the input.

**Q5: Should I always use max pooling?**
> Not necessarily. Strided convolutions (stride=2) have replaced max pooling in many modern architectures — they let the network learn how to downsample rather than using a hand-designed pooling. Global Average Pooling is preferred before the final classifier to reduce spatial maps to a single vector per channel.
