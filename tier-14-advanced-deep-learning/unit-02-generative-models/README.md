# Unit 2 — Generative Models

## Topics

### 1. Types of Generative Models
- **VAE** (Variational Autoencoder): learns a continuous latent space; smooth interpolation
- **GAN** (Generative Adversarial Network): adversarial training between generator and discriminator
- **Diffusion Models**: iteratively denoise random noise into a sample; state-of-the-art for images
- **Normalizing Flows**: exact likelihood; invertible transformations

### 2. Autoencoders
- Encoder: x → z (bottleneck latent representation)
- Decoder: z → x̂ (reconstruction)
- Loss: reconstruction error ‖x − x̂‖²
- Limitation: discrete, non-smooth latent space; cannot sample new data easily

### 3. Variational Autoencoders (VAE)
- Encoder outputs μ and σ² instead of a point (models the posterior q(z|x) ≈ N(μ, σ²))
- Reparameterization trick: z = μ + σ·ε, ε ~ N(0,I) — makes sampling differentiable
- Loss: Reconstruction loss + KL divergence KL(q(z|x) || p(z)) — keeps latent space regular
- Continuous latent space: can generate new samples and interpolate between them

### 4. Generative Adversarial Networks (GANs)
- **Generator** G: z → x̂ (generate fake images from noise)
- **Discriminator** D: x → P(real) (distinguish real from fake)
- Min-max game: min_G max_D [log D(x) + log(1 − D(G(z)))]
- Training: alternate between updating D and G

### 5. GAN Training Challenges
- **Mode collapse**: generator produces only a few diverse outputs
- **Training instability**: G and D can fail to converge
- Solutions: Wasserstein GAN (WGAN), spectral normalization, progressive growing, label smoothing

### 6. Diffusion Models
- Forward process: gradually add Gaussian noise over T steps until x_T ~ N(0,I)
- Reverse process: learn to denoise — train a U-Net to predict noise at each step
- At inference: start from noise, run reverse process T times
- Examples: DALL-E 2, Stable Diffusion, Imagen

---

## Exercises

### Exercise 1 — Variational Autoencoder on MNIST
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
X_test  = X_test.reshape(-1, 784).astype("float32") / 255.0

latent_dim = 2

# Encoder
inputs = tf.keras.Input(shape=(784,))
h = tf.keras.layers.Dense(256, activation='relu')(inputs)
z_mean   = tf.keras.layers.Dense(latent_dim)(h)
z_log_var = tf.keras.layers.Dense(latent_dim)(h)

# Reparameterization
def sampling(args):
    mean, log_var = args
    eps = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * eps

z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_h  = tf.keras.layers.Dense(256, activation='relu')
decoder_out = tf.keras.layers.Dense(784, activation='sigmoid')
x_recon = decoder_out(decoder_h(z))

vae = tf.keras.Model(inputs, x_recon)

# Custom loss = reconstruction + KL divergence
recon_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(inputs, x_recon)) * 784
kl_loss    = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
vae.add_loss(tf.reduce_mean(recon_loss + kl_loss))
vae.compile(optimizer='adam')
vae.fit(X_train, X_train, epochs=20, batch_size=128, validation_split=0.1, verbose=0)
print("VAE training complete")
```

### Exercise 2 — GAN Concept (Architecture Sketch)
```python
import tensorflow as tf

# Generator: noise → fake image
def make_generator(latent_dim=100):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(latent_dim,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(784, activation='tanh'),
        tf.keras.layers.Reshape((28, 28, 1))
    ])

# Discriminator: image → real/fake probability
def make_discriminator():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(512, activation='leaky_relu'),
        tf.keras.layers.Dense(256, activation='leaky_relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

G = make_generator(); D = make_discriminator()
G.summary(); D.summary()
```

---

## Questions & Answers

**Q1: What is the difference between a VAE and an autoencoder?**
> A standard autoencoder maps each input to a single point in latent space — the space is not regularized and may have gaps where sampling produces garbage. A VAE maps each input to a distribution (μ, σ²) and samples from it. The KL divergence term regularizes the latent space to be continuous and close to N(0,I), enabling smooth sampling and interpolation.

**Q2: What is the reparameterization trick in VAEs?**
> To backpropagate through a sampling operation z ~ N(μ, σ²), we reparameterize: z = μ + σ·ε where ε ~ N(0,I). Now z is a deterministic function of μ, σ, and ε — gradients can flow through μ and σ. Without this trick, sampling is non-differentiable and the encoder cannot be trained with gradient descent.

**Q3: Why is GAN training unstable?**
> GANs solve a minimax game between G and D. If D is too strong, the generator gets vanishing gradients (D(G(z)) ≈ 0 → log(1−D(G(z))) saturates). If G is too strong, D can't discriminate. Perfect balance is hard to maintain. Mode collapse occurs when G finds a few samples that fool D and never explores other modes. Wasserstein GANs replace the divergence metric to provide more stable gradients.

**Q4: Why have diffusion models overtaken GANs for image generation?**
> Diffusion models have more stable training (no adversarial game), better sample diversity, and a clear likelihood objective. They produce higher quality and more diverse images than GANs. DALL-E 2, Stable Diffusion, and Imagen are all diffusion-based. The trade-off is slower inference (requires many denoising steps), though distillation techniques are reducing this gap.

**Q5: What is latent space interpolation?**
> In a VAE, the latent space is continuous. Interpolating between two latent vectors z₁ and z₂ (e.g., z = α·z₁ + (1−α)·z₂ for α ∈ [0,1]) and decoding produces intermediate images that smoothly transition between the two inputs. This is a qualitative test of whether the latent space is well-structured.
