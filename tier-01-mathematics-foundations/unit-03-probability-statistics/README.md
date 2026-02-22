# Unit 3 — Probability & Statistics

## Topics

### 1. Probability Basics
- Sample space Ω, events, probability measure P
- Axioms: P(A) ≥ 0, P(Ω) = 1, P(A∪B) = P(A)+P(B) for disjoint A,B
- Complement: P(Aᶜ) = 1 − P(A)

### 2. Conditional Probability
- P(A|B) = P(A∩B) / P(B)
- Independence: P(A∩B) = P(A)·P(B)
- Law of total probability: P(A) = Σ P(A|Bᵢ)P(Bᵢ)

### 3. Bayes' Theorem
- P(A|B) = P(B|A) · P(A) / P(B)
- Components: posterior, likelihood, prior, evidence
- Application: spam filters, medical diagnosis, Naive Bayes classifier

### 4. Random Variables & Distributions
- Discrete: Bernoulli, Binomial, Categorical, Poisson
- Continuous: Uniform, Gaussian (Normal), Exponential
- PDF, PMF, CDF

### 5. Expectation and Variance
- E[X] = Σ x · P(X=x) (discrete) or ∫ x · f(x) dx (continuous)
- Var[X] = E[(X − μ)²] = E[X²] − (E[X])²
- Standard deviation: σ = √Var[X]

### 6. Key Distributions in ML
- Normal distribution N(μ, σ²): assumed in many models (linear regression residuals)
- Bernoulli / Binomial: binary classification probabilities
- Softmax and the Categorical distribution: multi-class outputs

### 7. Statistical Estimation
- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP) estimation
- Central Limit Theorem (CLT)

---

## Exercises

### Exercise 1 — Conditional Probability
A medical test for a disease has:
- Sensitivity (true positive rate): 95%
- Specificity (true negative rate): 90%
- Disease prevalence: 1%

What is the probability a person actually has the disease given a positive test result?

Use Bayes' theorem: P(Disease|Positive) = P(Positive|Disease) · P(Disease) / P(Positive)

```python
sensitivity = 0.95   # P(+ | disease)
specificity = 0.90   # P(- | no disease)
prevalence  = 0.01   # P(disease)

p_pos_given_disease    = sensitivity
p_pos_given_no_disease = 1 - specificity
p_disease              = prevalence
p_no_disease           = 1 - prevalence

p_positive = p_pos_given_disease * p_disease + p_pos_given_no_disease * p_no_disease
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive
print(f"P(Disease | Positive) = {p_disease_given_pos:.4f} ({p_disease_given_pos*100:.2f}%)")
```

### Exercise 2 — Working with the Normal Distribution
```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

mu, sigma = 170, 10  # height distribution (cm)

# What fraction of people are taller than 185 cm?
p_above_185 = 1 - stats.norm.cdf(185, loc=mu, scale=sigma)
print(f"P(height > 185) = {p_above_185:.4f}")

# Plot the distribution
x = np.linspace(140, 200, 300)
plt.plot(x, stats.norm.pdf(x, mu, sigma))
plt.axvline(185, color='red', linestyle='--', label='185 cm')
plt.title("Normal Distribution of Heights")
plt.xlabel("Height (cm)"); plt.ylabel("Density")
plt.legend(); plt.tight_layout(); plt.show()
```

### Exercise 3 — Maximum Likelihood Estimation
Given n coin flips with k heads, derive and compute the MLE estimate of the bias p.

MLE: maximize L(p) = p^k · (1-p)^(n-k) → p̂ = k/n

```python
import numpy as np

np.random.seed(0)
true_p = 0.7
n_flips = 1000
flips = np.random.binomial(1, true_p, n_flips)

k = flips.sum()
p_hat = k / n_flips
print(f"True p: {true_p}, Flips: {n_flips}, Heads: {k}, MLE p̂: {p_hat:.4f}")
```

### Exercise 4 — Central Limit Theorem
Demonstrate the CLT: show that sample means of an exponential distribution converge to a normal distribution as sample size grows.

```python
import numpy as np
import matplotlib.pyplot as plt

rate = 2.0
sample_sizes = [1, 5, 30, 100]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, n in zip(axes, sample_sizes):
    means = [np.random.exponential(1/rate, n).mean() for _ in range(2000)]
    ax.hist(means, bins=40, edgecolor='black', density=True)
    ax.set_title(f"n = {n}")
plt.suptitle("Central Limit Theorem Demo")
plt.tight_layout(); plt.show()
```

---

## Questions & Answers

**Q1: What is the difference between probability and statistics?**
> Probability goes from a known model to predictions about data ("Given a fair coin, what is P(3 heads in 5 flips)?"). Statistics goes the other direction — from observed data to inferences about an unknown model ("Given these 5 flips, is the coin fair?"). ML does both: it assumes a probabilistic model and uses observed data to estimate its parameters.

**Q2: Why is the Normal distribution so prevalent in ML?**
> Several reasons: (1) The Central Limit Theorem says that sums/averages of independent random variables converge to Normal, so noise in real measurements is often approximately Normal. (2) It is the maximum entropy distribution for a given mean and variance. (3) Gaussian assumptions lead to tractable, closed-form solutions in many algorithms (linear regression residuals, Gaussian Naive Bayes, etc.).

**Q3: What does MLE have to do with minimizing mean squared error?**
> If we assume residuals are normally distributed (y = **w**·x + ε, ε ~ N(0, σ²)), maximizing the likelihood L(**w**) is mathematically equivalent to minimizing the sum of squared residuals. This is why MSE is such a natural loss function for regression.

**Q4: What is the difference between prior, likelihood, and posterior in Bayes' theorem?**
> - **Prior** P(θ): our belief about parameter θ before seeing data
> - **Likelihood** P(data|θ): how probable the observed data is under parameter θ
> - **Posterior** P(θ|data): our updated belief after incorporating the data
> MAP estimation finds the θ that maximizes the posterior, while MLE ignores the prior.

**Q5: Why is Bayes' theorem counterintuitive for rare events?**
> Even a highly accurate test gives a low posterior probability of disease when prevalence is very low (as shown in Exercise 1). This is because the base rate (prevalence) dominates when the event is rare. Most false positives come from the much larger healthy population, even with a 90% specific test. This is called the base rate fallacy.
