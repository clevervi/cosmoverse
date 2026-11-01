# Bayesian Inference & Statistical Validation

This document outlines the likelihood-free and gradient-based Bayesian inference engines implemented in `cosmoverse`.

---

## 1. Simulation-Based Inference (SBI)

Traditional cosmological inference relies on Gaussian likelihood approximations $\ln \mathcal{L}(x \mid \theta) = -\frac{1}{2} (x - \mu)^T \mathbf{C}^{-1} (x - \mu)$. However, non-Gaussian cosmic structures and non-linear scales invalidate Gaussian likelihoods.

Simulation-Based Inference (also known as Likelihood-Free Inference) models the intractable posterior directly:

$$p(\theta \mid x) = \frac{p(x \mid \theta) p(\theta)}{p(x)}$$

### Neural Posterior Estimation (NPE)
In NPE, a conditional density estimator $q_\phi(\theta \mid x)$ (such as a conditional Normalizing Flow) is trained to minimize the negative log-likelihood over a simulation bank $\{(\theta_i, x_i)\}_{i=1}^N$:

$$\mathcal{L}(\phi) = -\frac{1}{N} \sum_{i=1}^N \ln q_\phi(\theta_i \mid x_i)$$

As $N \to \infty$, $q_\phi(\theta \mid x) \to p(\theta \mid x)$.

---

## 2. Conditional Normalizing Flows (RealNVP)

In `cosmoverse.inference.flows`, we implement Affine Coupling Layers where the base Gaussian distribution $z \sim \mathcal{N}(0, \mathbf{I})$ is transformed to parameter space $\theta$:

$$\begin{aligned}
y_{1:d} &= z_{1:d} \\
y_{d+1:D} &= z_{d+1:D} \odot \exp(s(z_{1:d}, x)) + t(z_{1:d}, x)
\end{aligned}$$

The Jacobian determinant is triangular:

$$\det\left(\frac{\partial y}{\partial z}\right) = \prod_{j=d+1}^D \exp(s_j(z_{1:d}, x)) \implies \ln \left|\det J\right| = \sum_{j=d+1}^D s_j(z_{1:d}, x)$$

---

## 3. Simulation-Based Calibration (SBC)

To prove that the posterior estimator is statistically unbiased and properly calibrated, we execute the SBC diagnostic (Talts et al. 2018):

1. Draw true parameter from prior: $\theta^* \sim p(\theta)$
2. Run forward simulator: $x^* \sim p(x \mid \theta^*)$
3. Draw $L$ posterior samples: $\{\theta^{(l)}\}_{l=1}^L \sim q_\phi(\theta \mid x^*)$
4. Compute the rank statistic of $\theta^*$ relative to posterior samples:
   $$r_j = \sum_{l=1}^L \mathbb{I}\left(\theta_j^{(l)} < \theta_j^*\right)$$

**Calibration Theorem**: If and only if $q_\phi(\theta \mid x) = p(\theta \mid x)$, the rank statistic is uniformly distributed:
$$r_j \sim \text{DiscreteUniform}(0, L)$$

We evaluate this using the Kolmogorov-Smirnov test against $\text{Uniform}(0, 1)$.

---

## 4. Figure of Merit (FoM) & Tension Metrics

The Dark Energy Task Force (DETF) Figure of Merit for parameter constraints is:

$$\text{FoM} = \frac{1}{\sqrt{\det \mathbf{C}(\Omega_m, \sigma_8)}}$$

Between two independent experiments (e.g., Early Universe CMB vs. Late Universe Weak Lensing), the Gaussian tension metric is:

$$T = \frac{|\mu_1 - \mu_2|}{\sqrt{\sigma_1^2 + \sigma_2^2}}$$
