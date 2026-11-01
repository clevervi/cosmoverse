# Theoretical & Computational Physics Reference

This document details the mathematical formulations and numerical schemes implemented in `cosmoverse`.

---

## 1. Cosmological Background Expansion

In a flat universe governed by cold dark matter and dynamical dark energy with the Chevallier-Polarski-Linder (CPL) parametrization $w(a) = w_0 + w_a(1-a)$, the Friedmann equation is:

$$E^2(z) \equiv \left(\frac{H(z)}{H_0}\right)^2 = \Omega_m (1+z)^3 + \Omega_\Lambda (1+z)^{3(1+w_0+w_a)} \exp\left(-3 w_a \frac{z}{1+z}\right)$$

### Distances
- **Comoving Distance**:
  $$\chi(z) = \frac{c}{H_0} \int_0^z \frac{dz'}{E(z')}$$
- **Angular Diameter Distance**:
  $$D_A(z) = \frac{\chi(z)}{1+z}$$
- **Luminosity Distance**:
  $$D_L(z) = (1+z) \chi(z)$$

---

## 2. Linear Perturbation Growth

The linear growth factor $D_1(a)$ satisfies the perturbation differential equation:

$$\frac{d^2 D_1}{da^2} + \left(\frac{3}{a} + \frac{d\ln H}{da}\right) \frac{d D_1}{da} - \frac{3 \Omega_m(a)}{2 a^2} D_1 = 0$$

Equivalently, using the Heath (1977) integral:

$$D_1(a) \propto \frac{H(a)}{H_0} \int_0^a \frac{da'}{(a' H(a') / H_0)^3}$$

Normalized such that $D_1(z=0) = 1$. The linear growth rate is:

$$f(z) \equiv \frac{d\ln D_1}{d\ln a} \approx \Omega_m(z)^\gamma, \quad \gamma \approx 0.55 + 0.05(1+w_0)$$

---

## 3. Linear Matter Power Spectrum: Eisenstein & Hu (1998)

The primordial power spectrum generated during cosmic inflation is $P_{\text{prim}}(k) \propto k^{n_s}$. The linear matter power spectrum at redshift $z$ is:

$$P_{\text{lin}}(k, z) = A \cdot k^{n_s} \cdot T^2(k) \cdot D_1^2(z)$$

where $T(k)$ is the transfer function accounting for sub-horizon perturbation growth and baryon acoustic oscillations (BAO), normalized such that the variance in spheres of radius $R = 8\, h^{-1}\text{Mpc}$ equals $\sigma_8^2$:

$$\sigma^2(R) = \frac{1}{2\pi^2} \int_0^\infty k^2 P_{\text{lin}}(k, 0) W^2(k R)\, dk, \quad W(x) = \frac{3(\sin x - x\cos x)}{x^3}$$

---

## 4. Particle-Mesh (PM) Dynamics

### Zel'dovich (1LPT) Initial Perturbations
Particles initially located on a regular Lagrangian grid $\vec{q}$ are displaced according to:

$$\vec{x}(\vec{q}, z) = \vec{q} + D_1(z) \vec{\Psi}_1(\vec{q})$$

where the displacement potential satisfies $\nabla^2 \Phi_1 = \delta$, and $\vec{\Psi}_1 = -\nabla \Phi_1$. In Fourier space:

$$\vec{\Psi}_1(\vec{k}) = -i \frac{\vec{k}}{k^2} \delta(\vec{k})$$

### Cloud-in-Cell (CIC) Mass Assignment
Continuous density is assigned onto a discrete mesh with cell size $\Delta x = L / N_{\text{mesh}}$ by weighting the volume overlap of a cube of width $\Delta x$ centered at each particle with the adjacent 8 grid cells.

### Gravitational Poisson Equation
$$\nabla^2 \phi(\vec{x}) = \frac{3}{2} \Omega_m \left(\frac{H_0^2}{a}\right) \delta(\vec{x})$$

Solved in Fourier space:

$$\phi(\vec{k}) = -\frac{3}{2} \Omega_m \left(\frac{H_0^2}{a}\right) \frac{\delta(\vec{k})}{k^2}, \quad \vec{g}(\vec{k}) = -i \vec{k} \phi(\vec{k})$$

### Kick-Drift-Kick (KDK) Leapfrog Scheme
$$\vec{v}^{n+1/2} = \vec{v}^n + \frac{\Delta t}{2a} \vec{g}(\vec{x}^n)$$
$$\vec{x}^{n+1} = \vec{x}^n + \frac{\Delta t}{a} \vec{v}^{n+1/2}$$
$$\vec{v}^{n+1} = \vec{v}^{n+1/2} + \frac{\Delta t}{2a} \vec{g}(\vec{x}^{n+1})$$

---

## 5. Baryonic Feedback Suppression

Supermassive black hole AGN accretion and supernova stellar feedback redistribute gas from cluster cores out into circumgalactic scales, creating a suppression dip in the matter power spectrum $S_b(k) = P_{\text{hydro}}(k) / P_{\text{DMO}}(k)$ at $k \sim 1 - 10\, h/\text{Mpc}$:

$$S_b(k) \approx 1 - \Delta_b(A_{\text{AGN}}) \exp\left(-\frac{1}{2} \left(\frac{\log_{10} k - \log_{10} k_{\text{dip}}}{\sigma}\right)^2\right) + \text{star\_boost}(A_{\text{SN}})$$
