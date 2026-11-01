# GitHub Copilot Instructions for COSMOVERSE

- Maintain strict PEP 8, full type hints, and Google-style docstrings with mathematical equations where applicable.
- In `cosmoverse.core`: calculate distances, linear growth $D_1(z)$, and Eisenstein & Hu (1998) transfer functions accurately.
- In `cosmoverse.simulation`: use Cloud-in-Cell (CIC) mass assignment and Kick-Drift-Kick (KDK) leapfrog time stepping.
- In `cosmoverse.inference`: maintain Simulation-Based Inference (SBI) conventions, rank tests (SBC), and normalizing flows.
- When generating random numbers, always use `np.random.default_rng(seed)`.
