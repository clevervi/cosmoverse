# AGENTS.md: Developer & AI Agent Guidelines for COSMOVERSE

> **AUTHORITATIVE DIRECTIVE FOR ALL AI AGENTS:**
> This file and [`docs/MASTER_DOCUMENT.md`](docs/MASTER_DOCUMENT.md) represent the **consolidated, definitive state of the COSMOVERSE project** after two rounds of critical methodological review. Any earlier formulation is superseded by this specification.

---

## 1. Executive Summary & Scientific Identity

**COSMOVERSE** is a large-scale hybrid cosmological inference framework and reproducible protocol connecting:
1. **Fast Neural Emulation**: **TUNeS** (Temporal U-Net, primary v1 architecture; `CosmoRGNet` / FNO reserved as Plan B pending DOI verification).
2. **Explicit Baryonic Marginalization**: **BCEmu** (1, 3, and 7-parameter baryonification in v1 for interpretable marginalization; `BACCOemu` reserved for v2).
3. **Observational Realism (Integrated BEFORE SBI)**: Survey geometry/masks, selection functions, linear/quadratic galaxy bias, Redshift-Space Distortions (RSD multipoles $P_0, P_2, P_4$), shot noise, and weak lensing shear calibration.
4. **Calibrated Bayesian Inference & Field-Level Extension**:
   - **Phase A (Paper 1 MVP — Stages 1–4)**: Simulation-Based Inference (SBI) with Conditional Normalizing Flows validated via **SBC**, **TARP**, fixed-parameter checks, and scale-dependent misspecification detection.
   - **Phase B (Paper 2 Extension — Stage 5)**: Differentiable field-level inference with **JaxPM + preconditioned MCLMC** (benchmarked against HMC/NUTS baselines).

### What COSMOVERSE IS NOT:
- **Not** a brand-new N-body simulator from scratch.
- **Not** a brand-new neural emulator architecture (we use TUNeS / CosmoRGNet / Mangrove).
- **Not** a brand-new MCMC sampler (we use preconditioned MCLMC / HMC).
- **Not** claiming readiness for raw LSST/Euclid/DESI observational catalogs in v1.

### Central Scientific Contribution & Quantitative Hypothesis:
> **Hypothesis**: *"In what scale regime and with what simulation budget does a hybrid framework — emulation + baryonic/observational marginalization + differentiable simulation — produce calibrated posteriors with residual bias below Stage-IV statistical precision, and how much does generalization degrade when transferring across simulation families?"*

---

## 2. Strict Three-Level Methodological Separation

Never mix these three levels in code or experiments:
1. **Level 1 — Physics of the Universe** (`cosmoverse.core`, `cosmoverse.simulation`, `cosmoverse.emulators`, `cosmoverse.baryons`): Background expansion, N-body evolution, TUNeS emulation, and BCEmu baryonic physics.
2. **Level 2 — Observational Process** (`cosmoverse.observations`): Survey masks, selection functions, galaxy bias, RSD, shot noise, and instrumental systematics. **Must be integrated BEFORE Stage 4 (SBI).**
3. **Level 3 — Statistical Inference & Validation** (`cosmoverse.inference`, `cosmoverse.validation`): SBI (Normalizing Flows), preconditioned MCLMC, SBC, TARP, misspecification detection, and cross-simulation transfer.

---

## 3. Four Demonstrable Results & Code Mapping

| Result | Scientific Objective | Module / Implementation | Acceptance Criterion |
| :--- | :--- | :--- | :--- |
| **Result 1** | Emulator accuracy in Gaussian & Non-Gaussian stats | `cosmoverse.emulators.unet.TUNeSEmulator`, `cosmoverse.emulators.statistics` | $P(k)$ MAPE $< 1\%$ at $k < 1\,h\,\mathrm{Mpc}^{-1}$; bounded error in 1-point PDF, 3D Peak Counts, and 3D Minkowski Functionals ($V_0$–$V_3$). |
| **Result 2** | Complete Bias–Variance Frontier | `cosmoverse.baryons.feedback.BCEmuModel`, `cosmoverse.baryons.frontier.BiasVarianceFrontier` | Report full bias vs. variance inflation curve under BCEmu + observational marginalization; identify Stage-IV compatible regime ($|\text{Bias}| < 0.3\,\sigma_{\text{Stage-IV}}$). |
| **Result 3** | SBI Posterior Calibration & Misspecification | `cosmoverse.validation.sbc`, `cosmoverse.validation.tarp`, `cosmoverse.validation.misspecification` | Blocking gate: empirical 68% / 95% coverage via **SBC** and **TARP** within nominal tolerance + fixed-parameter checks + scale-dependent misspecification flag. |
| **Result 4** | Cross-Simulation Family Transfer (Central Contribution) | `cosmoverse.validation.transfer.DatasetPartitionProtocol`, `CrossSimulationTransferEvaluator` | Strictly disjoint `(ic_seed, cosmo_id, family_id)` splits (NO random 80/20 splits!); quantify coverage & bias degradation before/after transfer (e.g. N-body $\to$ Hydro). |

---

## 4. Mandatory Data Partitioning Protocol (Section 7)

- **FORBIDDEN**: Naive random 80/20 train/test splits (`DatasetPartitionProtocol` raises `ValueError` if `split_mode="random_80_20"` is requested).
- **REQUIRED**:
  1. Partition strictly by **initial conditions (`ic_seed`)** and **cosmology (`cosmo_id`)**.
  2. Partition by **simulation family (`family_id`)** for transfer benchmarks (e.g. `CAMELS-DMO` $\to$ `CAMELS-IllustrisTNG`).
  3. Always verify zero leakage via `DatasetPartitionProtocol.verify_zero_leakage(train_set, test_set)`.

---

## 5. Physical Conventions, Units & Constants

| Quantity | Variable Name | Units | Notes |
| :--- | :--- | :--- | :--- |
| Speed of light $c$ | `C_KM_S` | $\text{km/s}$ | Exactly $299792.458$ |
| Hubble constant $H_0$ | `H0` | $\text{km/s/Mpc}$ | $H_0 = 100 \cdot h$ |
| Comoving Box Size $L$ | `box_size` | $\text{Mpc}/h$ | CAMELS Gen-2 default $50.0$, fiducial $100.0$ |
| Wavenumber $k$ | `k` | $h/\text{Mpc}$ | Fundamental $k_0 = 2\pi/L$, Nyquist $k_{\text{Ny}} = \pi N/L$ |
| Power Spectrum $P(k)$ | `pk` | $(\text{Mpc}/h)^3$ | Real-space $P(k)$ and RSD multipoles $P_0(k), P_2(k), P_4(k)$ |

---

## 6. Testing & Verification Commands

Always run both the unit test suite, the local CI script, and the Paper 1 MVP protocol after any code change:
```bash
uv run ruff check .
uv run pytest -v
uv run python experiments/run_paper1_mvp_protocol.py
./scripts/ci.ps1   # Windows
./scripts/ci.sh    # Linux/macOS/Git Bash
```

---

## 7. GitHub Workflow & Governance Rules (`clevervi/cosmoverse`)

- **Branches**: `main` is stable; `develop` is integration. Never commit directly to `main` or `develop`; branch from `develop` (`feat/...`, `fix/...`, `chore/...`, `docs/...`).
- **Commits**: Conventional Commits (`feat(emulators): ...`, `fix(validation): ...`), subject $\le 72$ chars.
- **Git Hooks**: Activate with `git config core.hooksPath .githooks`. Never bypass with `--no-verify`.
- **Local CI Gate**: `scripts/ci.ps1` (Windows) and `scripts/ci.sh` (Unix) are the mandatory pre-push gate. Hosted GitHub Actions (`ci.yml`) is set to `workflow_dispatch` and release tags (`v*`) to preserve quota.
- **UTF-8 GitHub Text Rule**: When publishing or editing GitHub issues, PRs, milestones, or comments via `gh`, **always** write the text to a UTF-8 file (no BOM) and pass `--body-file` or `gh api -F body=@file`—never pass multi-line or accented strings inline in PowerShell. Verify after publishing.
- **PRs**: Must reference `Closes #N`, include labels (`type:*`, `area:*`, `priority:*`), milestone (`M1`–`M6`), and real verification output.
