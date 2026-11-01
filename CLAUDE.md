# CLAUDE.md - AI Assistant Reference Guide for COSMOVERSE

> **Master Document Reference:** Always consult [`docs/MASTER_DOCUMENT.md`](docs/MASTER_DOCUMENT.md) and [`AGENTS.md`](AGENTS.md) for the consolidated project specification.

## Overview
**COSMOVERSE** is a hybrid cosmological inference protocol connecting neural emulation (**TUNeS**), explicit baryonic (**BCEmu**) and observational marginalization, and Simulation-Based Inference (**SBI** with Normalizing Flows), with an extension for differentiable field-level inference (**JaxPM + preconditioned MCLMC**).
- **Author:** `clevervi` (Adrian Dario V. - `adriandariov@gmail.com`)
- **Organization / Repo:** `cosmoverse-project / cosmoverse`
- **Paper 1 MVP:** Stages 1–4 (Emulator + BCEmu + Stage 3 Observational Realism + Calibrated SBI + Cross-Simulation Transfer).
- **Paper 2 Extension:** Stage 5 (Field-level inference with JaxPM + preconditioned MCLMC).

## Key Commands
- **Run all 86 unit tests:** `uv run pytest -v`
- **Run the complete Paper 1 MVP Protocol (Results 1–4):** `uv run python experiments/run_paper1_mvp_protocol.py`
- **Run CLI info:** `uv run cosmoverse info`
- **Run Result 2 Bias-Variance Frontier table:** `uv run cosmoverse frontier`
- **Run Particle-Mesh simulation:** `uv run cosmoverse simulate --mesh 32 --steps 5`
- **Run CAMELS benchmark LHS:** `uv run cosmoverse benchmark --n 10`

## Critical Methodological Rules
1. **Three-Level Separation:** Never mix (1) Universe Physics, (2) Observational Process, and (3) Statistical Inference.
2. **Single v1 Architecture per Layer:** Use `TUNeSEmulator` for v1 emulation (`FNO3dEmulator` / `CosmoRGNet` is Plan B pending verification) and `BCEmuModel` for v1 baryonic physics (`BACCOemu` is v2).
3. **Observational Realism BEFORE SBI:** Stage 3 (`ObservationalForwardPipeline`: masks, selection, galaxy bias $b_1, b_2$, RSD multipoles $P_0, P_2, P_4$, shot noise, lensing shear calibration) must precede Stage 4 (SBI).
4. **No Random 80/20 Splits:** Always use `DatasetPartitionProtocol` to split by disjoint `(ic_seed, cosmo_id, family_id)`.
5. **Blocking Validation Gates:** Do not advance stages without verified SBC + TARP coverage and non-Gaussian statistics (1-point PDF, 3D Peak Counts, 3D Minkowski Functionals).
