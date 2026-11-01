# Datasets — CAMELS Generation 2 & Quijote Protocols

This directory and the companion module [`cosmoverse/datasets/camels_gen2.py`](../cosmoverse/datasets/camels_gen2.py) manage dataset metadata, disjoint partitioning, and empirical Stage-IV precision calibration for **COSMOVERSE**:

1. **CAMELS Generation 2 (`CAMELS-Gen2-IllustrisTNG`)**:
   - **1,192 hydrodynamical simulations** in $(50\,h^{-1}\mathrm{Mpc})^3$ comoving boxes ($8\times$ the volume of CAMELS Gen-1) spanning a 35-dimensional cosmological and astrophysical parameter space:
     - `LH` (*Latin Hypercube*): `1,000` simulations (`LH_0`–`LH_999`) with strictly disjoint `cosmo_id` and `ic_seed`.
     - `CV` (*Cosmic Variance*): `27` simulations (`CV_0`–`CV_26`) at fiducial cosmology (`cosmo_id = 0`) with disjoint `ic_seed`.
     - `1P` (*One-Parameter Variation*): `161` simulations (`1P_0`–`1P_160`) varying one parameter at a time at fixed `ic_seed = 40000`.
     - `EX` (*Extreme Feedback*): `4` simulations (`EX_0`–`EX_3`) probing extreme AGN/SN feedback regimes.
2. **CAMELS-DMO & CAMELS-SIMBA Counterparts**:
   - Used by `CrossSimulationTransferEvaluator` (`cosmoverse.validation.transfer`) to measure calibration degradation ($\Delta\text{Coverage}_{68/95}$) and systematic bias shift ($\Delta\text{Bias}$) under cross-simulation transfer.
3. **Stage-IV Precision Calibration (`StageIVPrecisionCalibrator`)**:
   - Scales the $(50\,h^{-1}\mathrm{Mpc})^3$ cosmic variance baseline from the 27 `CV` realizations to the effective volume $V_{\mathrm{eff}}$ of `Euclid-Wide`, `LSST-Y10`, and `DESI-LRG`, setting the quantitative acceptance threshold $|\text{Bias}| < 0.3\,\sigma_{\text{Stage-IV}}$.

## Mandatory Partitioning Rule (Section 7 of `docs/MASTER_DOCUMENT.md`)

- **NEVER** use naive random 80/20 splits (`split_mode="random_80_20"` is rejected by `DatasetPartitionProtocol`).
- **NEVER** split `CV` (which shares `cosmo_id=0`) or `1P` (which shares `ic_seed=40000`) across train and test sets (`DatasetPartitionProtocol.verify_zero_leakage` raises `ValueError`).
- Always partition `LH` strictly by `(ic_seed, cosmo_id, family_id)` via `CAMELSGen2Catalog.create_disjoint_lh_partition()`.
