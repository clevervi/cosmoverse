---
title: 'COSMOVERSE: A Reproducible Hybrid Protocol for Neural Cosmological Emulation, Baryonic Marginalization, and Calibrated Bayesian Inference'
tags:
  - Python
  - cosmology
  - large-scale structure
  - simulation-based inference
  - baryonic feedback
  - neural emulators
authors:
  - name: Adrian Dario V.
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: COSMOVERSE Open Computational Cosmology Project
    index: 1
date: 1 November 2026
bibliography: references.bib
---

# Summary

`COSMOVERSE` is an open-source Python framework and reproducible scientific protocol that connects fast three-dimensional neural emulation (`TUNeS`), explicit parametric marginalization of baryonic feedback (`BCEmu` / `BACCOemu`), Stage-IV observational forward modeling (survey masks, radial selection functions, nonlinear galaxy bias, redshift-space distortion multipoles $P_0, P_2, P_4$, and joint $3\times 2\text{pt}$ weak lensing spectra), and Simulation-Based Inference (SBI) with Conditional Normalizing Flows, alongside an advanced Stage 5 module for differentiable field-level initial condition reconstruction using preconditioned Microcanonical Langevin Monte Carlo (`MCLMC`).

# Statement of Need

Stage-IV cosmological surveys—including Euclid, the Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST), and the Dark Energy Spectroscopic Instrument (DESI)—achieve sub-percent statistical precision on scales $k \sim 0.1\text{--}2\,h\,\mathrm{Mpc}^{-1}$. In this regime, two distinct sources of systematic error threaten cosmological parameter recovery:

1. **Physical and Observational Systematics**: Active galactic nuclei (AGN) and supernova feedback redistribute baryons on scales $k \gtrsim 0.3\,h\,\mathrm{Mpc}^{-1}$ [@GiriSchneider2021; @Arico2021BACCO], while redshift-space distortions and survey geometry couple modes.
2. **Methodological Leakage and Uncalibrated Posteriors**: Mixing physical evolution, observational effects, and statistical inference in ad-hoc pipelines—or evaluating neural emulators on random train/test splits that share initial condition seeds (`ic_seed`) or cosmologies (`cosmo_id`)—produces overconfident posteriors that fail when transferred across hydrodynamical simulation suites [@VillaescusaNavarro2021CAMELS].

Rather than proposing another standalone emulator or sampler in isolation, `COSMOVERSE` provides a strict **three-level methodological separation** and automated **blocking acceptance gates** enforcing posterior calibration via Simulation-Based Calibration (`SBC`) [@Talts2018SBC] and Tests of Accuracy with Random Points (`TARP`) [@Lemos2023TARP].

# Three-Level Architecture and Key Features

1. **Level 1 — Universe Physics (`cosmoverse.core`, `simulation`, `emulators`, `baryons`)**:
   - Temporal U-Net field emulation (`TUNeSEmulator`) [@TUNeS2026] achieving $< 1\%$ $P(k)$ MAPE for $k \le 1\,h\,\mathrm{Mpc}^{-1}$ from as few as $N_{\min} = 8$ training simulations, validated on 1-point PDFs, 3D peak counts, and 3D Minkowski functionals $V_0\text{--}V_3$.
   - Optional post-v1 merger-tree Graph Neural Network (`MangroveMergerTreeGNN`) [@Jespersen2024Mangrove] and reversible Fourier Neural Operator (`FNO3dEmulator`) [@Li2020FNO; @Jamieson2023FieldEmulator].
   - Explicit baryonic suppression $S_b(k, z)$ via `BCEmu1/3/7` [@GiriSchneider2021] with a reversible `BACCOemu` [@Arico2021BACCO] switch and full Bias–Variance Frontier quantification.
2. **Level 2 — Observational Realism Integrated Before SBI (`cosmoverse.observations`)**:
   - 3D survey masks, radial selection functions, Eulerian galaxy bias $(b_1, b_2)$, RSD multipoles $(P_0, P_2, P_4)$, and joint $3\times 2\text{pt}$ cosmic shear and galaxy-galaxy lensing pipelines (`Joint3x2ptObservationalPipeline`).
3. **Level 3 — Calibrated Inference and Cross-Simulation Transfer (`cosmoverse.inference`, `validation`)**:
   - Conditional RealNVP Normalizing Flows (`SBIEngine`) and Fourier-preconditioned `MCLMC` (`FieldLevelInferenceEngine`) [@SimonOnfroy2025JaxPMMCLMC; @Robnik2023MCLMC].
   - `DatasetPartitionProtocol` blocking naive 80/20 splits and `CAMELSHydroSpectraConnector` quantifying cross-suite transfer from `CAMELS-DMO + BCEmu3` to `CAMELS-IllustrisTNG`, `CAMELS-SIMBA`, and `CAMELS-Astrid`.

# Acknowledgements

This work builds upon the public specifications and benchmarks of the CAMELS project [@VillaescusaNavarro2021CAMELS], `TUNeS` [@TUNeS2026], `BCEmu` [@GiriSchneider2021], `TARP` [@Lemos2023TARP], and `MCLMC` [@Robnik2023MCLMC].
