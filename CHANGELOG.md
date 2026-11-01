# Changelog

Todos los cambios relevantes de **COSMOVERSE** se documentan en este archivo.
El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [Unreleased]

### Added
- Documento Maestro consolidado (`docs/MASTER_DOCUMENT.md`) tras dos rondas de revisión metodológica.
- Separación estricta en 3 niveles: (1) Física del Universo, (2) Proceso Observacional, (3) Inferencia Estadística y Validación.
- Etapa 1 (`cosmoverse.emulators`): Arquitectura primaria `TUNeSEmulator` (U-Net temporal) y estadísticas gaussianas y no gaussianas (`compute_pk_mape`, `compute_one_point_pdf`, `compute_peak_counts_3d`, `compute_minkowski_functionals_3d`).
- Etapa 2 (`cosmoverse.baryons`): Emulador bariónico único para v1 `BCEmuModel` (`BCEmu1`, `BCEmu3`, `BCEmu7`) y analizador de la frontera sesgo–varianza `BiasVarianceFrontier` frente a precisión Stage-IV.
- Etapa 3 (`cosmoverse.observations`): `ObservationalForwardPipeline` integrado antes de la SBI con máscaras 3D, función de selección, *galaxy bias* ($b_1, b_2$), distorsiones en espacio de redshifts (multipolos $P_0, P_2, P_4$), *shot noise* y calibración de cizalla débil.
- Etapa 4 (`cosmoverse.inference` y `cosmoverse.validation`): `SBIEngine` con `ConditionalRealNVP`, validación mediante `SimulationBasedCalibration` (SBC + chequeo a parámetro fijo), `TARPCalibration` (Tests of Accuracy with Random Points) y `ScaleDependentMisspecification`.
- Etapa 5 (`cosmoverse.inference.mclmc`): `MicrocanonicalHMC` con precondicionamiento explícito del espacio latente e integrador isocinético en $S^{d-1}$ junto con baseline HMC.
- Etapa 6 (`cosmoverse.validation.transfer`): `DatasetPartitionProtocol` (prohibiendo splits aleatorios 80/20 y exigiendo partición disjunta por `ic_seed`, `cosmo_id` y `family_id`) y `CrossSimulationTransferEvaluator`.
- Gobernanza de repositorio: `.githooks`, `scripts/ci.ps1`, `scripts/ci.sh`, `scripts/check_repo.py`, plantillas de issues/PRs y `CODEOWNERS`.
