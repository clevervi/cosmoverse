# Changelog

Todos los cambios relevantes de **COSMOVERSE** se documentan en este archivo.
El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [0.4.0] - 2026-11-01

### Added
- `CAMELSFieldSnapshotDataset` y `build_leakage_guarded_dataloaders` en `cosmoverse.datasets.loaders`: carga y síntesis determinista de campos de densidad 3D $\delta(\mathbf{x})$ con bloqueo automático de fugas (`ic_seed`, `cosmo_id`) antes de entregar tensores de entrenamiento o prueba (#32, PR #34).
- `Paper1FigureGenerator` en `cosmoverse.validation.plotting`, script `experiments/generate_paper1_figures.py` y subcomando `cosmoverse figures`: generación determinista de las 4 figuras vectoriales SVG para los 4 Resultados Demostrables en `docs/figures/` embebidas en `README.md` (#33, PR #35).

## [0.3.0] - 2026-11-01

### Added
- `BaryonicEmulatorProtocol` y conmutador reversible hacia `BACCOemu` (`BACCOEmuPlanBModel`, Aricò et al. 2021) como Plan B para v2 manteniendo `BCEmu` como único emulador activo por defecto en v1 (#22, PR #25).
- Motor de inferencia diferenciable de campo completo `FieldLevelInferenceEngine` (Etapa 5 / Paper 2 Extension) con reconstrucción de condiciones iniciales 3D ($r(k) = 0.9728$) y benchmark de **MCLMC precondicionado** frente a **MCLMC sin precondicionar** y **HMC** (#23, PR #26).
- Generador de reportes auditables en JSON y Markdown `ScientificBenchmarkReporter` para los 4 Resultados Demostrables y sus criterios de aceptación bloqueantes (#24, PR #27).
- Subcomandos del CLI `cosmoverse`: `camels`, `field-level` y `audit-citations` junto con suite de pruebas `tests/test_cli.py`.

## [0.2.0] - 2026-11-01

### Added
- Auditoría bibliográfica en `docs/literature_audit.md`, registro `EMULATOR_BIBLIOGRAPHY_REGISTRY` y excepción `CitationVerificationError` bloqueando referencias no verificadas como `CosmoRGNet` (#8, PR #19).
- Módulo `cosmoverse.datasets.camels_gen2` con el catálogo reproducible de **1,192 simulaciones IllustrisTNG de CAMELS Generation 2** en cajas de $(50\,h^{-1}\mathrm{Mpc})^3$ (`LH=1000`, `CV=27`, `1P=161`, `EX=4`) y calibrador empírico `StageIVPrecisionCalibrator` (#9, PR #20).

## [0.1.0] - 2026-11-01

### Added
- Documento Maestro consolidado (`docs/MASTER_DOCUMENT.md`) tras dos rondas de revisión metodológica.
- Separación estricta en 3 niveles: (1) Física del Universo, (2) Proceso Observacional, (3) Inferencia Estadística y Validación (#2, PR #11).
- Etapa 1 (`cosmoverse.emulators`): Arquitectura primaria `TUNeSEmulator` (U-Net temporal) y estadísticas gaussianas y no gaussianas (`compute_pk_mape`, `compute_one_point_pdf`, `compute_peak_counts_3d`, `compute_minkowski_functionals_3d`) (#3, PR #12).
- Etapa 2 (`cosmoverse.baryons`): Emulador bariónico único para v1 `BCEmuModel` (`BCEmu1`, `BCEmu3`, `BCEmu7`) y analizador de la frontera sesgo–varianza `BiasVarianceFrontier` (#4, PR #13).
- Etapa 3 (`cosmoverse.observations`): `ObservationalForwardPipeline` integrado antes de la SBI con máscaras 3D, función de selección, *galaxy bias* ($b_1, b_2$), multipolos RSD ($P_0, P_2, P_4$), *shot noise* y calibración de cizalla débil (#5, PR #15).
- Etapa 4 y 5 (`cosmoverse.inference` y `cosmoverse.validation`): `SBIEngine` (`ConditionalRealNVP`), `MicrocanonicalHMC`, `SimulationBasedCalibration` (SBC + parámetro fijo), `TARPCalibration` y `ScaleDependentMisspecification` (#6, PR #16).
- Etapa 6 (`cosmoverse.validation.transfer`): `DatasetPartitionProtocol` (bloqueo de splits aleatorios 80/20 y partición disjunta por `ic_seed`, `cosmo_id` y `family_id`) y `CrossSimulationTransferEvaluator` (#7, PR #17).
- Gobernanza de repositorio: `.githooks`, `scripts/ci.ps1`, `scripts/ci.sh`, `scripts/check_repo.py`, plantillas de issues/PRs y `CODEOWNERS` (#1, PR #10).
