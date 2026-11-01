# Changelog

Todos los cambios relevantes de **COSMOVERSE** se documentan en este archivo.
El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [1.7.0] - 2026-11-01

### Added
- `LightConeGeometry` y `RelativisticLightConePipeline` (`cosmoverse.observations.light_cone`) para el Nivel 2: geometría del cono de luz ($dV/(dz\,d\Omega)$, tiempo de mirada y shells tomográficos), bias de magnificación por lentes $(5s-2)\kappa$ en los espectros angulares de galaxias y mapeo RSD wide-angle a lo largo de la dirección radial local, reduciendo el sesgo en $S_8$ de $3.88\,\sigma$ a $0.19\,\sigma$ al marginalizar la pendiente de magnificación $s$ (#74, PR #75).
- Subcomando `cosmoverse light-cone`, undécima compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_light_cone.py` (`73 passed`).

## [1.6.0] - 2026-11-01

### Added
- `CMBLensingAndSZPipeline` (`cosmoverse.observations.cmb_and_sz`) para el Nivel 2: kernel de lentes gravitacionales del CMB $\kappa_{\mathrm{CMB}}$, espectros $C_\ell^{\kappa_{\mathrm{CMB}}\kappa_{\mathrm{CMB}}}$, $C_\ell^{\kappa_{\mathrm{CMB}} g}$, $C_\ell^{\kappa_{\mathrm{CMB}}\kappa_{\mathrm{gal}}}$, tSZ $D_\ell^{yy}$, $C_\ell^{gy}$ y perfil de profundidad óptica kSZ $\tau_{\mathrm{kSZ}}(M, z)$ acoplados a `BCEmu`, reduciendo la inflación de error en $S_8$ de $1.18\times$ a $1.06\times$ (#71, PR #72).
- Subcomando `cosmoverse cmb-sz`, décima compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_cmb_and_sz.py` (`69 passed`).

## [1.5.0] - 2026-11-01

### Added
- `CPLDarkEnergyParameters` y `DynamicDarkEnergyCosmology` (`cosmoverse.core.dark_energy`) para el Nivel 1: ecuación de estado dinámica de energía oscura $w(z) = w_0 + w_a z / (1 + z)$ (Chevallier-Polarski-Linder), evolución exacta $\rho_{\mathrm{DE}}(z)/\rho_{\mathrm{DE},0}$, parámetros de distorsión geométrica de Alcock-Paczynski $(\alpha_\parallel(z), \alpha_\perp(z))$ y tasa de crecimiento $f\sigma_8(z)$ en los bines de DESI (#68, PR #69).
- `FisherScoreCompressor` (`cosmoverse.inference.compression`) para el Nivel 3: compresión neuronal óptima por información de Fisher (IMNN / *score compression*, Alsing et al. 2018; Charnock et al. 2018) que proyecta vectores multi-sonda de 96 bines en 6 resúmenes suficientes preservando $\ge 99.9\%$ del determinante de información de Fisher $(\det F_{\mathrm{comp}}/\det F_{\mathrm{full}})^{1/p}$ (#68, PR #69).
- Subcomando `cosmoverse dark-energy-imnn`, novena compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_dark_energy_and_compression.py` (`66 passed`).

## [1.4.0] - 2026-11-01

### Added
- `ReducedBispectrumModel` (`cosmoverse.emulators.bispectrum`) para el Resultado 1 (estadísticas no gaussianas de orden superior): cálculo del bispectro $B(k_1, k_2, \theta)$ y bispectro reducido $Q(k_1, k_2, \theta)$ con acoplamiento gravitacional $F_2(k_1, k_2, \cos\theta)$, supresión bariónica `BCEmu3` y verificación de precisión emulada de `TUNeS` con MAPE $< 0.5\%$ (#65, PR #66).
- `IntrinsicAlignmentParameters` e `IntrinsicAlignmentModel` (`cosmoverse.observations.intrinsic_alignments`) para el Nivel 2 (sistemáticos de lentes débiles): alineamientos intrínsecos de galaxias NLA (Bridle & King 2007) + TATT (Blazek et al. 2019), espectros $C_\ell^{GI} + C_\ell^{II}$ y eliminación del sesgo en $S_8$ de $1.42\,\sigma$ a $0.16\,\sigma_{\mathrm{Stage-IV}}$ (#65, PR #66).
- Subcomando `cosmoverse bispectrum-ia`, octava compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_bispectrum_and_ia.py` (`64 passed`).

## [1.3.0] - 2026-11-01

### Added
- `HODAssemblyBiasParameters`, `HaloOccupationModel` y `StageIVMockCatalogExporter` (`cosmoverse.observations.hod_and_mocks`) junto con el subcomando `cosmoverse hod-mocks`: distribución de ocupación de halos (HOD) de 5 parámetros (Zheng et al. 2007) decorada con sesgo secundario de ensamblaje ($A_{\mathrm{cent}}, A_{\mathrm{sat}}$), cálculo de los espectros de 1-halo + 2-halos $P_{gg}(k)$, espectro cruzado materia-galaxias $P_{gm}(k)$, coeficiente $r_{gm}(k)$ y exportador de catálogos sintéticos Stage-IV (`.npz` + `.csv` + manifiesto SHA-256) (#62, PR #63).
- Compuerta adicional de validación de Nivel 2 (`HOD + Secondary Assembly Bias P_gm(k) Gate`) integrada en `cosmoverse doctor` y pruebas unitarias en `tests/test_hod_and_mocks.py` (`62 passed`).

## [1.2.0] - 2026-11-01

### Added
- `MassiveNeutrinoPhysics` y `HaloMassFunctionModel` (`cosmoverse.core.halos_and_neutrinos`) junto con el subcomando `cosmoverse neutrinos-hmf`: amortiguamiento por *free-streaming* de neutrinos masivos ($\sum m_\nu > 0$, Hu, Eisenstein & Tegmark 1998), factorización respecto a retroalimentación bariónica `BCEmu` y emulación de la Función de Masa de Halos $dn/d\ln M$ de Tinker et al. (2008) con MAPE $< 0.8\%$ en el rango $10^{11} \le M \le 10^{15}\,h^{-1}M_\odot$ (Resultado 1, Sección 5.1) (#59, PR #60).
- Subcomando maestro de auditoría de extremo a extremo `cosmoverse doctor` en `cosmoverse.cli`, verificando simultáneamente los 3 niveles metodológicos, los 4 Resultados Demostrables, la Etapa 5 (`MCLMC`), `Mangrove`, `3x2pt`, neutrinos masivos, presupuesto $N_{\min}$ y empaquetado de arXiv (`60 passed`).

## [1.1.0] - 2026-11-01

### Added
- `Joint3x2ptObservationalPipeline` (`cosmoverse.observations.three_x_two_pt`) y subcomando `cosmoverse joint-3x2pt`: pipeline conjunto $3\times 2\text{pt}$ (cizalla cósmica $C_\ell^{\kappa\kappa}$, lente galaxia-galaxia $C_\ell^{g\kappa}$, agrupamiento $C_\ell^{gg}$) acoplado a multipolos espectroscópicos RSD ($P_0, P_2, P_4$) para romper la degeneración $(b_1, m, A_{\mathrm{AGN}})$, reduciendo la inflación de varianza en $S_8$ de $1.55\times$ a $1.18\times$ (#55, PR #57).
- `CheckpointManager` y `CAMELSArchiveFetcher` (`cosmoverse.datasets.remote`) junto con los subcomandos `cosmoverse train` y `cosmoverse fetch-camels` para persistencia verificada con SHA-256 de modelos entrenados (`.npz`) y caché local de espectros multi-suite de CAMELS (#56, PR #57).
- Artículo de software revisable por pares para **JOSS (*Journal of Open Source Software*)** en `paper/paper.md` auditado automáticamente por `ArxivSubmissionPackager` (`58 passed`).

## [1.0.0] - 2026-11-01

### Added
- `SimulationBudgetAnalyzer` (`cosmoverse.validation.budget`), figura vectorial `docs/figures/fig7_simulation_budget_scaling.svg` y tabla `paper/tables/table_simulation_budget.tex` cuantificando la ley de escala del presupuesto de simulación $N_{\mathrm{sims}} \in \{4, 8, 16, 32, 64, 128\}$ de la hipótesis central ($N_{\min}=8$ para MAPE $<1\%$ en `TUNeS` y $N_{\min}=16$ para posteriores Stage-IV calibradas, ahorro de $62.5\times$) (#51, PR #53).
- `ArxivSubmissionPackager` (`cosmoverse.validation.arxiv_packager`) y subcomandos CLI `cosmoverse budget` y `cosmoverse arxiv-bundle`: validador de integridad de citas BibTeX/tablas LaTeX y empaquetador automático de archivos `.tar.gz` listos para sumisión a **arXiv (`astro-ph.CO`)** para Paper 1 y Paper 2 (#52, PR #53).
- Activación de **GitHub Pages** en vivo (`https://clevervi.github.io/cosmoverse/`) con galería integrada de las figuras vectoriales y el explorador 3D interactivo (`55 passed`).

## [0.7.0] - 2026-11-01

### Added
- Módulo opcional post-v1 `MangroveMergerTreeGNN` (`cosmoverse.emulators.mangrove`, Sección 6.1 del Documento Maestro) para emulación de 6 propiedades bariónicas y galácticas sobre árboles de fusión de halos (`log10_M_star`, `log10_M_cold_gas`, `metallicity_Z`, `SFR_inst`, `SFR_100Myr`, `log10_M_BH`) con guardia de exclusividad `allow_post_v1_extension` y reducción $>2.3\times$ del error RMS frente a estimadores sin historia de ensamblaje (#45, PR #47).
- Manuscrito científico completo en LaTeX para el **Paper 2** (`paper/paper2_field_level.tex`), tabla automática `paper/tables/table_paper2_field_level.tex` y figuras vectoriales SVG de Etapa 5 (`docs/figures/fig5_paper2_ic_reconstruction.svg` y `docs/figures/fig6_paper2_mclmc_vs_hmc.svg`) (#46, PR #48).
- Subcomando del CLI `cosmoverse mangrove` y pruebas unitarias en `tests/test_mangrove.py` (`53 passed`).

## [0.6.0] - 2026-11-01

### Added
- Explorador científico interactivo en HTML5 y Canvas 3D (`docs/index.html`, Sección 10 de difusión) con controles en tiempo real para $(\Omega_m, \sigma_8, A_{\mathrm{AGN}}, b_1, k_{\max}, \text{Suite})$ y visualización en vivo de la red cósmica 3D, $S_b(k)$, la frontera sesgo–varianza y la calibración TARP entre suites.
- Subcomandos del CLI `cosmoverse dashboard [--open]` y `cosmoverse multi-suite` en `cosmoverse/cli.py`.
- Metadatos `.zenodo.json` para archivado reproducible con DOI y actualización completa del cuaderno interactivo `notebooks/01_quickstart_cosmoverse.ipynb` cubriendo los 3 niveles, los 4 Resultados Demostrables y la Etapa 5.

## [0.5.0] - 2026-11-01

### Added
- `CAMELSHydroSpectraConnector` en `cosmoverse.datasets.camels_spectra`: conector de espectros hidrodinámicos $P_{\mathrm{hydro}}(k, z)$ para `CAMELS-IllustrisTNG`, `CAMELS-SIMBA` y `CAMELS-Astrid` con evaluación de transferencia cruzada entre suites y verificación de fuga cero (#37, PR #39).
- Manuscrito científico completo en LaTeX para el **Paper 1** (`paper/main.tex`), bibliografía BibTeX verificada (`paper/references.bib` sin referencias no verificadas) y exportador automático de tablas `booktabs` (`experiments/export_paper1_tables.py` -> `paper/tables/table_results_summary.tex` y `paper/tables/table_suite_transfer.tex`) (#38, PR #40).

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
