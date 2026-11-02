# Changelog

Todos los cambios relevantes de **COSMOVERSE** se documentan en este archivo.
El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).

## [3.3.0] - 2026-11-01

### Added
- `MultiFamilyTransferPipeline` (`cosmoverse.inference.multi_family_transfer`): auditoría de transferencia a `CAMELS-IllustrisTNG`, `CAMELS-SIMBA` y `CAMELS-Astrid`, con partición leakage-free y flujo PM + MLP como fuente (#101, PR #102).
- `model_data_vector_with_response` en el pipeline PM para inyectar una supresión objetivo arbitraria \(S_{\rm target}(k)\).
- Hook `--data-dir` para cargar tablas reales `{sim_id}_Pk.txt`; por defecto se usan curvas CAMELS calibradas y el resultado se etiqueta `calibrated_camels_response` o `disk_table`.
- Subcomando `cosmoverse multi-family-transfer`, vigésima compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_multi_family_transfer.py` (`99 passed`).

### Validation
- Fuente calibrada y partición leakage-free: **PASS**.
- Pre-transferencia: cobertura 65.9% / 93.2%, mediana de sesgo 0.75 $\sigma$, TARP-ICE 0.021.
- TNG: post 67% / 75%, mediana 1.39 $\sigma$, $\Delta$TARP-ICE $+0.225$; cota **FAIL**.
- SIMBA: post 49% / 58%, mediana 2.35 $\sigma$, $\Delta$TARP-ICE $+0.148$; cota **FAIL**.
- Astrid: post 74% / 79%, mediana 1.37 $\sigma$, $\Delta$TARP-ICE $+0.221$; cota **FAIL**.
- Compuerta metodológica multi-familia: **PASS**; cota de degradación para las tres familias: **FAIL**. Resultado negativo del Resultado 4: el flujo entrenado en DMO + BCEmu no transfiere robustamente a las respuestas hidrodinámicas calibradas.

### Notes
- El repositorio **no incluye tablas CAMELS reales**. El `CAMELSArchiveFetcher` existente genera archivos a partir de las curvas calibradas; no descarga datos públicos.
- El hook `--data-dir` carga tablas reales si el usuario las provee, pero usa la **forma de la respuesta** de la tabla; no realiza una inferencia completa de parámetros CAMELS reales (redshift y parameterización quedan como limitaciones declaradas).

## [3.2.0] - 2026-11-01

### Added
- `FlowTransferPipeline` (`cosmoverse.inference.transfer_sbi`) para el Resultado 4: entrena el flujo PM + MLP en la familia fuente `CAMELS-DMO + BCEmu` y evalúa la degradación en una familia objetivo `CAMELS-IllustrisTNG` con respuesta hidrodinámica controlada y partición leakage-free (#98, PR #99).
- Modo `family="target"` y `mismatch_strength` en los forward models PM y multi-sonda.
- Métricas robustas de transferencia en `CrossSimulationTransferEvaluator`: mediana de sesgo en $\sigma$, percentil 95, fracción dentro de $1\sigma$ y $\Delta$TARP-ICE.
- Subcomando `cosmoverse flow-transfer`, decimonovena compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_transfer_sbi.py` (`97 passed`).

### Validation
- Fuente: flujo PM + MLP (v3.1, 1600 simulaciones), calibración SBC/TARP fuente **PASS**.
- Mismatch físico TNG/BCEmu (`strength=1.0`): cobertura pre 65.9% / 88.6%, post 67.0% / 72.0%; mediana de sesgo 0.63 -> 1.02 $\sigma$; $\Delta$TARP-ICE $+0.149$; compuerta de transferencia **PASS** (degradación acotada).
- Stress-test 3x: mediana de sesgo 2.09 $\sigma$; $\Delta$TARP-ICE $+0.191$; compuerta **FAIL** (degradación no acotada), como se espera.

### Notes
- La respuesta TNG/BCEmu es una curva calibrada, no una simulación hidrodinámica real.
- La compuerta usa tolerancias pre-registradas: mediana de sesgo < 1.10 $\sigma$, cobertura 95% $\ge$ 0.70 y $\Delta$TARP-ICE $\le$ 0.20.

## [3.1.0] - 2026-11-01

### Added
- Redes de acoplamiento MLP en `AffineCouplingLayer` (`hidden_dim`) con backprop manual completo, y permutaciones aleatorias por capa para mejorar el mixing de dimensiones en flujos con MLP (#95, PR #96).
- Opción `flow_hidden_dim` en `SBIEngine`, `MultiProbeSBIPipeline` y `PMEmulatorSBIPipeline`.
- Subcomando `cosmoverse pm-flow-sbi`, decimoctava compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_flow_training.py` y `tests/test_pm_emulator_sbi.py` (`96 passed`).

### Validation
- Emulador PM (LOO 0.82%) + flujo MLP: 1600 simulaciones de entrenamiento, 6 capas, `hidden=16`, 800 épocas, `scale_bound=6.0`, 100 trials SBC y 100 TARP.
- SBC 68%: \(\Omega_m=0.60\), \(\sigma_8=0.67\), \(b_1=0.65\), \(A_{\mathrm{AGN}}=0.63\); SBC 95%: 0.91-0.92; KS \(p>0.04\).
- TARP 68%=0.67, 95%=0.94, KS \(p=0.165\); compuerta conjunta **PASS**.

### Notes
- El flujo lineal de v3.0 conserva la permutación de reversión histórica; las permutaciones aleatorias se usan solo con MLP.
- El coste de entrenamiento del flujo MLP es mayor (~30 s) pero sigue siendo CPU-only y sin dependencias nuevas.

## [3.0.0] - 2026-11-01

### Added
- Entrenamiento por gradiente del `ConditionalRealNVP` (`cosmoverse.inference.flows`): `log_prob`, `inverse` con caché, `backward_log_prob` con backprop manual en NumPy, optimizador Adam y `scale_bound` configurable para controlar la contracción de escala.
- Modo `density_estimator="flow"` en `SBIEngine`: la posterior se aprende con un flujo condicional entrenado de verdad, sin la regresión cuadrática + residuales aleatorios del modo `"regression"`.
- Subcomando `cosmoverse flow-sbi`, decimoséptima compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_flow_training.py` (`94 passed`).

### Validation
- Multi-sonda 4D \(w(\theta)+P_0+P_2+P_4\), 400 simulaciones de entrenamiento.
- Flujo de 6 capas, 800 épocas, Adam con `lr=1e-3`, `scale_bound=5.0`.
- SBC 68%: \(\Omega_m=0.66\), \(\sigma_8=0.74\), \(b_1=0.74\), \(A_{\mathrm{AGN}}=0.66\); SBC 95%: 0.96-1.00; KS \(p>0.01\).
- TARP 68%=0.74, 95%=0.96, KS \(p=0.256\); compuerta conjunta **PASS**.

### Notes
- Backprop manual en NumPy; no se añadió JAX ni PyTorch como dependencia obligatoria.
- El pipeline PM-emulator (v2.2.0) sigue usando el modo `"regression"`; su calibración con flujo entrenado requiere ajuste adicional y queda para v3.1.

## [2.2.0] - 2026-11-01

### Added
- `PMPowerSpectrumEmulator` (`cosmoverse.emulators.pm_pk`) para el Nivel 1: superficie de respuesta cuadrática para $\log P(k;\Omega_m,\sigma_8)$ entrenada sobre una grilla $5\times5$ de simulaciones PM reales (caja $200\,h^{-1}\mathrm{Mpc}$, malla $32^3$, $z=0.1$), con extrapolación híbrida hacia la forma analítica de Eisenstein-Hu fuera del rango medido (#89, PR #90).
- `PMEmulatorSBIPipeline` (`cosmoverse.inference.pm_emulator_sbi`) para el Nivel 3: SBI multi-sonda de cuatro parámetros usando el emulador PM de $P(k)$ en lugar del surrogate analítico, con SBC/TARP bloqueantes.
- Subcomando `cosmoverse pm-emulator-sbi`, decimosexta compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_pm_emulator_sbi.py` (`92 passed`).

### Validation
- Emulador PM: MAPE de entrenamiento 0.60%, LOO 0.82% (compuerta LOO < 2%).
- SBC 68%: $\Omega_m=0.70$, $\sigma_8=0.64$, $b_1=0.70$, $A_{\mathrm{AGN}}=0.70$; SBC 95%: 0.88-0.96; KS $p>0.06$.
- TARP 68%=0.60, 95%=0.92, KS $p=0.308$; compuerta conjunta **PASS**.

### Notes
- La proyección angular $w(\theta)$ sigue siendo Limber + Kaiser/FoG, no un ray-tracing completo del cono de luz.
- El emulador PM cubre $k\approx0.037-0.423\,h\,\mathrm{Mpc}^{-1}$; fuera de ese rango se usa la forma analítica con corrección de amplitud en los bordes.

## [2.1.0] - 2026-11-01

### Added
- `MultiProbeSBIPipeline` (`cosmoverse.inference.multiprobe_sbi`) para el Nivel 3: SBI de cuatro parámetros $(\Omega_m, \sigma_8, b_1, A_{\mathrm{AGN}})$ sobre el vector conjunto $[w(\theta), P_0, P_2, P_4]$ (12 + 24 bines). El cuádrupolo y el hexadecárupolo rompen la degeneración $\sigma_8$–$b_1$, permitiendo liberar $b_1$ (#86, PR #87).
- `compute_light_cone_w_theta` y `compute_rsd_multipoles` como forward models compartidos; `LightConeSBIPipeline` ahora reutiliza el primero.
- Subcomando `cosmoverse multiprobe-sbi`, decimoquinta compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_multiprobe_sbi.py` (`89 passed`).

### Validation
- 400 simulaciones de entrenamiento, 50 trials SBC y 50 trials TARP.
- SBC 68%: $\Omega_m=0.72$, $\sigma_8=0.66$, $b_1=0.70$, $A_{\mathrm{AGN}}=0.62$; SBC 95%: 0.88-0.96; KS $p>0.08$.
- TARP 68%=0.70, 95%=0.88, KS $p=0.741$; compuerta conjunta **PASS**.

### Notes
- El forward model sigue siendo un surrogate Limber + Kaiser/FoG + BCEmu, no una simulación PM completa por muestra.
- La liberación de $b_1$ requiere el cuádrupolo RSD; solo con $w(\theta)$ el modo 4D no pasa SBC.

## [2.0.0] - 2026-11-01

### Added
- `LightConeSBIPipeline` (`cosmoverse.inference.light_cone_sbi`) para el Nivel 3: inferencia SBI de extremo a extremo sobre el vector de 12 bines $w(\theta)$ del cono de luz, usando un surrogate rápido Limber + BCEmu y los parámetros $(\Omega_m, \sigma_8, A_{\mathrm{AGN}})$ con $b_1=1.65$ fijo (#83, PR #84).
- `SBIEngine` ampliado con `feature_mode="quadratic"` (proyección condicional cuadrática) y `covariance_inflation` (corrección de calibración de la covarianza residual), manteniendo el modo lineal por defecto.
- Subcomando `cosmoverse light-cone-sbi`, decimocuarta compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_light_cone_sbi.py` (`86 passed`).

### Validation
- 200 simulaciones de entrenamiento, 50 trials SBC y 50 trials TARP.
- SBC 68%: $\Omega_m=0.72$, $\sigma_8=0.78$, $A_{\mathrm{AGN}}=0.70$; SBC 95%: 0.96-1.00; KS $p>0.25$.
- TARP 68%=0.62, 95%=0.94, KS $p=0.741$; compuerta conjunta **PASS**.

### Notes
- El modo de 4 parámetros con $b_1$ libre no pasa SBC por la degeneración $\sigma_8$–$b_1$; se fija $b_1$ al prior galáctico y se infieren 3 parámetros. Un análisis multi-sonda (p. ej. $w(\theta)$ + RSD) queda para v2.1.
- El forward model de entrenamiento es un surrogate Limber + BCEmu, no una simulación PM completa por muestra.

## [1.9.0] - 2026-11-01

### Added
- `SurveyMask`, `GalaxyBiasSampler`, `AngularClusteringMeasurer` y `LightConeSurveyPipeline` (`cosmoverse.observations.light_cone`) para el Nivel 2: máscara esférica RA/Dec/redshift, muestreo de galaxias con sesgo lineal sobre un proxy de densidad kNN, estimadores Davis-Peebles auto y cruzados, y medición de $w(\theta)$ frente a la predicción Limber. El catálogo de prueba produce 12,799 objetos de materia, 2,843 galaxias, $b_{\mathrm{eff}}=1.09$ y $r(w_{gm},w_{mm})=0.964$; el control sin sesgo ($b_1=0$) falla con $b_{\mathrm{eff}}=0.90$ (#80, PR #81).
- Subcomando `cosmoverse light-cone-survey`, decimotercera compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_light_cone_survey.py` (`83 passed`).

### Notes
- El sesgo efectivo recuperado (1.09) es menor que el $b_1$ de entrada (1.50) porque el proxy de densidad kNN atenúa la señal. La compuerta valida la presencia de un sesgo positivo ($b_{\mathrm{eff}}>1$) y la coherencia espacial $w_{gm}\propto w_{mm}$, no la recuperación exacta de $b_1$.

## [1.8.0] - 2026-11-01

### Added
- `ParticleMeshSimulator.run_with_snapshots` (`cosmoverse.simulation.nbody`) para registrar posiciones, velocidades y densidad en redshifts intermedios sin duplicar el integrador KDK.
- `LightConeBuilder` (`cosmoverse.observations.light_cone`) para construir un catálogo de cono de luz pasado a partir de snapshots PM: tiling periódico a lo largo de la dirección radial, selección en shells comóviles, RSD wide-angle, métricas de ocupación y exportación NPZ con SHA-256 (#77, PR #78).
- Subcomando `cosmoverse light-cone-build`, duodécima compuerta en `cosmoverse doctor` y pruebas unitarias en `tests/test_light_cone_builder.py` (`79 passed`).

### Fixed
- Normalización de las velocidades peculiares iniciales en `setup_initial_particles`: se eliminó una división espuria por `n_mesh^3` que reducía las velocidades a ~0.03 km/s; el PM produce ahora ~10^2 km/s y el RSD del cono de luz es no nulo.

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
