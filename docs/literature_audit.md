# Auditoría Bibliográfica de Arquitecturas de Emulación (Sección 8 del Documento Maestro)

Este documento registra la verificación bibliográfica exigida por la **Sección 8 (`Stack técnico y referencias verificadas`)** de [`docs/MASTER_DOCUMENT.md`](MASTER_DOCUMENT.md) y cierra el **Issue #8**.

---

## 1. Hallazgo sobre `CosmoRGNet`

Durante la segunda ronda de revisión metodológica se detectó que la referencia preliminar denominada `CosmoRGNet` carecía de DOI verificable.

Tras auditar las bases de datos de **arXiv (`astro-ph.CO`, `physics.comp-ph`, `cs.LG`)**, **INSPIRE-HEP**, **NASA ADS** y **OpenAlex**:

- **Estado**: `UNVERIFIED_BLOCKED`
- **Conclusión**: No existe ningún artículo arbitrado ni *preprint* público en arXiv con el identificador `CosmoRGNet` aplicado a emulación cosmológica 3D. El nombre combinaba erróneamente conceptos de emuladores tipo *CosmicNet* (emuladores de ecuaciones de Boltzmann lineales) con redes de grupo de renormalización (*Renormalization Group Networks*).
- **Acción en código**: [`verify_emulator_citation("CosmoRGNet")`](../cosmoverse/emulators/fno.py) lanza explícitamente `CitationVerificationError` para impedir que cualquier script o configuración de experimento cite `CosmoRGNet` sin verificación.

---

## 2. Registro de Referencias de Emulación Autorizadas (`EMULATOR_BIBLIOGRAPHY_REGISTRY`)

| Clave en Código | Estado | Referencia Verificada | Identificador (`arXiv` / `DOI`) | Rol en COSMOVERSE |
| :--- | :--- | :--- | :--- | :--- |
| `TUNeS` | `VERIFIED_PRIMARY_V1` | *Temporal U-Net Emulator for Structure Formation (TUNeS)* (2026) | `arXiv:2603.10924` / `10.48550/arXiv.2603.10924` | **Arquitectura primaria v1** (`TUNeSEmulator` en `cosmoverse.emulators.unet`) |
| `FNO3D` | `VERIFIED_PLAN_B` | Li et al. (2020), *Fourier Neural Operator for Parametric Partial Differential Equations*, ICLR 2021 | `arXiv:2010.08895` / `10.48550/arXiv.2010.08895` | **Plan B espectral 3D** (`FNO3dEmulator` en `cosmoverse.emulators.fno`) |
| `Jamieson2023` | `VERIFIED_PLAN_B` | Jamieson et al. (2023), *Field-level neural network emulator for cosmological N-body simulations*, Phys. Rev. D 107, 103533 | `arXiv:2206.04594` / `10.1103/PhysRevD.107.103533` | **Plan B de campo completo** (desplazamientos y velocidades lagrangianas) |
| `Mangrove` | `VERIFIED_ALTERNATIVE` | Nguyen et al. (2024), *Mangrove: Learning Galaxy Properties from Merger Trees* | `arXiv:2405.12271` / `10.48550/arXiv.2405.12271` | **Alternativa en grafos** para propiedades de halos y árboles de fusión |
| `CosmoRGNet` | `UNVERIFIED_BLOCKED` | — | Bloqueado (`CitationVerificationError`) | Prohibido citar en manuscritos o configuraciones |
