# Datasets & Mock Catalogs

This directory contains observational datasets, mock galaxy catalogs, and simulation suites used by `cosmoverse`.

## Directory Structure
- `datasets/camels/`: Subsets of the CAMELS simulation suite (Latin Hypercube & 1P sets).
- `datasets/surveys/`: Redshift distributions, footprints, and covariance matrices for DESI, Euclid, and LSST.
- `datasets/mocks/`: Fast synthetic galaxy mock catalogs generated from Particle-Mesh simulations.

## Generating Mock Observations
To generate a mock power spectrum dataset:
```python
from cosmoverse.core import Cosmology, CosmologicalParameters
from cosmoverse.observations import SurveySpecification, inject_observational_noise

cosmo = Cosmology(CosmologicalParameters.planck2018())
survey = SurveySpecification.desi_lrg()
# See examples/02_matter_power_spectrum.py for end-to-end generation
```
