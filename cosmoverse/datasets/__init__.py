"""
Dataset catalogs and Stage-IV empirical precision calibration:
  - CAMELS Generation 2 (1,192 IllustrisTNG simulations in 50 Mpc/h boxes)
  - Disjoint partition integration with DatasetPartitionProtocol
  - Stage-IV statistical uncertainty calibrator (Euclid-Wide, LSST-Y10, DESI-LRG)
"""

from cosmoverse.datasets.camels_gen2 import (
    CAMELSGen2Catalog,
    CAMELSGen2Record,
    StageIVPrecisionCalibrator,
)
from cosmoverse.datasets.camels_spectra import CAMELSHydroSpectraConnector
from cosmoverse.datasets.loaders import (
    CAMELSFieldSnapshotDataset,
    build_camels_gen2_lh_dataloaders,
    build_leakage_guarded_dataloaders,
)

__all__ = [
    "CAMELSGen2Record",
    "CAMELSGen2Catalog",
    "StageIVPrecisionCalibrator",
    "CAMELSFieldSnapshotDataset",
    "build_leakage_guarded_dataloaders",
    "build_camels_gen2_lh_dataloaders",
    "CAMELSHydroSpectraConnector",
]
