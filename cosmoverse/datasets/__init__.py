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

__all__ = [
    "CAMELSGen2Record",
    "CAMELSGen2Catalog",
    "StageIVPrecisionCalibrator",
]
