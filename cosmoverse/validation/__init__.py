"""
Scientific validation (Non-Negotiable — Master Document Section 6.6 & 7):
  - SBC (Simulation-Based Calibration + fixed-parameter checks)
  - TARP (Tests of Accuracy with Random Points)
  - EmpiricalCoverage
  - ScaleDependentMisspecification
  - DatasetPartitionProtocol (strict disjoint ic_seed, cosmo_id, family_id)
  - CrossSimulationTransferEvaluator (Resultado 4 central contribution)
  - CAMELS Gen-2 & LHS Benchmarks
"""

from cosmoverse.validation.arxiv_packager import ArxivSubmissionPackager
from cosmoverse.validation.benchmarks import (
    camels_gen2_catalog_metadata,
    camels_latin_hypercube_benchmark,
    generate_latin_hypercube_samples,
)
from cosmoverse.validation.budget import SimulationBudgetAnalyzer
from cosmoverse.validation.coverage import EmpiricalCoverage
from cosmoverse.validation.misspecification import ScaleDependentMisspecification
from cosmoverse.validation.plotting import Paper1FigureGenerator
from cosmoverse.validation.progress_dashboard import ProgressDashboard
from cosmoverse.validation.reporting import ScientificBenchmarkReporter
from cosmoverse.validation.sbc import SimulationBasedCalibration
from cosmoverse.validation.tarp import TARPCalibration
from cosmoverse.validation.transfer import (
    CrossSimulationTransferEvaluator,
    DatasetPartitionProtocol,
)

__all__ = [
    "SimulationBasedCalibration",
    "TARPCalibration",
    "EmpiricalCoverage",
    "ScaleDependentMisspecification",
    "DatasetPartitionProtocol",
    "CrossSimulationTransferEvaluator",
    "ScientificBenchmarkReporter",
    "Paper1FigureGenerator",
    "ProgressDashboard",
    "SimulationBudgetAnalyzer",
    "ArxivSubmissionPackager",
    "generate_latin_hypercube_samples",
    "camels_latin_hypercube_benchmark",
    "camels_gen2_catalog_metadata",
]
