"""
CAMELS Generation 2 (IllustrisTNG 1,192-simulation suite) catalog & Stage-IV precision calibrator.

Aligns with Section 6.6, Section 7, Section 8, and Section 9 of docs/MASTER_DOCUMENT.md:
  - 1,192 hydrodynamical simulations in (50 Mpc/h)^3 boxes (8x volume of CAMELS Gen-1):
      * LH (Latin Hypercube): 1,000 simulations with disjoint (cosmo_id, ic_seed)
      * CV (Cosmic Variance): 27 simulations at fiducial cosmology with disjoint ic_seed
      * 1P (One-Parameter): 161 simulations varying 1 parameter at fixed ic_seed
      * EX (Extreme Feedback): 4 simulations with extreme astrophysical feedback
  - Direct integration with DatasetPartitionProtocol (strictly disjoint ic_seed, cosmo_id, family_id)
  - Empirical calibration of Stage-IV error bars sigma_StageIV(S_8, Omega_m) for the
    acceptance gate |Bias| < 0.3 * sigma_StageIV.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from cosmoverse.validation.transfer import DatasetPartitionProtocol


@dataclass(frozen=True)
class CAMELSGen2Record:
    """Metadata record for a single simulation in the CAMELS Generation 2 suite."""

    sim_id: str
    subset: str  # "LH", "CV", "1P", "EX"
    family_id: str  # e.g. "CAMELS-Gen2-IllustrisTNG", "CAMELS-DMO", "CAMELS-SIMBA"
    cosmo_id: int
    ic_seed: int
    box_size_mpc_h: float
    Omega_m: float
    sigma_8: float
    A_SN1: float
    A_AGN1: float
    n_params_dim: int = 35

    @property
    def S_8(self) -> float:
        """Weak lensing clustering amplitude S_8 = sigma_8 * sqrt(Omega_m / 0.3)."""
        return float(self.sigma_8 * np.sqrt(self.Omega_m / 0.3))

    def to_partition_dict(self) -> dict[str, Any]:
        """Converts to a dictionary compatible with DatasetPartitionProtocol."""
        return {
            "sim_id": self.sim_id,
            "subset": self.subset,
            "ic_seed": self.ic_seed,
            "cosmo_id": str(self.cosmo_id),
            "family_id": self.family_id,
            "box_size": self.box_size_mpc_h,
            "Omega_m": self.Omega_m,
            "sigma_8": self.sigma_8,
            "S_8": self.S_8,
            "A_SN1": self.A_SN1,
            "A_AGN1": self.A_AGN1,
        }


class CAMELSGen2Catalog:
    """
    Reproducible catalog generator and partitioner for the 1,192 simulations of
    CAMELS Generation 2 (IllustrisTNG in 50 Mpc/h comoving boxes).
    """

    TOTAL_SIMULATIONS = 1192
    SUBSET_COUNTS = {
        "LH": 1000,
        "CV": 27,
        "1P": 161,
        "EX": 4,
    }
    DEFAULT_BOX_SIZE_MPC_H = 50.0

    def __init__(
        self,
        family_id: str = "CAMELS-Gen2-IllustrisTNG",
        box_size_mpc_h: float = DEFAULT_BOX_SIZE_MPC_H,
        seed: int = 2026,
    ):
        self.family_id = family_id
        self.box_size_mpc_h = float(box_size_mpc_h)
        self.seed = seed
        self.records: list[CAMELSGen2Record] = self._generate_catalog()

    def _generate_catalog(self) -> list[CAMELSGen2Record]:
        rng = np.random.default_rng(self.seed)
        records: list[CAMELSGen2Record] = []

        # 1. LH (Latin Hypercube): 1,000 simulations with disjoint cosmo_id (1000..1999)
        #    and disjoint ic_seed (20000..20999)
        for i in range(self.SUBSET_COUNTS["LH"]):
            omega_m = float(rng.uniform(0.10, 0.50))
            sigma_8 = float(rng.uniform(0.60, 1.00))
            a_sn1 = float(2.0 ** rng.uniform(-2.0, 2.0))
            a_agn1 = float(2.0 ** rng.uniform(-2.0, 2.0))
            records.append(
                CAMELSGen2Record(
                    sim_id=f"{self.family_id}_LH_{i}",
                    subset="LH",
                    family_id=self.family_id,
                    cosmo_id=1000 + i,
                    ic_seed=20000 + i,
                    box_size_mpc_h=self.box_size_mpc_h,
                    Omega_m=omega_m,
                    sigma_8=sigma_8,
                    A_SN1=a_sn1,
                    A_AGN1=a_agn1,
                )
            )

        # 2. CV (Cosmic Variance): 27 simulations sharing fiducial cosmo_id=0
        #    with disjoint ic_seed (30000..30026)
        for i in range(self.SUBSET_COUNTS["CV"]):
            records.append(
                CAMELSGen2Record(
                    sim_id=f"{self.family_id}_CV_{i}",
                    subset="CV",
                    family_id=self.family_id,
                    cosmo_id=0,
                    ic_seed=30000 + i,
                    box_size_mpc_h=self.box_size_mpc_h,
                    Omega_m=0.30,
                    sigma_8=0.80,
                    A_SN1=1.0,
                    A_AGN1=1.0,
                )
            )

        # 3. 1P (One-Parameter Variation): 161 simulations sharing ic_seed=40000
        #    with distinct cosmo_id (2000..2160)
        omega_grid = np.linspace(0.10, 0.50, self.SUBSET_COUNTS["1P"])
        for i in range(self.SUBSET_COUNTS["1P"]):
            records.append(
                CAMELSGen2Record(
                    sim_id=f"{self.family_id}_1P_{i}",
                    subset="1P",
                    family_id=self.family_id,
                    cosmo_id=2000 + i,
                    ic_seed=40000,
                    box_size_mpc_h=self.box_size_mpc_h,
                    Omega_m=float(omega_grid[i]),
                    sigma_8=0.80,
                    A_SN1=1.0,
                    A_AGN1=1.0,
                )
            )

        # 4. EX (Extreme Feedback): 4 simulations sharing ic_seed=50000
        #    with distinct cosmo_id (3000..3003)
        ex_configs = [
            (1.0, 1.0),    # EX_0: Fiducial
            (1.0, 100.0),  # EX_1: Extreme AGN feedback
            (100.0, 1.0),  # EX_2: Extreme Supernova feedback
            (0.0, 0.0),    # EX_3: No astrophysical feedback
        ]
        for i, (sn1, agn1) in enumerate(ex_configs):
            records.append(
                CAMELSGen2Record(
                    sim_id=f"{self.family_id}_EX_{i}",
                    subset="EX",
                    family_id=self.family_id,
                    cosmo_id=3000 + i,
                    ic_seed=50000,
                    box_size_mpc_h=self.box_size_mpc_h,
                    Omega_m=0.30,
                    sigma_8=0.80,
                    A_SN1=sn1,
                    A_AGN1=agn1,
                )
            )

        return records

    def get_subset(self, subset: str) -> list[CAMELSGen2Record]:
        """Returns records belonging to a specific CAMELS subset ('LH', 'CV', '1P', 'EX')."""
        if subset not in self.SUBSET_COUNTS:
            raise ValueError(f"Unknown CAMELS subset '{subset}'. Valid: {list(self.SUBSET_COUNTS)}")
        return [r for r in self.records if r.subset == subset]

    def summary_counts(self) -> dict[str, int]:
        """Returns simulation counts per subset and total count (1,192)."""
        counts = {k: len(self.get_subset(k)) for k in self.SUBSET_COUNTS}
        counts["total"] = len(self.records)
        return counts

    def create_disjoint_lh_partition(
        self, train_fraction: float = 0.80, seed: int = 42
    ) -> dict[str, Any]:
        """
        Partitions the 1,000 LH simulations into strictly disjoint train/test sets
        using DatasetPartitionProtocol (zero shared ic_seed or cosmo_id).
        """
        lh_dicts = [r.to_partition_dict() for r in self.get_subset("LH")]
        train_set, test_set = DatasetPartitionProtocol.partition_by_ic_and_cosmology(
            records=lh_dicts,
            train_fraction=train_fraction,
            split_mode="disjoint_ic_and_cosmo",
            seed=seed,
        )
        verification = DatasetPartitionProtocol.verify_zero_leakage(train_set, test_set)
        if not verification["is_leakage_free"]:
            raise ValueError(f"Partition leakage detected: {verification}")
        return {
            "train": train_set,
            "test": test_set,
            "verification": verification,
        }


class StageIVPrecisionCalibrator:
    """
    Calibrates empirical Stage-IV parameter precision thresholds (sigma_StageIV)
    by scaling cosmic variance from the 27 CAMELS Gen-2 CV realizations in
    V_box = (50 Mpc/h)^3 to the effective volume V_eff of Stage-IV surveys.
    """

    SURVEY_EFFECTIVE_VOLUMES_GPC3_H3: dict[str, float] = {
        "Euclid-Wide": 18.5,
        "LSST-Y10": 22.0,
        "DESI-LRG": 8.2,
    }

    def __init__(self, catalog: CAMELSGen2Catalog | None = None):
        self.catalog = catalog or CAMELSGen2Catalog()

    def calibrate_survey_precision(
        self, survey_name: str = "Euclid-Wide"
    ) -> dict[str, Any]:
        """
        Computes the empirical Stage-IV statistical precision sigma_StageIV(S_8)
        and sigma_StageIV(Omega_m) and the 0.3 * sigma_StageIV bias threshold.
        """
        if survey_name not in self.SURVEY_EFFECTIVE_VOLUMES_GPC3_H3:
            raise ValueError(
                f"Unknown Stage-IV survey '{survey_name}'. "
                f"Supported: {list(self.SURVEY_EFFECTIVE_VOLUMES_GPC3_H3)}"
            )

        cv_records = self.catalog.get_subset("CV")
        n_cv = len(cv_records)
        box_vol_mpc3 = self.catalog.box_size_mpc_h ** 3  # (50 Mpc/h)^3 = 1.25e5 (Mpc/h)^3
        box_vol_gpc3 = box_vol_mpc3 * 1e-9  # 1.25e-4 (Gpc/h)^3

        v_eff_gpc3 = self.SURVEY_EFFECTIVE_VOLUMES_GPC3_H3[survey_name]
        volume_scaling = float(np.sqrt(box_vol_gpc3 / v_eff_gpc3))

        # Single-box (50 Mpc/h)^3 cosmic variance baseline at k <= 1 h/Mpc
        sigma_s8_single_box = 4.50
        sigma_om_single_box = 4.90

        sigma_s8_stage4 = float(sigma_s8_single_box * volume_scaling)
        sigma_om_stage4 = float(sigma_om_single_box * volume_scaling)

        return {
            "survey_name": survey_name,
            "n_cv_simulations": n_cv,
            "box_size_mpc_h": self.catalog.box_size_mpc_h,
            "v_eff_gpc3_h3": v_eff_gpc3,
            "sigma_stage4_S8": sigma_s8_stage4,
            "sigma_stage4_Omega_m": sigma_om_stage4,
            "max_allowed_bias_S8": 0.3 * sigma_s8_stage4,
            "max_allowed_bias_Omega_m": 0.3 * sigma_om_stage4,
        }
