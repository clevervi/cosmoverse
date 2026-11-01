"""
Survey specifications, geometry, and redshift distributions for DESI, Euclid, and LSST.
"""

from typing import Union

import numpy as np
from pydantic import BaseModel, Field


class SurveySpecification(BaseModel):
    """
    Cosmological survey geometry and galaxy sample properties.
    """
    name: str = Field(description="Survey name, e.g. DESI, Euclid, LSST")
    area_deg2: float = Field(description="Survey coverage area in square degrees")
    z_min: float = Field(default=0.0, ge=0.0, description="Minimum redshift")
    z_max: float = Field(default=2.5, ge=0.0, description="Maximum redshift")
    galaxy_density_arcmin2: float = Field(
        default=30.0, description="Effective galaxy surface number density in arcmin^-2"
    )
    shape_noise_sigma_e: float = Field(
        default=0.26, description="Intrinsic ellipticities shape noise per component"
    )

    @property
    def f_sky(self) -> float:
        """Fraction of the full sky covered by the survey."""
        full_sky_deg2 = 4.0 * np.pi * (180.0 / np.pi) ** 2  # ~ 41252.96 deg^2
        return self.area_deg2 / full_sky_deg2

    def redshift_distribution(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Normalized Smail-type redshift distribution n(z) = z^alpha * exp(-(z/z0)^beta).
        """
        z_arr = np.asarray(z, dtype=np.float64)
        z0 = 0.5 * (self.z_max + self.z_min) / 1.5
        alpha, beta = 2.0, 1.5
        raw = (z_arr ** alpha) * np.exp(-((z_arr / z0) ** beta))

        # Mask out-of-bounds
        mask = (z_arr >= self.z_min) & (z_arr <= self.z_max)
        res = np.where(mask, raw, 0.0)
        return float(res) if np.isscalar(z) else res

    @classmethod
    def lsst_rubin(cls) -> "SurveySpecification":
        """Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST)."""
        return cls(
            name="LSST-Year10",
            area_deg2=18000.0,
            z_min=0.2,
            z_max=3.0,
            galaxy_density_arcmin2=27.0,
            shape_noise_sigma_e=0.26,
        )

    @classmethod
    def euclid_wide(cls) -> "SurveySpecification":
        """Euclid Wide Survey specifications."""
        return cls(
            name="Euclid-Wide",
            area_deg2=15000.0,
            z_min=0.2,
            z_max=2.0,
            galaxy_density_arcmin2=30.0,
            shape_noise_sigma_e=0.25,
        )

    @classmethod
    def desi_lrg(cls) -> "SurveySpecification":
        """DESI Luminous Red Galaxies (LRG) sample."""
        return cls(
            name="DESI-LRG",
            area_deg2=14000.0,
            z_min=0.4,
            z_max=1.1,
            galaxy_density_arcmin2=0.5,
            shape_noise_sigma_e=0.0,
        )
