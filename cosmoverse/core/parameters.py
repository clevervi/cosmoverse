"""
Cosmological, baryonic (BCEmu), observational nuisance, and numerical simulation parameters.
Uses Pydantic for validation, serialization, and strict 3-level methodological separation:
  1. Level 1 (Universe Physics): CosmologicalParameters, BaryonParameters, SimulationParameters
  2. Level 2 (Observational Process): ObservationalNuisanceParameters
  3. Level 3 (Statistical Inference): Prior bounds & calibration tolerances
"""

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, model_validator


class CosmologicalParameters(BaseModel):
    """
    Standard Lambda-CDM and w0-wa-CDM cosmological parameters (Level 1: Universe Physics).

    Attributes:
        omega_m: Total matter density parameter (Omega_m = Omega_cdm + Omega_b + Omega_nu)
        omega_b: Baryon density parameter (Omega_b)
        h: Dimensionless Hubble parameter H0 / (100 km/s/Mpc)
        sigma8: Amplitude of matter fluctuations on 8 h^-1 Mpc scale at z=0
        ns: Scalar spectral index of primordial perturbations
        w0: Dark energy equation of state parameter today (w(a) = w0 + wa*(1-a))
        wa: Dark energy evolution parameter
        m_nu: Sum of neutrino masses in eV
    """
    omega_m: float = Field(default=0.315, ge=0.01, le=1.0, description="Total matter density fraction")
    omega_b: float = Field(default=0.049, ge=0.005, le=0.2, description="Baryon density fraction")
    h: float = Field(default=0.674, ge=0.4, le=1.0, description="Hubble parameter h = H0/100")
    sigma8: float = Field(default=0.811, ge=0.2, le=1.5, description="Root-mean-square amplitude of fluctuations at 8 Mpc/h")
    ns: float = Field(default=0.965, ge=0.7, le=1.3, description="Primordial scalar spectral index")
    w0: float = Field(default=-1.0, ge=-2.5, le=-0.3, description="Dark energy equation of state at z=0")
    wa: float = Field(default=0.0, ge=-2.0, le=2.0, description="Dark energy evolution parameter")
    m_nu: float = Field(default=0.0, ge=0.0, le=2.0, description="Sum of neutrino masses in eV")
    omega_lambda: Optional[float] = Field(default=None, description="Dark energy density fraction (defaults to 1 - omega_m)")

    @model_validator(mode="after")
    def compute_defaults_and_validate(self) -> "CosmologicalParameters":
        if self.omega_b > self.omega_m:
            raise ValueError(f"Baryon density omega_b ({self.omega_b}) cannot exceed total matter omega_m ({self.omega_m})")
        if self.omega_lambda is None:
            object.__setattr__(self, "omega_lambda", 1.0 - self.omega_m)
        return self

    @property
    def omega_cdm(self) -> float:
        """Cold dark matter density fraction."""
        return self.omega_m - self.omega_b

    @property
    def H0(self) -> float:
        """Hubble constant in km / s / Mpc."""
        return 100.0 * self.h

    @property
    def S8(self) -> float:
        """S8 clustering parameter (sigma8 * sqrt(omega_m / 0.3))."""
        return float(self.sigma8 * np.sqrt(self.omega_m / 0.3))

    @classmethod
    def planck2018(cls) -> "CosmologicalParameters":
        """Returns baseline Planck 2018 TT,TE,EE+lowE+lensing best-fit parameters."""
        return cls(
            omega_m=0.3153,
            omega_b=0.0493,
            h=0.6736,
            sigma8=0.8111,
            ns=0.9649,
            w0=-1.0,
            wa=0.0,
        )

    @classmethod
    def sample_prior(
        cls,
        n_samples: int = 1,
        prior_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None,
    ) -> list["CosmologicalParameters"]:
        """
        Draw cosmological parameter samples from uniform priors (standard for SBI / CAMELS).
        """
        rng = np.random.default_rng(seed)
        defaults: Dict[str, Tuple[float, float]] = {
            "omega_m": (0.1, 0.5),
            "sigma8": (0.6, 1.0),
            "h": (0.6, 0.8),
            "ns": (0.9, 1.05),
            "w0": (-1.5, -0.5),
        }
        bounds = prior_ranges or defaults
        samples = []
        for _ in range(n_samples):
            kwargs: Dict[str, Any] = {}
            for param, (low, high) in bounds.items():
                kwargs[param] = float(rng.uniform(low, high))
            if "omega_b" not in kwargs:
                kwargs["omega_b"] = 0.049
            samples.append(cls(**kwargs))
        return samples


class BaryonParameters(BaseModel):
    """
    Baryonic physics parameters (Level 1: Universe Physics).
    Primary v1 model: BCEmu (Giri & Schneider 2021) supporting 1, 3, and 7-parameter
    baryonification regimes (BCEmu1, BCEmu3, BCEmu7), plus CAMELS feedback parameters.
    Note: Only ONE baryonic emulator (BCEmu) is active in v1 for error attribution;
    BACCOemu is reserved for v2 cross-validation.

    Attributes:
        mode: Active BCEmu complexity mode ('BCEmu1', 'BCEmu3', 'BCEmu7')
        log10_Mcrit: Critical halo mass where half the gas is ejected (log10 M_c in Msun/h)
        mu_baryon: Mass-dependence slope of gas ejection (BCEmu3/7)
        theta_ej: Maximum gas ejection radius in units of r_200 (BCEmu3/7)
        gamma_baryon: Star formation efficiency slope (BCEmu7)
        delta_baryon: Outer gas profile slope (BCEmu7)
        eta_b: Total stellar mass fraction parameter (BCEmu7)
        eta_delta: Central galaxy stellar fraction parameter (BCEmu7)
        A_AGN: CAMELS-compatible AGN feedback energy scale
        A_SN: CAMELS-compatible Supernova feedback scale
    """
    mode: Literal["BCEmu1", "BCEmu3", "BCEmu7"] = Field(default="BCEmu3", description="BCEmu parameter regime")
    log10_Mcrit: float = Field(default=13.5, ge=11.0, le=15.5, description="log10(M_c [Msun/h]) critical gas mass")
    mu_baryon: float = Field(default=0.4, ge=0.0, le=2.0, description="BCEmu mass slope mu")
    theta_ej: float = Field(default=4.0, ge=2.0, le=8.0, description="Ejection radius ratio r_ej / r_200")
    gamma_baryon: float = Field(default=2.5, ge=1.0, le=4.0, description="Stellar fraction slope gamma")
    delta_baryon: float = Field(default=7.0, ge=3.0, le=11.0, description="Outer gas profile slope delta")
    eta_b: float = Field(default=0.2, ge=0.05, le=1.5, description="Total stellar mass scaling eta")
    eta_delta: float = Field(default=0.25, ge=0.05, le=0.6, description="Central galaxy stellar scaling eta_delta")
    A_AGN: float = Field(default=1.0, ge=0.0, le=10.0, description="AGN feedback energy scale (CAMELS)")
    A_SN: float = Field(default=1.0, ge=0.0, le=10.0, description="Supernova feedback scale (CAMELS)")


class ObservationalNuisanceParameters(BaseModel):
    """
    Observational process parameters (Level 2: Survey & Systematics).
    Explicitly separated from physical cosmology and baryons to avoid mixing levels.

    Attributes:
        b1: Linear galaxy bias (delta_g = b1 * delta_m)
        b2: Quadratic non-linear galaxy bias
        sigma_fog: Finger-of-God velocity dispersion in Mpc/h
        sigma_z: Redshift measurement uncertainty sigma_z / (1+z)
        shot_noise_scale: Multiplicative residual shot-noise nuisance parameter A_shot
        shear_m_bias: Multiplicative weak lensing shear calibration bias m
    """
    b1: float = Field(default=1.5, ge=0.2, le=5.0, description="Linear galaxy bias")
    b2: float = Field(default=0.0, ge=-2.0, le=2.0, description="Quadratic galaxy bias")
    sigma_fog: float = Field(default=3.5, ge=0.0, le=15.0, description="FoG velocity dispersion scale [Mpc/h]")
    sigma_z: float = Field(default=0.001, ge=0.0, le=0.1, description="Redshift error sigma_0")
    shot_noise_scale: float = Field(default=1.0, ge=0.0, le=3.0, description="Shot noise amplitude scaling")
    shear_m_bias: float = Field(default=0.0, ge=-0.1, le=0.1, description="Weak lensing multiplicative shear bias m")


class SimulationParameters(BaseModel):
    """
    N-body / Particle-Mesh numerical simulation parameters.

    Attributes:
        box_size: Comoving box size in Mpc/h
        n_mesh: Grid size per dimension (N_mesh^3 cells)
        n_particles: Number of simulation particles per dimension (N_part^3)
        z_initial: Initial redshift for perturbation setup
        z_final: Final redshift for simulation snapshot
        n_steps: Number of integration time steps (Kick-Drift-Kick)
        seed: Random seed for initial Gaussian random field
        family_id: Simulation suite identifier (e.g. 'CAMELS-DMO', 'CAMELS-IllustrisTNG')
        cosmo_id: Unique cosmology identifier for strict train/test partitioning
    """
    box_size: float = Field(default=100.0, gt=0.0, description="Box size in Mpc/h")
    n_mesh: int = Field(default=64, ge=16, le=1024, description="Mesh grid resolution per axis")
    n_particles: Optional[int] = Field(default=None, description="Particle count per axis; defaults to n_mesh")
    z_initial: float = Field(default=50.0, ge=1.0, le=200.0, description="Initial redshift")
    z_final: float = Field(default=0.0, ge=0.0, le=10.0, description="Final redshift")
    n_steps: int = Field(default=10, ge=1, le=500, description="Time integration steps")
    seed: int = Field(default=42, description="Initial conditions random seed (ic_seed)")
    family_id: str = Field(default="PM-Nbody", description="Simulation family ID for transfer tests")
    cosmo_id: str = Field(default="fiducial-000", description="Cosmology ID for leakage-free partitioning")

    @model_validator(mode="after")
    def set_particles(self) -> "SimulationParameters":
        if self.n_particles is None:
            object.__setattr__(self, "n_particles", self.n_mesh)
        return self

    @property
    def cell_size(self) -> float:
        """Physical/comoving cell size dx = L / N_mesh in Mpc/h."""
        return self.box_size / self.n_mesh

    @property
    def nyquist_k(self) -> float:
        """Nyquist wavenumber k_Ny = pi * N_mesh / L in h/Mpc."""
        return float(np.pi * self.n_mesh / self.box_size)
