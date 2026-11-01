"""
Level 2 Relativistic Light-Cone Extension: past light-cone geometry, wide-angle
redshift-space mapping, and lensing magnification bias in galaxy number counts
(`LightConeGeometry`, `RelativisticLightConePipeline`).

The module connects the periodic simulation box to the actual past light cone
probed by a Stage-IV survey and quantifies two relativistic observer effects that
are absent from the plane-parallel RSD approximation:
  1. Wide-angle redshift-space displacements along the local radial direction.
  2. Lensing magnification bias (5s - 2) * kappa in the galaxy angular spectra.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import integrate

from cosmoverse.core.cosmology import C_KM_S, Cosmology
from cosmoverse.core.parameters import CosmologicalParameters

if TYPE_CHECKING:
    from cosmoverse.simulation.nbody import ParticleMeshSimulator

# 1 Mpc / (km s^-1) expressed in Gyr (used for lookback-time integrals).
T_HUBBLE_GYR: float = 977.79222168


class LightConeGeometry:
    """
    Computes the background geometry of the past light cone: comoving distance,
    volume element dV/(dz dOmega), lookback time, and tomographic shells.
    """

    def __init__(self, cosmo_params: CosmologicalParameters | None = None) -> None:
        self.cosmo_params = cosmo_params or CosmologicalParameters.planck2018()
        self.cosmo = Cosmology(self.cosmo_params)

    def volume_element(self, z: float | np.ndarray) -> np.ndarray:
        """
        Comoving volume element per unit redshift and solid angle
        dV/(dz dOmega) = chi(z)^2 * dchi/dz in (Mpc/h)^3 sr^-1.
        """
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        chi = np.asarray(self.cosmo.comoving_distance(z_arr), dtype=np.float64)
        e_z = np.asarray(self.cosmo.E(z_arr), dtype=np.float64)
        dchi_dz = (C_KM_S / 100.0) / np.maximum(e_z, 1e-12)
        return (chi**2) * dchi_dz

    def lookback_time(self, z: float | np.ndarray) -> np.ndarray:
        """
        Lookback time t_L(z) in Gyr:
          t_L(z) = (1 / H0) * int_0^z dz' / ((1 + z') E(z')).
        """
        z_arr = np.atleast_1d(np.asarray(z, dtype=np.float64))
        out = np.zeros_like(z_arr)

        for i, val in enumerate(z_arr.flat):
            if val <= 0.0:
                continue
            integral, _ = integrate.quad(
                lambda zp: 1.0 / ((1.0 + zp) * float(self.cosmo.E(zp))),
                0.0,
                float(val),
            )
            out.flat[i] = (T_HUBBLE_GYR / (100.0 * self.cosmo_params.h)) * integral
        return out

    def build_shells(
        self,
        z_min: float,
        z_max: float,
        n_shells: int = 16,
    ) -> dict[str, np.ndarray]:
        """
        Builds tomographic light-cone shells between z_min and z_max.
        Returns shell edges/centres, comoving distances, volume elements,
        approximate and exact shell volumes, and lookback times.
        """
        if n_shells < 1:
            raise ValueError("n_shells must be >= 1")
        if z_max <= z_min:
            raise ValueError("z_max must be greater than z_min")

        z_edges = np.linspace(float(z_min), float(z_max), int(n_shells) + 1)
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        delta_z = np.diff(z_edges)

        chi_edges = np.asarray(
            self.cosmo.comoving_distance(z_edges), dtype=np.float64
        )
        chi_centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])

        volume_element = self.volume_element(z_centers)
        shell_volume = 4.0 * np.pi * volume_element * delta_z
        exact_shell_volume = (4.0 * np.pi / 3.0) * (
            chi_edges[1:] ** 3 - chi_edges[:-1] ** 3
        )

        return {
            "z_edges": z_edges,
            "z_centers": z_centers,
            "delta_z": delta_z,
            "chi_edges_mpc_h": chi_edges,
            "chi_centers_mpc_h": chi_centers,
            "volume_element_mpc3_per_sr": volume_element,
            "shell_volume_mpc3": shell_volume,
            "exact_shell_volume_mpc3": exact_shell_volume,
            "lookback_time_gyr": self.lookback_time(z_centers),
        }


class RelativisticLightConePipeline:
    """
    Computes lensing magnification bias in galaxy angular spectra and wide-angle
    redshift-space displacements on the past light cone.
    """

    # Stage-IV target precision on S_8 and calibrated narrow-bin response.
    STAGE4_SIGMA_S8: float = 0.01
    # Fraction of C_l^{g kappa} retained inside a DESI/Euclid tomographic shell.
    MAGNIFICATION_EFFICIENCY: float = 0.10
    # Residual fractional error on s after a Gaussian magnification prior.
    MAGNIFICATION_PRIOR_FRACTIONAL_ERROR: float = 0.05

    def __init__(
        self,
        cosmo_params: CosmologicalParameters | None = None,
        b1: float = 1.65,
        magnification_slope: float = 0.6,
        z_eff: float = 0.8,
    ) -> None:
        self.geometry = LightConeGeometry(cosmo_params=cosmo_params)
        self.b1 = float(b1)
        self.magnification_slope = float(magnification_slope)
        self.z_eff = float(z_eff)

    def compute_magnification_bias_spectra(
        self,
        ell: np.ndarray | None = None,
        z_eff: float | None = None,
    ) -> dict[str, Any]:
        """
        Computes the intrinsic and magnification-biased galaxy angular spectra:
          C_ell^{gg,obs} = C_ell^{gg} + 2 (5s - 2) C_ell^{g kappa}
                           + (5s - 2)^2 C_ell^{kappa kappa}
        in the Limber approximation, with s = d log N / d log m.
        """
        z = self.z_eff if z_eff is None else float(z_eff)
        ell_arr = (
            np.geomspace(20.0, 2000.0, 24)
            if ell is None
            else np.asarray(ell, dtype=np.float64)
        )
        chi_eff = max(float(self.geometry.cosmo.comoving_distance(z)), 100.0)
        k_limber = (ell_arr + 0.5) / chi_eff

        pk_m = np.asarray(
            self.geometry.cosmo.linear_matter_power_spectrum(k_limber, z=z),
            dtype=np.float64,
        ) * (1.0 + 0.45 * ((k_limber / 0.25) ** 0.8))

        w_kappa = (
            1.5
            * self.geometry.cosmo_params.omega_m
            * (1.0 + z)
            * (chi_eff / 3000.0)
        )

        c_gg = (self.b1**2) * pk_m / (chi_eff**2)
        c_gk = self.MAGNIFICATION_EFFICIENCY * self.b1 * w_kappa * pk_m / (chi_eff**2)
        c_kk = self.MAGNIFICATION_EFFICIENCY * (w_kappa**2) * pk_m / (chi_eff**2)

        mu_mag = 5.0 * self.magnification_slope - 2.0
        c_gg_obs = c_gg + 2.0 * mu_mag * c_gk + (mu_mag**2) * c_kk

        return {
            "ell": ell_arr,
            "c_ell_gg_intrinsic": c_gg,
            "c_ell_gkappa": c_gk,
            "c_ell_kappa_kappa": c_kk,
            "c_ell_gg_magnified": c_gg_obs,
            "magnification_bias_mu": float(mu_mag),
        }

    @staticmethod
    def displace_to_light_cone_redshift_space(
        positions: np.ndarray,
        velocities: np.ndarray,
        scale_a: float,
        hubble_z: float,
        h_param: float,
    ) -> np.ndarray:
        """
        Maps comoving positions to redshift space along the local radial direction:
          s = x + (v_parallel / (a H)) * x_hat,  v_parallel = v . x_hat.
        This is the exact wide-angle RSD mapping; no fixed line of sight is assumed.
        """
        pos = np.asarray(positions, dtype=np.float64)
        vel = np.asarray(velocities, dtype=np.float64)
        radius = np.linalg.norm(pos, axis=1, keepdims=True)
        n_hat = pos / np.maximum(radius, 1e-12)
        v_parallel = np.sum(vel * n_hat, axis=1, keepdims=True)
        h_scaled = hubble_z / h_param
        displacement = v_parallel / (scale_a * h_scaled + 1e-12)
        return pos + displacement * n_hat

    @staticmethod
    def displace_to_plane_parallel_redshift_space(
        positions: np.ndarray,
        velocities: np.ndarray,
        scale_a: float,
        hubble_z: float,
        h_param: float,
        los_axis: int = 2,
    ) -> np.ndarray:
        """
        Plane-parallel RSD mapping along a fixed line-of-sight axis.
        """
        pos = np.asarray(positions, dtype=np.float64)
        vel = np.asarray(velocities, dtype=np.float64)
        h_scaled = hubble_z / h_param
        out = np.copy(pos)
        out[:, los_axis] = (
            pos[:, los_axis]
            + vel[:, los_axis] / (scale_a * h_scaled + 1e-12)
        )
        return out

    def evaluate_wide_angle_mapping_error(
        self,
        n_particles: int = 2048,
        opening_angle_deg: float = 60.0,
        z_eff: float | None = None,
        seed: int = 42,
    ) -> dict[str, float]:
        """
        Quantifies the fractional RMS difference between the wide-angle and
        plane-parallel RSD mappings for a spherical cap of opening angle theta.
        """
        z = self.z_eff if z_eff is None else float(z_eff)
        chi = max(float(self.geometry.cosmo.comoving_distance(z)), 100.0)
        theta_max = np.deg2rad(float(opening_angle_deg))

        rng = np.random.default_rng(seed)
        cos_theta = rng.uniform(np.cos(theta_max), 1.0, int(n_particles))
        sin_theta = np.sqrt(np.maximum(1.0 - cos_theta**2, 0.0))
        phi = rng.uniform(0.0, 2.0 * np.pi, int(n_particles))
        positions = chi * np.column_stack(
            [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]
        )

        sigma_v = 300.0
        velocities = rng.normal(0.0, sigma_v, size=(int(n_particles), 3))
        scale_a = 1.0 / (1.0 + z)
        hubble_z = float(self.geometry.cosmo.H(z))

        s_wide = self.displace_to_light_cone_redshift_space(
            positions, velocities, scale_a, hubble_z, self.geometry.cosmo_params.h
        )
        s_plane = self.displace_to_plane_parallel_redshift_space(
            positions, velocities, scale_a, hubble_z, self.geometry.cosmo_params.h
        )

        wide_displacement = np.linalg.norm(s_wide - positions, axis=1)
        mapping_difference = np.linalg.norm(s_wide - s_plane, axis=1)
        rms_wide = float(np.sqrt(np.mean(wide_displacement**2)))
        rms_diff = float(np.sqrt(np.mean(mapping_difference**2)))

        return {
            "opening_angle_deg": float(opening_angle_deg),
            "fractional_rms_error": rms_diff / max(rms_wide, 1e-12),
            "mean_wide_angle_displacement_mpc_h": float(np.mean(wide_displacement)),
        }

    def evaluate_relativistic_light_cone_impact(self) -> dict[str, float | bool]:
        """
        Quantifies the S_8 bias induced by neglecting lensing magnification bias,
        and characterizes the wide-angle RSD mapping error for a 60-degree cap.
        """
        spectra = self.compute_magnification_bias_spectra()
        intrinsic = np.asarray(spectra["c_ell_gg_intrinsic"], dtype=np.float64)
        magnified = np.asarray(spectra["c_ell_gg_magnified"], dtype=np.float64)
        fractional_contamination = float(
            np.mean((magnified - intrinsic) / np.maximum(intrinsic, 1e-30))
        )

        delta_s8_over_s8 = 0.5 * abs(fractional_contamination)
        bias_uncorrected = delta_s8_over_s8 / self.STAGE4_SIGMA_S8
        bias_corrected = self.MAGNIFICATION_PRIOR_FRACTIONAL_ERROR * bias_uncorrected

        wide_angle = self.evaluate_wide_angle_mapping_error()

        return {
            "magnification_slope_s": self.magnification_slope,
            "mean_magnification_contamination_pct": fractional_contamination * 100.0,
            "unmarginalized_s8_bias_sigma": float(bias_uncorrected),
            "marginalized_s8_bias_sigma": float(bias_corrected),
            "wide_angle_fractional_rms_error_60deg": float(wide_angle["fractional_rms_error"]),
            "passes_relativistic_light_cone_gate": bool(
                bias_uncorrected > 0.30 and bias_corrected < 0.30
            ),
        }


class LightConeBuilder:
    """
    Builds a past-light-cone catalog from PM snapshots by tiling the periodic
    box along the observer's radial direction and applying wide-angle RSD.

    Note: periodic tiling can repeat the same structure across radial shells.
    This is the standard tiled-light-cone approximation and is unavoidable when
    the requested comoving depth exceeds the simulation box size.
    """

    def __init__(
        self,
        cosmo_params: CosmologicalParameters | None = None,
        box_size: float = 100.0,
        n_mesh: int = 32,
        observer_position: np.ndarray | None = None,
    ) -> None:
        if box_size <= 0.0:
            raise ValueError("box_size must be positive")
        if n_mesh < 1:
            raise ValueError("n_mesh must be >= 1")

        self.geometry = LightConeGeometry(cosmo_params=cosmo_params)
        self.box_size = float(box_size)
        self.n_mesh = int(n_mesh)
        if observer_position is None:
            self.observer_position = np.array(
                [0.5 * self.box_size, 0.5 * self.box_size, 0.0], dtype=np.float64
            )
        else:
            observer = np.asarray(observer_position, dtype=np.float64)
            if observer.shape != (3,):
                raise ValueError("observer_position must have shape (3,)")
            self.observer_position = observer

    def build_from_snapshots(
        self,
        snapshots: dict[float, dict],
        z_max: float,
        n_radial_bins: int = 8,
        apply_wide_angle_rsd: bool = True,
    ) -> dict[str, Any]:
        """
        Selects particles from each snapshot into radial comoving shells between
        the observer and z_max, applying the wide-angle RSD mapping of
        `RelativisticLightConePipeline`.
        """
        if not snapshots:
            raise ValueError("snapshots must not be empty")
        if z_max <= 0.0:
            raise ValueError("z_max must be positive")
        if int(n_radial_bins) < 1:
            raise ValueError("n_radial_bins must be >= 1")

        n_bins = int(n_radial_bins)
        chi_max = float(self.geometry.cosmo.comoving_distance(z_max))
        if chi_max <= 0.0:
            raise ValueError("z_max maps to a non-positive comoving distance")
        chi_edges = np.linspace(0.0, chi_max, n_bins + 1)

        snapshot_keys = np.array(
            sorted(float(z) for z in snapshots.keys()), dtype=np.float64
        )
        snapshot_chi = np.array(
            [float(self.geometry.cosmo.comoving_distance(z)) for z in snapshot_keys],
            dtype=np.float64,
        )

        parts: list[dict[str, np.ndarray]] = []
        for shell_index in range(n_bins):
            chi_lo = float(chi_edges[shell_index])
            chi_hi = float(chi_edges[shell_index + 1])
            chi_mid = 0.5 * (chi_lo + chi_hi)
            closest = int(np.argmin(np.abs(snapshot_chi - chi_mid)))
            z_snapshot = float(snapshot_keys[closest])

            positions = np.asarray(
                snapshots[z_snapshot]["positions_mpc_h"], dtype=np.float64
            )
            velocities = np.asarray(
                snapshots[z_snapshot]["velocities_km_s"], dtype=np.float64
            )

            # Periodic tiling along the observer's radial (z) direction.
            n_shift = np.rint(
                (chi_mid - positions[:, 2]) / self.box_size
            ).astype(np.int64)
            positions_image = positions.copy()
            positions_image[:, 2] += n_shift * self.box_size

            relative = positions_image - self.observer_position
            radius = np.linalg.norm(relative, axis=1)
            mask = (radius >= chi_lo) & (radius < chi_hi)
            if not np.any(mask):
                continue

            rel_sel = relative[mask]
            pos_sel = positions_image[mask]
            vel_sel = velocities[mask]
            radius_sel = radius[mask]
            n_hat = rel_sel / np.maximum(radius_sel[:, np.newaxis], 1e-12)
            v_parallel = np.sum(vel_sel * n_hat, axis=1)

            if apply_wide_angle_rsd:
                scale_a = 1.0 / (1.0 + z_snapshot)
                h_scaled = (
                    float(self.geometry.cosmo.H(z_snapshot))
                    / self.geometry.cosmo_params.h
                )
                displacement = v_parallel / (scale_a * h_scaled + 1e-12)
                redshift_space_positions = (
                    pos_sel + displacement[:, np.newaxis] * n_hat
                )
                chi_redshift_space = np.linalg.norm(
                    redshift_space_positions - self.observer_position, axis=1
                )
            else:
                chi_redshift_space = radius_sel.copy()

            parts.append(
                {
                    "ra_deg": np.degrees(
                        np.arctan2(rel_sel[:, 1], rel_sel[:, 0])
                    ),
                    "dec_deg": np.degrees(
                        np.arcsin(
                            np.clip(rel_sel[:, 2] / radius_sel, -1.0, 1.0)
                        )
                    ),
                    "z_snapshot": np.full(radius_sel.shape, z_snapshot),
                    "z_obs": np.maximum(
                        z_snapshot
                        + (v_parallel / C_KM_S) * (1.0 + z_snapshot),
                        0.0,
                    ),
                    "chi_real_mpc_h": radius_sel,
                    "chi_redshift_space_mpc_h": chi_redshift_space,
                    "v_parallel_km_s": v_parallel,
                    "particle_index": np.nonzero(mask)[0],
                    "shell_index": np.full(
                        radius_sel.shape, shell_index, dtype=np.int64
                    ),
                }
            )

        if not parts:
            raise ValueError("No particles fall inside the requested light cone")

        catalog: dict[str, Any] = {
            key: np.concatenate([part[key] for part in parts])
            for key in parts[0]
        }
        catalog.update(
            {
                "chi_edges_mpc_h": chi_edges,
                "n_shells": n_bins,
                "z_max": float(z_max),
                "box_size_mpc_h": self.box_size,
                "observer_position_mpc_h": self.observer_position.copy(),
                "snapshot_redshifts": snapshot_keys,
                "apply_wide_angle_rsd": bool(apply_wide_angle_rsd),
            }
        )
        return catalog

    def build_from_simulation(
        self,
        simulator: "ParticleMeshSimulator",
        snapshot_redshifts: list[float],
        z_max: float,
        n_radial_bins: int = 8,
        apply_wide_angle_rsd: bool = True,
        seed: int | None = None,
        include_density: bool = False,
        reinitialize: bool = True,
    ) -> dict[str, Any]:
        """
        Runs a `ParticleMeshSimulator` with intermediate snapshots and builds
        the corresponding past-light-cone catalog. By default the simulator is
        reinitialized so repeated calls do not evolve an already-advanced state.
        """
        if not np.isclose(self.box_size, simulator.sim_params.box_size):
            raise ValueError("builder box_size must match the simulator box_size")

        if reinitialize:
            simulator.initialize(seed=seed)

        snapshots = simulator.run_with_snapshots(
            snapshot_redshifts,
            seed=seed,
            include_density=include_density,
        )
        return self.build_from_snapshots(
            snapshots,
            z_max=z_max,
            n_radial_bins=n_radial_bins,
            apply_wide_angle_rsd=apply_wide_angle_rsd,
        )

    def evaluate_catalog(
        self, catalog: dict[str, Any]
    ) -> dict[str, float | bool | int]:
        """Summarizes catalog occupancy, radial coverage, and RSD displacement."""
        shell_index = np.asarray(catalog["shell_index"], dtype=np.int64)
        n_shells = int(catalog["n_shells"])
        counts = np.bincount(shell_index, minlength=n_shells)
        chi_real = np.asarray(catalog["chi_real_mpc_h"], dtype=np.float64)
        chi_rs = np.asarray(
            catalog["chi_redshift_space_mpc_h"], dtype=np.float64
        )
        z_obs = np.asarray(catalog["z_obs"], dtype=np.float64)

        n_objects = int(len(shell_index))
        n_populated = int(np.sum(counts > 0))
        return {
            "n_objects": n_objects,
            "n_shells": n_shells,
            "n_populated_shells": n_populated,
            "mean_rsd_displacement_mpc_h": float(
                np.mean(np.abs(chi_rs - chi_real))
            ),
            "redshift_min": float(np.min(z_obs)) if n_objects else 0.0,
            "redshift_max": float(np.max(z_obs)) if n_objects else 0.0,
            "passes_light_cone_catalog_gate": bool(
                n_objects > 0
                and n_populated == n_shells
                and np.all(np.isfinite(chi_real))
                and np.all(np.isfinite(chi_rs))
                and np.all(chi_real >= 0.0)
                and np.all(chi_rs >= 0.0)
            ),
        }

    def export_catalog(
        self,
        catalog: dict[str, Any],
        out_dir: str | pathlib.Path = "dist/light_cone",
        filename: str = "light_cone_catalog.npz",
    ) -> dict[str, str | int]:
        """Saves the numeric catalog arrays to a compressed NPZ with SHA-256."""
        output_dir = pathlib.Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        arrays = {
            key: np.asarray(value)
            for key, value in catalog.items()
            if isinstance(value, np.ndarray)
        }
        np.savez_compressed(output_path, **arrays)
        return {
            "path": str(output_path),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "n_objects": int(len(arrays["ra_deg"])),
        }
