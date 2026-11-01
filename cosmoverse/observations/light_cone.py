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
from scipy import integrate, ndimage, spatial, special

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
        density_smoothing_cells: float = 1.5,
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

            delta_matter = np.zeros(int(np.count_nonzero(mask)), dtype=np.float64)
            if "density" in snapshots[z_snapshot]:
                density = np.asarray(snapshots[z_snapshot]["density"])
                if density_smoothing_cells > 0.0:
                    density = ndimage.gaussian_filter(
                        density,
                        sigma=float(density_smoothing_cells),
                        mode="wrap",
                    )
                dx = self.box_size / density.shape[0]
                coordinates = np.vstack(
                    [
                        positions[mask, 0] / dx,
                        positions[mask, 1] / dx,
                        positions[mask, 2] / dx,
                    ]
                )
                delta_matter = ndimage.map_coordinates(
                    density, coordinates, order=1, mode="wrap"
                )

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
                    "delta_matter": delta_matter,
                    "x_mpc_h": rel_sel[:, 0],
                    "y_mpc_h": rel_sel[:, 1],
                    "z_mpc_h": rel_sel[:, 2],
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
        include_density: bool = True,
        reinitialize: bool = True,
        density_smoothing_cells: float = 1.5,
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
            density_smoothing_cells=density_smoothing_cells,
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


class SurveyMask:
    """
    Spherical-rectangle survey mask in RA/Dec and observed redshift.
    """

    def __init__(
        self,
        ra_min: float = 0.0,
        ra_max: float = 360.0,
        dec_min: float = 70.0,
        dec_max: float = 90.0,
        z_min: float = 0.0,
        z_max: float = 0.2,
    ) -> None:
        if not 0.0 <= ra_min < ra_max <= 360.0:
            raise ValueError("RA bounds must satisfy 0 <= ra_min < ra_max <= 360")
        if not -90.0 <= dec_min < dec_max <= 90.0:
            raise ValueError("Dec bounds must satisfy -90 <= dec_min < dec_max <= 90")
        if not 0.0 <= z_min < z_max:
            raise ValueError("redshift bounds must satisfy 0 <= z_min < z_max")

        self.ra_min = float(ra_min)
        self.ra_max = float(ra_max)
        self.dec_min = float(dec_min)
        self.dec_max = float(dec_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    @property
    def area_deg2(self) -> float:
        """Area of the spherical rectangle in square degrees."""
        ra_span = np.deg2rad(self.ra_max - self.ra_min)
        dec_span = np.sin(np.deg2rad(self.dec_max)) - np.sin(
            np.deg2rad(self.dec_min)
        )
        return float(ra_span * dec_span * (180.0 / np.pi) ** 2)

    @staticmethod
    def subset(catalog: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
        """Applies a boolean mask to per-object arrays, preserving metadata."""
        n_objects = int(np.count_nonzero(mask))
        result: dict[str, Any] = {}
        for key, value in catalog.items():
            if (
                isinstance(value, np.ndarray)
                and value.ndim >= 1
                and value.shape[0] == mask.shape[0]
            ):
                result[key] = value[mask]
            else:
                result[key] = value
        if "ra_deg" in result:
            result["n_objects"] = n_objects
        return result

    def apply(self, catalog: dict[str, Any]) -> dict[str, Any]:
        """Selects objects inside the mask footprint and redshift range."""
        ra = np.mod(np.asarray(catalog["ra_deg"], dtype=np.float64), 360.0)
        dec = np.asarray(catalog["dec_deg"], dtype=np.float64)
        z_obs = np.asarray(catalog["z_obs"], dtype=np.float64)
        mask = (
            (ra >= self.ra_min)
            & (ra <= self.ra_max)
            & (dec >= self.dec_min)
            & (dec <= self.dec_max)
            & (z_obs >= self.z_min)
            & (z_obs <= self.z_max)
        )
        return self.subset(catalog, mask)


class GalaxyBiasSampler:
    """
    Poisson-samples a galaxy catalog from a matter light-cone catalog using
    a local linear bias model p_g ∝ 1 + b1 * delta_m.
    """

    def __init__(
        self,
        b1: float = 1.5,
        sampling_rate: float = 0.5,
        seed: int = 42,
        density_proxy: str = "knn",
        knn_radius_mpc_h: float = 10.0,
    ) -> None:
        if b1 < 0.0:
            raise ValueError("b1 must be non-negative")
        if not 0.0 < sampling_rate <= 1.0:
            raise ValueError("sampling_rate must lie in (0, 1]")
        if density_proxy not in {"knn", "snapshot"}:
            raise ValueError("density_proxy must be 'knn' or 'snapshot'")
        self.b1 = float(b1)
        self.sampling_rate = float(sampling_rate)
        self.seed = int(seed)
        self.density_proxy = density_proxy
        self.knn_radius_mpc_h = float(knn_radius_mpc_h)

    def sample(self, catalog: dict[str, Any]) -> dict[str, Any]:
        """Returns a biased galaxy subsample of the input matter catalog."""
        n_objects = len(catalog["ra_deg"])
        if self.density_proxy == "knn":
            required = ("x_mpc_h", "y_mpc_h", "z_mpc_h")
            if not all(key in catalog for key in required):
                raise ValueError(
                    "kNN density proxy requires x_mpc_h, y_mpc_h and z_mpc_h"
                )
            positions = np.column_stack(
                [
                    np.asarray(catalog["x_mpc_h"], dtype=np.float64),
                    np.asarray(catalog["y_mpc_h"], dtype=np.float64),
                    np.asarray(catalog["z_mpc_h"], dtype=np.float64),
                ]
            )
            if positions.shape != (n_objects, 3):
                raise ValueError("Cartesian positions have an invalid shape")
            tree = spatial.cKDTree(positions)
            counts = tree.query_ball_point(
                positions, self.knn_radius_mpc_h, return_length=True
            ).astype(np.float64) - 1.0
            delta = (counts - float(np.mean(counts))) / max(
                float(np.std(counts)), 1e-12
            )
        else:
            if "delta_matter" not in catalog:
                raise ValueError("catalog must contain the delta_matter field")
            delta = np.asarray(catalog["delta_matter"], dtype=np.float64)
            if delta.shape != (n_objects,):
                raise ValueError("delta_matter must be a 1D per-object array")
            delta = delta - float(np.mean(delta))
            if float(np.std(delta)) < 1e-12:
                raise ValueError(
                    "delta_matter has zero variance; cannot apply bias"
                )
            delta = delta / float(np.std(delta))

        weight = np.clip(1.0 + self.b1 * delta, 1e-6, None)
        probability = self.sampling_rate * weight / float(np.mean(weight))
        probability = np.clip(probability, 0.0, 1.0)

        rng = np.random.default_rng(self.seed)
        selected = rng.random(delta.shape) < probability
        result = SurveyMask.subset(catalog, selected)
        result["b1_input"] = self.b1
        result["sampling_rate_input"] = self.sampling_rate
        result["selection_probability"] = probability[selected]
        result["selection_mask"] = selected
        return result


class AngularClusteringMeasurer:
    """
    Measures the angular correlation function w(theta) with the Davis-Peebles
    estimator DD/DR - 1 using a random catalog matched to the survey footprint.
    """

    def __init__(
        self,
        cosmo_params: CosmologicalParameters | None = None,
        theta_min_deg: float = 1.0,
        theta_max_deg: float = 20.0,
        n_theta_bins: int = 12,
        n_random_multiplier: float = 2.0,
        seed: int = 42,
    ) -> None:
        if theta_min_deg <= 0.0 or theta_max_deg <= theta_min_deg:
            raise ValueError("theta bounds must satisfy 0 < theta_min < theta_max")
        if n_theta_bins < 2:
            raise ValueError("n_theta_bins must be >= 2")
        self.geometry = LightConeGeometry(cosmo_params=cosmo_params)
        self.theta_min_deg = float(theta_min_deg)
        self.theta_max_deg = float(theta_max_deg)
        self.n_theta_bins = int(n_theta_bins)
        self.n_random_multiplier = float(n_random_multiplier)
        self.seed = int(seed)

    @staticmethod
    def unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
        ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
        dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
        return np.column_stack(
            [
                np.cos(dec) * np.cos(ra),
                np.cos(dec) * np.sin(ra),
                np.sin(dec),
            ]
        )

    def generate_random_catalog(
        self,
        catalog: dict[str, Any],
        box_size: float,
        theta_cap_deg: float = 20.0,
    ) -> dict[str, np.ndarray]:
        """
        Generates a random catalog with the same redshift distribution and an
        approximate z-dependent angular cap matching the periodic box footprint.
        """
        z_obs = np.asarray(catalog["z_obs"], dtype=np.float64)
        n_random = int(self.n_random_multiplier * len(z_obs))
        rng = np.random.default_rng(self.seed)

        z_random = rng.choice(z_obs, size=n_random, replace=True)
        chi = np.asarray(
            self.geometry.cosmo.comoving_distance(z_random), dtype=np.float64
        )
        transverse_radius = float(box_size) / np.sqrt(2.0)
        theta_cap = np.deg2rad(float(theta_cap_deg))
        theta_max = np.minimum(
            theta_cap,
            np.arctan(transverse_radius / np.maximum(chi, 1e-6)),
        )
        cos_theta = rng.uniform(np.cos(theta_max), 1.0, n_random)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        phi = rng.uniform(0.0, 2.0 * np.pi, n_random)

        return {
            "ra_deg": np.degrees(phi),
            "dec_deg": 90.0 - np.degrees(theta),
            "z_obs": z_random,
        }

    def measure_w_theta(
        self,
        data: dict[str, Any],
        random: dict[str, Any],
    ) -> dict[str, np.ndarray | float | int]:
        """Computes w(theta) in angular bins via the Davis-Peebles estimator."""
        theta_edges = np.linspace(
            np.deg2rad(self.theta_min_deg),
            np.deg2rad(self.theta_max_deg),
            self.n_theta_bins + 1,
        )
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        chord_edges = 2.0 * np.sin(theta_edges / 2.0)

        data_vectors = self.unit_vectors(data["ra_deg"], data["dec_deg"])
        random_vectors = self.unit_vectors(random["ra_deg"], random["dec_deg"])
        data_tree = spatial.cKDTree(data_vectors)
        random_tree = spatial.cKDTree(random_vectors)

        dd_cumulative = data_tree.count_neighbors(data_tree, chord_edges)
        dr_cumulative = data_tree.count_neighbors(random_tree, chord_edges)
        dd = np.diff(dd_cumulative) / 2.0
        dr = np.diff(dr_cumulative)

        n_data = int(len(data_vectors))
        n_random = int(len(random_vectors))
        # DD is stored as unique pairs (i < j), while DR counts ordered
        # data-random pairs, hence the factor of 2 in the normalization.
        w_theta = (
            2.0
            * (n_random / max(n_data, 1))
            * dd
            / np.maximum(dr, 1e-12)
            - 1.0
        )

        return {
            "theta_deg": np.degrees(theta_centers),
            "w_theta": w_theta,
            "dd": dd,
            "dr": dr,
            "n_data": n_data,
            "n_random": n_random,
        }

    def measure_cross_w_theta(
        self,
        data1: dict[str, Any],
        data2: dict[str, Any],
        random: dict[str, Any],
    ) -> dict[str, np.ndarray | float | int]:
        """
        Cross-correlation between two data catalogs using the Davis-Peebles
        estimator w12 = (N_r / N_2) * D1D2 / D1R - 1.
        """
        theta_edges = np.linspace(
            np.deg2rad(self.theta_min_deg),
            np.deg2rad(self.theta_max_deg),
            self.n_theta_bins + 1,
        )
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        chord_edges = 2.0 * np.sin(theta_edges / 2.0)

        tree1 = spatial.cKDTree(
            self.unit_vectors(data1["ra_deg"], data1["dec_deg"])
        )
        tree2 = spatial.cKDTree(
            self.unit_vectors(data2["ra_deg"], data2["dec_deg"])
        )
        random_tree = spatial.cKDTree(
            self.unit_vectors(random["ra_deg"], random["dec_deg"])
        )

        dd = np.diff(tree1.count_neighbors(tree2, chord_edges))
        dr1 = np.diff(tree1.count_neighbors(random_tree, chord_edges))

        n1 = len(data1["ra_deg"])
        n2 = len(data2["ra_deg"])
        n_random = len(random["ra_deg"])
        w_cross = (
            (n_random / max(n2, 1))
            * dd
            / np.maximum(dr1, 1e-12)
            - 1.0
        )
        return {
            "theta_deg": np.degrees(theta_centers),
            "w_theta": w_cross,
            "dd": dd,
            "dr": dr1,
            "n_data1": n1,
            "n_data2": n2,
            "n_random": n_random,
        }


class LightConeSurveyPipeline:
    """
    Converts a particle light-cone catalog into a masked, biased galaxy catalog
    and validates the measured w(theta) against the analytic Limber prediction.
    """

    def __init__(
        self,
        cosmo_params: CosmologicalParameters | None = None,
        b1: float = 1.5,
        sampling_rate: float = 0.5,
        mask: SurveyMask | None = None,
        measurer: AngularClusteringMeasurer | None = None,
        seed: int = 42,
    ) -> None:
        self.geometry = LightConeGeometry(cosmo_params=cosmo_params)
        self.b1 = float(b1)
        self.sampling_rate = float(sampling_rate)
        self.seed = int(seed)
        self.mask = mask or SurveyMask()
        self.sampler = GalaxyBiasSampler(
            b1=self.b1, sampling_rate=self.sampling_rate, seed=seed
        )
        self.measurer = measurer or AngularClusteringMeasurer(
            cosmo_params=cosmo_params, seed=seed
        )

    def analytic_w_theta(
        self,
        theta_deg: np.ndarray,
        z_eff: float,
        b1: float | None = None,
        ell_max: float = 5000.0,
        n_ell: int = 512,
    ) -> np.ndarray:
        """
        Computes the flat-sky Hankel transform of the Limber C_ell^{gg}:
          w(theta) = int ell d ell / (2 pi) C_ell J0(ell theta).
        """
        bias = self.b1 if b1 is None else float(b1)
        chi = max(float(self.geometry.cosmo.comoving_distance(z_eff)), 100.0)
        ell = np.linspace(1.0, float(ell_max), int(n_ell))
        k = (ell + 0.5) / chi
        pk_m = np.asarray(
            self.geometry.cosmo.linear_matter_power_spectrum(k, z=z_eff),
            dtype=np.float64,
        ) * (1.0 + 0.45 * ((k / 0.25) ** 0.8))
        c_ell = (bias**2) * pk_m / (chi**2)

        theta_rad = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
        w_theta = np.array(
            [
                integrate.simpson(
                    ell * c_ell * special.j0(ell * theta) / (2.0 * np.pi),
                    x=ell,
                )
                for theta in theta_rad
            ]
        )
        return w_theta

    def evaluate(
        self,
        catalog: dict[str, Any],
        box_size: float,
        theta_cap_deg: float = 20.0,
    ) -> dict[str, Any]:
        """Builds the galaxy catalog and validates w(theta) and the effective bias."""
        matter = self.mask.apply(catalog)
        if len(matter["ra_deg"]) == 0:
            raise ValueError("survey mask selects no objects")

        # Split the matter catalog into two disjoint halves: galaxies are
        # sampled from half A and cross-correlated with half B, eliminating
        # self-pair shot noise without using a density-selected complement.
        rng = np.random.default_rng(self.seed)
        split = rng.random(len(matter["ra_deg"])) < 0.5
        matter_a = SurveyMask.subset(matter, split)
        matter_b = SurveyMask.subset(matter, ~split)

        galaxy = self.sampler.sample(matter_a)
        if len(galaxy["ra_deg"]) == 0:
            raise ValueError("galaxy sampler selects no objects")
        if len(matter_b["ra_deg"]) == 0:
            raise ValueError("matter split leaves no cross-correlation sample")

        random = self.measurer.generate_random_catalog(
            matter, box_size=box_size, theta_cap_deg=theta_cap_deg
        )
        w_mm = self.measurer.measure_w_theta(matter, random)
        w_gg = self.measurer.measure_w_theta(galaxy, random)
        w_gm = self.measurer.measure_cross_w_theta(galaxy, matter_b, random)

        theta = np.asarray(w_gm["theta_deg"], dtype=np.float64)
        w_gg_values = np.asarray(w_gg["w_theta"], dtype=np.float64)
        w_mm_values = np.asarray(w_mm["w_theta"], dtype=np.float64)
        w_gm_values = np.asarray(w_gm["w_theta"], dtype=np.float64)

        # Least-squares slope through the origin, w_gm = b_eff * w_mm.
        # This is more stable than averaging per-bin ratios when w_mm is noisy.
        fit_mask = (w_mm_values > 0.0) & np.isfinite(w_gm_values)
        if np.count_nonzero(fit_mask) >= 2:
            b_eff = float(
                np.sum(w_gm_values[fit_mask] * w_mm_values[fit_mask])
                / np.sum(w_mm_values[fit_mask] ** 2)
            )
        else:
            b_eff = 0.0

        z_eff = float(np.median(np.asarray(galaxy["z_obs"], dtype=np.float64)))
        analytic = self.analytic_w_theta(theta, z_eff=z_eff)

        valid_cross = np.isfinite(w_gm_values) & np.isfinite(w_mm_values)
        if (
            np.count_nonzero(valid_cross) >= 2
            and np.std(w_gm_values[valid_cross]) > 0.0
            and np.std(w_mm_values[valid_cross]) > 0.0
        ):
            pearson_gm_mm = float(
                np.corrcoef(
                    w_gm_values[valid_cross], w_mm_values[valid_cross]
                )[0, 1]
            )
        else:
            pearson_gm_mm = 0.0

        valid_analytic = np.isfinite(w_gm_values) & np.isfinite(analytic)
        if (
            np.count_nonzero(valid_analytic) >= 2
            and np.std(w_gm_values[valid_analytic]) > 0.0
            and np.std(analytic[valid_analytic]) > 0.0
        ):
            pearson_analytic = float(
                np.corrcoef(
                    w_gm_values[valid_analytic], analytic[valid_analytic]
                )[0, 1]
            )
        else:
            pearson_analytic = 0.0

        bias_ratio = b_eff / self.b1 if self.b1 > 0.0 else 0.0
        return {
            "n_matter": int(len(matter["ra_deg"])),
            "n_galaxies": int(len(galaxy["ra_deg"])),
            "mask_area_deg2": self.mask.area_deg2,
            "b1_input": self.b1,
            "b1_effective": b_eff,
            "bias_ratio": float(bias_ratio),
            "theta_deg": theta,
            "w_theta_gg": w_gg_values,
            "w_theta_mm": w_mm_values,
            "w_theta_gm": w_gm_values,
            "w_theta_analytic": analytic,
            "pearson_r_gm_mm": pearson_gm_mm,
            "pearson_r_analytic": pearson_analytic,
            "passes_light_cone_survey_gate": bool(
                len(galaxy["ra_deg"]) > 100
                and b_eff > 1.0
                and pearson_gm_mm > 0.8
            ),
        }
