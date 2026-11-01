"""
Cosmological background calculations, distances, linear growth, and power spectra.
"""

from typing import Optional, Union

import numpy as np
from scipy import integrate

from cosmoverse.core.parameters import CosmologicalParameters

# Speed of light in km/s
C_KM_S: float = 299792.458


class Cosmology:
    """
    Cosmological model providing background expansion, distances, growth factors,
    and linear matter power spectra for flat Lambda-CDM and w0-wa-CDM universes.
    """

    def __init__(self, params: Optional[CosmologicalParameters] = None):
        self.params = params or CosmologicalParameters()
        self._sigma8_norm: Optional[float] = None

    @property
    def H0(self) -> float:
        """Hubble constant in km/s/Mpc."""
        return self.params.H0

    def E(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Dimensionless Hubble expansion rate E(z) = H(z) / H0.
        Supports Chevallier-Polarski-Linder (w0, wa) parametrization.
        """
        z_arr = np.asarray(z, dtype=np.float64)
        a = 1.0 / (1.0 + z_arr)
        om = self.params.omega_m
        ol = self.params.omega_lambda if self.params.omega_lambda is not None else (1.0 - om)

        # Dark energy equation of state: rho_de(a) / rho_de(1) = a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
        w0 = self.params.w0
        wa = self.params.wa
        de_factor = (a ** (-3.0 * (1.0 + w0 + wa))) * np.exp(-3.0 * wa * (1.0 - a))

        e2 = om * (1.0 + z_arr) ** 3 + ol * de_factor
        val = np.sqrt(np.maximum(e2, 1e-12))
        return float(val) if np.isscalar(z) else val

    def H(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Hubble parameter at redshift z in km/s/Mpc."""
        return self.H0 * self.E(z)

    def omega_m_z(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Matter density fraction at redshift z: Omega_m(z) = Omega_m * (1+z)^3 / E(z)^2."""
        z_arr = np.asarray(z, dtype=np.float64)
        return (self.params.omega_m * (1.0 + z_arr) ** 3) / (self.E(z_arr) ** 2)

    def comoving_distance(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Comoving distance chi(z) in Mpc/h.
        chi(z) = (c / 100) * int_0^z dz' / E(z')
        """
        c_over_100 = C_KM_S / 100.0

        if np.isscalar(z):
            if float(z) <= 0.0:
                return 0.0
            res, _ = integrate.quad(lambda zp: 1.0 / self.E(zp), 0.0, float(z))
            return c_over_100 * res

        z_arr = np.asarray(z, dtype=np.float64)
        result = np.zeros_like(z_arr)
        for i, val in enumerate(z_arr.flat):
            if val > 0.0:
                res, _ = integrate.quad(lambda zp: 1.0 / self.E(zp), 0.0, val)
                result.flat[i] = c_over_100 * res
        return result

    def angular_diameter_distance(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Angular diameter distance D_A(z) in Mpc/h."""
        z_arr = np.asarray(z, dtype=np.float64)
        chi = self.comoving_distance(z_arr)
        return chi / (1.0 + z_arr)

    def luminosity_distance(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Luminosity distance D_L(z) in Mpc/h."""
        z_arr = np.asarray(z, dtype=np.float64)
        chi = self.comoving_distance(z_arr)
        return (1.0 + z_arr) * chi

    def growth_factor(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Normalized linear growth factor D1(z), normalized such that D1(0) = 1.0.
        Uses the Heath (1977) integral:
        D1(a) proportional to H(a) * int_0^a da' / (a' * H(a'))^3
        """
        def unnorm_growth(scale_a: float) -> float:
            if scale_a <= 1e-7:
                return scale_a
            redshift = 1.0 / scale_a - 1.0
            e_val = float(self.E(redshift))
            # Integral over a'
            def integrand(ap):
                return 1.0 / ((ap * float(self.E(1.0 / ap - 1.0))) ** 3 + 1e-12)
            res, _ = integrate.quad(integrand, 1e-6, scale_a, limit=100)
            return e_val * res

        d0 = unnorm_growth(1.0)

        if np.isscalar(z):
            a = 1.0 / (1.0 + float(z))
            return float(unnorm_growth(a) / d0)

        z_arr = np.asarray(z, dtype=np.float64)
        res_arr = np.zeros_like(z_arr)
        for i, val in enumerate(z_arr.flat):
            a = 1.0 / (1.0 + val)
            res_arr.flat[i] = unnorm_growth(a) / d0
        return res_arr

    def growth_rate(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Linear growth rate f(z) = d ln D / d ln a approx Omega_m(z)^gamma.
        For standard Lambda-CDM, gamma approx 0.55 + 0.05 * (1 + w0).
        """
        gamma = 0.55 + 0.05 * (1.0 + self.params.w0)
        return self.omega_m_z(z) ** gamma

    def transfer_function_eh98(
        self, k: np.ndarray, with_baryon_wiggles: bool = True
    ) -> np.ndarray:
        """
        Eisenstein & Hu (1998) transfer function T(k).
        Wavenumber k is expected in h / Mpc.

        Reference:
            Eisenstein, D. J., & Hu, W. (1998). "Baryonic Features in the Matter Transfer Function".
            Astrophysical Journal, 496, 605.
        """
        k = np.asarray(k, dtype=np.float64)
        om = self.params.omega_m
        ob = self.params.omega_b
        h = self.params.h
        theta_cmb = 2.7255 / 2.7  # CMB temperature ratio

        om_h2 = om * (h ** 2)
        ob_h2 = ob * (h ** 2)
        fc = (om - ob) / om
        fb = ob / om

        # Sound horizon and equality scales
        z_eq = 2.50e4 * om_h2 * (theta_cmb ** -4)
        k_eq = 0.0746 * om_h2 * (theta_cmb ** -2)

        # Drag epoch redshift (Hu & Sugiyama 1996)
        b1 = 0.313 * (om_h2 ** -0.419) * (1.0 + 0.607 * (om_h2 ** 0.674))
        b2 = 0.238 * (om_h2 ** 0.223)
        z_drag = 1291.0 * ((om_h2 ** 0.251) / (1.0 + 0.659 * (om_h2 ** 0.828))) * (
            1.0 + b1 * (ob_h2 ** b2)
        )

        r_drag = 31.5 * ob_h2 * (theta_cmb ** -4) * (1000.0 / z_drag)
        r_eq = 31.5 * ob_h2 * (theta_cmb ** -4) * (1000.0 / z_eq)
        sound_horizon = (2.0 / (3.0 * k_eq)) * np.sqrt(6.0 / r_eq) * np.log(
            (np.sqrt(1.0 + r_drag) + np.sqrt(r_drag + r_eq)) / (1.0 + np.sqrt(r_eq))
        )

        # Silk damping scale
        k_silk = 1.6 * (ob_h2 ** 0.52) * (om_h2 ** 0.73) * (1.0 + (10.6 * om_h2) ** -0.22)

        # CDM transfer function
        q = k / (13.41 * k_eq)
        a1 = (46.9 * om_h2) ** 0.670 * (1.0 + (32.1 * om_h2) ** -0.532)
        a2 = (12.0 * om_h2) ** 0.424 * (1.0 + (45.0 * om_h2) ** -0.582)
        alpha_c = a1 ** (-fb) * a2 ** (-(fb ** 3))

        b1_c = 0.944 / (1.0 + (458.0 * om_h2) ** -0.708)
        b2_c = (0.198 * om_h2) ** 0.02
        beta_c = 1.0 / (1.0 + b1_c * ((fc ** b2_c) - 1.0))

        def f_func(k_val: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + (k_val * sound_horizon / 5.4) ** 4)

        c_val = 14.2 / alpha_c + 386.0 / (1.0 + 69.9 * (q ** 1.08))
        t_c = f_func(k) * (np.log(np.e + 1.8 * beta_c * q) / (
            np.log(np.e + 1.8 * beta_c * q) + c_val * (q ** 2)
        )) + (1.0 - f_func(k)) * (np.log(np.e + 1.8 * beta_c * q) / (
            np.log(np.e + 1.8 * beta_c * q) + (14.2 + 386.0 / (1.0 + 69.9 * (q ** 1.08))) * (q ** 2)
        ))

        if not with_baryon_wiggles:
            # Eisenstein & Hu no-wiggle smooth transfer function
            alpha_gamma = 1.0 - 0.328 * np.log(126.0 * om_h2) * fb + 0.38 * np.log(45.0 * om_h2) * (fb ** 2)
            gamma_eff = om * h * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k * sound_horizon) ** 4))
            q_eff = k * (theta_cmb ** 2) / gamma_eff
            l0 = np.log(2.0 * np.e + 1.8 * q_eff)
            c0 = 14.2 + 731.0 / (1.0 + 62.5 * q_eff)
            return l0 / (l0 + c0 * (q_eff ** 2))

        # Baryon transfer function with acoustic oscillations
        s_tilde = sound_horizon / ((1.0 + (1.0 / (k * sound_horizon)) ** 3) ** (1.0 / 3.0))
        beta_node = 8.41 * (om_h2 ** 0.435)
        beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * np.sqrt((17.2 * om_h2) ** 2 + 1.0)
        t_b_sin = np.sin(k * s_tilde) / (k * s_tilde + 1e-12)
        t_b_decay = np.exp(-((k / k_silk) ** 1.4))

        t_b = (np.log(np.e + 1.8 * q) / (np.log(np.e + 1.8 * q) + c_val * (q ** 2)) / (
            1.0 + (k * sound_horizon / 5.4) ** 2
        ) + (beta_b / (1.0 + (beta_node / (k * sound_horizon + 1e-12)) ** 3)) * t_b_sin) * t_b_decay

        return fc * t_c + fb * t_b

    def linear_matter_power_spectrum(
        self,
        k: Union[float, np.ndarray],
        z: float = 0.0,
        with_baryon_wiggles: bool = True,
    ) -> Union[float, np.ndarray]:
        """
        Computes the linear matter power spectrum P_lin(k, z) in (Mpc/h)^3.
        Normalized to reproduce sigma8 at z=0.
        """
        k_arr = np.asarray(k, dtype=np.float64)

        # Ensure sigma8 normalization factor is computed
        if self._sigma8_norm is None:
            self._compute_sigma8_norm(with_baryon_wiggles=with_baryon_wiggles)

        # Unnormalized P(k) = k^ns * T(k)^2
        t_k = self.transfer_function_eh98(k_arr, with_baryon_wiggles=with_baryon_wiggles)
        p_unnorm = (k_arr ** self.params.ns) * (t_k ** 2)

        # Scale with growth factor D1(z)
        d_z = self.growth_factor(z)
        p_k = self._sigma8_norm * p_unnorm * (d_z ** 2)

        return float(p_k) if np.isscalar(k) else p_k

    def _compute_sigma8_norm(self, with_baryon_wiggles: bool = True) -> None:
        """Computes amplitude normalization factor to match sigma8 at z=0."""
        r8 = 8.0  # 8 Mpc/h

        def integrand(lnk: float) -> float:
            k = np.exp(lnk)
            t_k = self.transfer_function_eh98(np.array([k]), with_baryon_wiggles=with_baryon_wiggles)[0]
            p_unnorm = (k ** self.params.ns) * (t_k ** 2)
            x = k * r8
            # Spherical top-hat window function W(k*R)
            if x < 1e-3:
                w = 1.0 - (x ** 2) / 10.0
            else:
                w = 3.0 * (np.sin(x) - x * np.cos(x)) / (x ** 3)
            # Integration in d(ln k): k^3 * P(k) * W^2(kR) / (2*pi^2)
            return (k ** 3) * p_unnorm * (w ** 2) / (2.0 * np.pi ** 2)

        # Integrate from k = 1e-4 to 1e2 h/Mpc
        integral, _ = integrate.quad(integrand, np.log(1e-4), np.log(1e2), limit=200)
        sigma8_unnorm = np.sqrt(np.maximum(integral, 1e-12))

        self._sigma8_norm = (self.params.sigma8 / sigma8_unnorm) ** 2
