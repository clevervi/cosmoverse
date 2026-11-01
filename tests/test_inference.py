"""
Unit tests for Normalizing Flows, PosteriorSummary, SBI (Phase A), and Preconditioned MCLMC (Phase B).
"""

import numpy as np

from cosmoverse.inference.flows import ConditionalRealNVP
from cosmoverse.inference.mclmc import MicrocanonicalHMC
from cosmoverse.inference.posterior import PosteriorSummary
from cosmoverse.inference.sbi import SBIEngine


def test_conditional_real_nvp():
    flow = ConditionalRealNVP(param_dim=2, context_dim=5, n_layers=2, seed=42)
    context = np.ones(5)
    samples = flow.sample(context, n_samples=50, seed=42)
    assert samples.shape == (50, 2)


def test_posterior_summary():
    samples = np.random.normal(loc=[0.3, 0.8], scale=[0.02, 0.04], size=(1000, 2))
    summary = PosteriorSummary(samples, param_names=["omega_m", "sigma8"])

    ci = summary.credible_intervals()
    assert np.isclose(ci["omega_m"][0], 0.3, atol=0.01)
    assert np.isclose(ci["sigma8"][0], 0.8, atol=0.01)

    fom = summary.figure_of_merit("omega_m", "sigma8")
    assert fom > 0.0


def test_sbi_pipeline():
    rng = np.random.default_rng(42)
    def sim_fn(theta):
        return theta * 2.0 + rng.normal(0, 0.02, size=len(theta))
    bounds = {"omega_m": (0.1, 0.5), "sigma8": (0.6, 1.0)}
    sbi = SBIEngine(simulator_fn=sim_fn, param_bounds=bounds, seed=42)

    thetas, xs = sbi.generate_training_bank(n_simulations=25)
    assert thetas.shape == (25, 2)
    assert xs.shape == (25, 2)

    post = sbi.infer(observed_x=np.array([0.6, 1.6]), n_posterior_samples=200)
    assert post.n_samples == 200
    ci = post.credible_intervals()
    assert np.isclose(ci["omega_m"][0], 0.3, atol=0.05)


def test_preconditioned_mclmc_and_hmc_plan_b():
    # Anisotropic Gaussian potential U(q) = 0.5 * sum((q / sigma)^2)
    sigmas = np.array([0.05, 1.5, 0.2, 0.8])
    def pot_fn(q):
        return float(0.5 * np.sum((q / sigmas) ** 2))
    def grad_fn(q):
        return q / (sigmas ** 2)

    sampler = MicrocanonicalHMC(
        potential_fn=pot_fn,
        grad_potential_fn=grad_fn,
        step_size=0.08,
        n_leapfrog_steps=5,
        preconditioner_scale=sigmas,
    )
    mclmc_samples = sampler.sample_mclmc_chain(np.zeros(4), n_samples=100, seed=42)
    assert mclmc_samples.shape == (100, 4)
    assert sampler.last_diagnostics["preconditioned"] is True
    assert sampler.last_diagnostics["is_energy_stable"] is True

    hmc_samples = sampler.sample_chain(np.zeros(4), n_samples=60, seed=42)
    assert hmc_samples.shape == (60, 4)
    assert sampler.last_diagnostics["acceptance_rate"] > 0.5
