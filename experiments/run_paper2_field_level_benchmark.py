"""
Executable Stage 5 (Paper 2 Extension) Benchmark:
Differentiable Field-Level Initial Condition Reconstruction & Preconditioned MCLMC vs HMC.
"""

from cosmoverse.inference.field_level import FieldLevelInferenceEngine


def main() -> None:
    print("=" * 76)
    print("COSMOVERSE -- PAPER 2 (STAGE 5) DIFFERENTIABLE FIELD-LEVEL BENCHMARK")
    print("=" * 76)

    engine = FieldLevelInferenceEngine(grid_size=8, box_size=100.0, noise_sigma=0.25)
    report = engine.benchmark_mclmc_vs_hmc(n_samples=80, n_leapfrog=6, seed=42)

    print(f"Field-Level State Dimension : {report['dimension']} (3D IC grid + amplitude)")
    print(f"Large-Scale IC Cross-Corr r(k): {report['mean_large_scale_rk']:.4f}")
    print(
        f"Preconditioned MCLMC EEVPD    : {report['preconditioned_mclmc']['eevpd']:.6e} "
        f"(stable={report['preconditioned_mclmc']['is_energy_stable']})"
    )
    print(
        f"Unpreconditioned MCLMC EEVPD  : {report['unpreconditioned_mclmc']['eevpd']:.6e} "
        f"(degradation={report['unpreconditioned_mclmc']['degradation_ratio_eevpd']:.1f}x)"
    )
    print(
        f"Preconditioned HMC Acceptance : {report['preconditioned_hmc_plan_b']['acceptance_rate']:.2%}"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
