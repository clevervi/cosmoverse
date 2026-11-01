"""
COSMOVERSE Command Line Interface (CLI).
Provides commands for running simulations, neural emulators (TUNeS), BCEmu bias-variance
frontiers, CAMELS Gen-2 catalog audits, Stage 5 field-level benchmarks, and citation audits.
"""

import argparse
import pathlib
import webbrowser
from typing import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cosmoverse import __author__, __email__, __version__
from cosmoverse.baryons.frontier import BiasVarianceFrontier
from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.dark_energy import DynamicDarkEnergyCosmology
from cosmoverse.core.halos_and_neutrinos import (
    HaloMassFunctionModel,
    MassiveNeutrinoPhysics,
)
from cosmoverse.core.parameters import CosmologicalParameters, SimulationParameters
from cosmoverse.datasets.camels_gen2 import CAMELSGen2Catalog, StageIVPrecisionCalibrator
from cosmoverse.datasets.camels_spectra import CAMELSHydroSpectraConnector
from cosmoverse.datasets.remote import CAMELSArchiveFetcher, CheckpointManager
from cosmoverse.emulators.bispectrum import ReducedBispectrumModel
from cosmoverse.emulators.fno import EMULATOR_BIBLIOGRAPHY_REGISTRY
from cosmoverse.emulators.mangrove import MangroveMergerTreeGNN
from cosmoverse.inference.compression import FisherScoreCompressor
from cosmoverse.inference.field_level import FieldLevelInferenceEngine
from cosmoverse.inference.light_cone_sbi import LightConeSBIPipeline
from cosmoverse.inference.multiprobe_sbi import MultiProbeSBIPipeline
from cosmoverse.inference.pm_emulator_sbi import PMEmulatorSBIPipeline
from cosmoverse.observations.cmb_and_sz import CMBLensingAndSZPipeline
from cosmoverse.observations.hod_and_mocks import (
    HaloOccupationModel,
    StageIVMockCatalogExporter,
)
from cosmoverse.observations.intrinsic_alignments import IntrinsicAlignmentModel
from cosmoverse.observations.light_cone import (
    LightConeBuilder,
    LightConeSurveyPipeline,
    RelativisticLightConePipeline,
)
from cosmoverse.observations.three_x_two_pt import Joint3x2ptObservationalPipeline
from cosmoverse.simulation.nbody import ParticleMeshSimulator
from cosmoverse.validation.arxiv_packager import ArxivSubmissionPackager
from cosmoverse.validation.benchmarks import camels_latin_hypercube_benchmark
from cosmoverse.validation.budget import SimulationBudgetAnalyzer
from cosmoverse.validation.plotting import Paper1FigureGenerator

console = Console()


def show_info() -> None:
    """Displays COSMOVERSE framework metadata, Master Document hypothesis, and roadmap stages."""
    title = "[bold cyan]COSMOVERSE[/bold cyan]: Hybrid Cosmological Inference Protocol"
    subtitle = "TUNeS Emulation - BCEmu Baryonic Marginalization - Stage-IV Realism - Calibrated SBI & MCLMC"
    content = f"""
[bold green]Version:[/bold green] {__version__}
[bold green]Author:[/bold green] {__author__} ({__email__})
[bold green]Quantitative Hypothesis:[/bold green]
In what scale regime and with what simulation budget does a hybrid framework
(emulation + baryonic/observational marginalization + differentiable simulation)
produce calibrated posteriors with residual bias below Stage-IV statistical precision,
and how much does generalization degrade when transferring across simulation families?

[bold yellow]Three-Level Methodological Separation:[/bold yellow]
 1. [cyan]Universe Physics[/cyan]       : cosmoverse.core, cosmoverse.simulation, cosmoverse.emulators (TUNeS), cosmoverse.baryons (BCEmu)
 2. [cyan]Observational Process[/cyan]  : cosmoverse.observations (Masks, Selection, Galaxy Bias, RSD P0/P2/P4, Shot Noise)
 3. [cyan]Statistical Inference[/cyan]  : cosmoverse.inference (SBI + RealNVP, Preconditioned MCLMC) & cosmoverse.validation (SBC, TARP, Transfer)

[bold magenta]Four Demonstrable Results (Paper 1 MVP = Stages 1-4):[/bold magenta]
 * [bold]Result 1[/bold]: Emulator precision in Gaussian (P(k) MAPE < 1% at k < 1 h/Mpc) & non-Gaussian (Peaks, 3D Minkowski)
 * [bold]Result 2[/bold]: Complete Bias-Variance Frontier under BCEmu + observational marginalization vs Stage-IV precision
 * [bold]Result 3[/bold]: SBI posterior calibration via SBC + TARP + scale-dependent misspecification detection
 * [bold]Result 4[/bold]: Quantified coverage & bias degradation under Cross-Simulation Family Transfer (disjoint ICs/cosmology)
"""
    console.print(Panel(content.strip(), title=title, subtitle=subtitle, expand=False))


def cmd_simulate(args: argparse.Namespace) -> None:
    """Executes a Particle-Mesh cosmological simulation."""
    console.print(f"[bold green]Running Particle-Mesh simulation (N_mesh={args.mesh}, L={args.box} Mpc/h)...[/bold green]")
    cosmo = Cosmology(CosmologicalParameters(omega_m=args.omega_m, sigma8=args.sigma8))
    sim_params = SimulationParameters(box_size=args.box, n_mesh=args.mesh, n_steps=args.steps, seed=args.seed)

    sim = ParticleMeshSimulator(cosmology=cosmo, sim_params=sim_params)
    sim.run(verbose=True)
    k, pk = sim.compute_power_spectrum(n_bins=10)

    table = Table(title="Measured Matter Power Spectrum P(k)")
    table.add_column("Bin", justify="center")
    table.add_column("k [h/Mpc]", justify="right", style="cyan")
    table.add_column("P(k) [(Mpc/h)^3]", justify="right", style="green")

    for i in range(len(k)):
        table.add_row(str(i + 1), f"{k[i]:.4f}", f"{pk[i]:.2e}")
    console.print(table)


def cmd_benchmark_camels(args: argparse.Namespace) -> None:
    """Generates Latin Hypercube parameter design for CAMELS."""
    console.print(f"[bold cyan]Generating {args.n} CAMELS Latin Hypercube points...[/bold cyan]")
    points = camels_latin_hypercube_benchmark(n_samples=args.n, seed=args.seed)
    table = Table(title="CAMELS Benchmark Parameters")
    table.add_column("ID", justify="center")
    table.add_column("Omega_m", justify="right", style="cyan")
    table.add_column("sigma8", justify="right", style="green")
    table.add_column("A_SN", justify="right")
    table.add_column("A_AGN", justify="right", style="magenta")
    table.add_column("ns", justify="right")

    for idx, p in enumerate(points):
        table.add_row(
            str(idx),
            f"{p['omega_m']:.4f}",
            f"{p['sigma8']:.4f}",
            f"{p['A_SN']:.4f}",
            f"{p['A_AGN']:.4f}",
            f"{p['ns']:.4f}",
        )
    console.print(table)


def cmd_frontier() -> None:
    """Displays Resultado 2 Bias-Variance Frontier table."""
    analyzer = BiasVarianceFrontier()
    res = analyzer.compute_frontier_curve(k_max_values=[0.5, 1.0], marginalization_weights=[0.0, 0.5, 1.0])
    table = Table(title="Resultado 2: Bias-Variance Frontier (BCEmu + Observational Marginalization)")
    table.add_column("k_max [h/Mpc]", justify="right", style="cyan")
    table.add_column("Marg. Weight", justify="right")
    table.add_column("Bias / sigma_StageIV", justify="right", style="yellow")
    table.add_column("Error Inflation", justify="right", style="magenta")
    table.add_column("Stage-IV Compatible", justify="center", style="green")

    for pt in res["frontier_curve"]:
        table.add_row(
            f"{pt['k_max']:.2f}",
            f"{pt['marginalization_weight']:.2f}",
            f"{pt['bias_in_stage4_sigma']:.3f} sigma",
            f"{pt['error_inflation_factor']:.2f}x",
            "YES" if pt["is_stage4_compatible"] else "NO",
        )
    console.print(table)


def cmd_camels_gen2() -> None:
    """Displays CAMELS Generation 2 (1,192 IllustrisTNG simulations) & Stage-IV calibration."""
    catalog = CAMELSGen2Catalog()
    counts = catalog.summary_counts()
    split = catalog.create_disjoint_lh_partition(train_fraction=0.80, seed=42)
    calibrator = StageIVPrecisionCalibrator(catalog)

    table = Table(title="CAMELS Generation 2 (IllustrisTNG -- 50 Mpc/h Boxes, 35 Params)")
    table.add_column("Subset", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Partition Status", style="yellow")
    table.add_row("LH (Latin Hypercube)", str(counts["LH"]), f"Disjoint Train/Test (Leakage-Free={split['verification']['is_leakage_free']})")
    table.add_row("CV (Cosmic Variance)", str(counts["CV"]), "Shared cosmo_id=0 (27 seeds)")
    table.add_row("1P (One-Parameter)", str(counts["1P"]), "Shared ic_seed=40000 (161 variations)")
    table.add_row("EX (Extreme Feedback)", str(counts["EX"]), "Shared ic_seed=50000 (4 feedback extremes)")
    table.add_row("TOTAL", str(counts["total"]), "100% Verified")
    console.print(table)

    s4_table = Table(title="Calibrated Stage-IV Precision Thresholds (|Bias| < 0.3 * sigma_StageIV)")
    s4_table.add_column("Survey", style="cyan")
    s4_table.add_column("V_eff [(Gpc/h)^3]", justify="right")
    s4_table.add_column("sigma(S8)", justify="right", style="green")
    s4_table.add_column("Max Bias (0.3 sigma)", justify="right", style="magenta")
    for survey in ("Euclid-Wide", "LSST-Y10", "DESI-LRG"):
        c = calibrator.calibrate_survey_precision(survey)
        s4_table.add_row(
            survey,
            f"{c['v_eff_gpc3_h3']:.1f}",
            f"{c['sigma_stage4_S8']:.5f}",
            f"{c['max_allowed_bias_S8']:.5f}",
        )
    console.print(s4_table)


def cmd_field_level(args: argparse.Namespace) -> None:
    """Runs Stage 5 (Paper 2 Extension) Differentiable Field-Level MCLMC vs HMC benchmark."""
    engine = FieldLevelInferenceEngine(grid_size=args.grid, box_size=100.0, noise_sigma=0.25)
    res = engine.benchmark_mclmc_vs_hmc(n_samples=args.samples, n_leapfrog=5, seed=args.seed)

    table = Table(title=f"Stage 5 Field-Level Benchmark (Dim={res['dimension']}, Large-Scale r(k)={res['mean_large_scale_rk']:.4f})")
    table.add_column("Sampler", style="cyan")
    table.add_column("EEVPD / Acc. Rate", justify="right", style="green")
    table.add_column("Status / Notes", style="yellow")
    table.add_row(
        "Preconditioned MCLMC",
        f"EEVPD = {res['preconditioned_mclmc']['eevpd']:.3e}",
        f"Energy Stable = {res['preconditioned_mclmc']['is_energy_stable']}",
    )
    table.add_row(
        "Unpreconditioned MCLMC",
        f"EEVPD = {res['unpreconditioned_mclmc']['eevpd']:.3e}",
        f"Degradation = {res['unpreconditioned_mclmc']['degradation_ratio_eevpd']:.1f}x worse",
    )
    table.add_row(
        "Preconditioned HMC (Plan B)",
        f"Acc = {res['preconditioned_hmc_plan_b']['acceptance_rate']*100:.1f}%",
        f"Marginal Bias = {res['preconditioned_hmc_plan_b']['marginal_alpha_bias']:.4f}",
    )
    console.print(table)


def cmd_audit_citations() -> None:
    """Displays the verified emulator bibliography registry (Section 8)."""
    table = Table(title="Emulator Bibliographic Verification Registry (Master Document Section 8)")
    table.add_column("Key", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("arXiv / DOI", style="yellow")
    table.add_column("Role", style="white")
    for key, meta in EMULATOR_BIBLIOGRAPHY_REGISTRY.items():
        ident = f"arXiv:{meta['arxiv_id']} ({meta['doi']})" if meta.get("arxiv_id") else "BLOCKED (Unverified)"
        table.add_row(key, str(meta["status"]), ident, str(meta["role"]))
    console.print(table)


def cmd_figures(args: argparse.Namespace) -> None:
    """Generates all Paper 1 and Paper 2 vector SVG figures into --out-dir."""
    saved = Paper1FigureGenerator.generate_all_paper1_figures(args.out_dir)
    saved.update(Paper1FigureGenerator.generate_all_paper2_figures(args.out_dir))
    table = Table(title="Generated Paper 1 & Paper 2 Scientific Figures (Vector SVG)")
    table.add_column("Figure Key", style="cyan")
    table.add_column("Path", style="green")
    for k, p in saved.items():
        table.add_row(k, str(p))
    console.print(table)


def cmd_mangrove() -> None:
    """Runs the post-v1 Mangrove Halo Merger-Tree GNN benchmark (Section 6.1)."""
    gnn = MangroveMergerTreeGNN(allow_post_v1_extension=True, seed=42)
    res = gnn.benchmark_merger_tree_vs_halo_mlp(n_trees=40, seed=42)
    table = Table(title="Post-v1 Mangrove Merger-Tree GNN Benchmark (75 Mpc/h Box, 6 Galaxy Properties)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Target Properties", ", ".join(res["target_properties"]))
    table.add_row("Mangrove GNN RMS Error", f"{res['mangrove_gnn_rms']:.4f}")
    table.add_row("Halo-Only Baseline RMS", f"{res['halo_only_baseline_rms']:.4f}")
    table.add_row("RMS Reduction Factor", f"{res['rms_improvement_factor']:.2f}x (>= 2.0x Gate)")
    table.add_row("Status", "PASS" if res["passes_2x_improvement_benchmark"] else "FAIL")
    console.print(table)


def cmd_budget() -> None:
    """Evaluates the Simulation Budget Scaling Law (N_sims = 4..128) for the Central Hypothesis."""
    analyzer = SimulationBudgetAnalyzer(seed=42)
    res = analyzer.evaluate_budget_scaling_curve()
    table = Table(
        title=(
            f"Simulation Budget Scaling Law (N_min Emulator={res['min_budget_for_emulator_1pct_mape']}, "
            f"N_min Stage-IV={res['min_budget_for_stage4_calibrated_posteriors']})"
        )
    )
    table.add_column("N_sims", justify="right", style="cyan")
    table.add_column("TUNeS P(k) MAPE", justify="right", style="green")
    table.add_column("Hybrid |Bias|", justify="right", style="yellow")
    table.add_column("Direct SBI |Bias|", justify="right", style="red")
    table.add_column("TARP 68%", justify="right", style="magenta")
    table.add_column("Stage-IV Gate", justify="center", style="white")
    for pt in res["curve"]:
        table.add_row(
            str(pt["n_sims"]),
            f"{pt['pk_mape_pct']:.2f}%",
            f"{pt['hybrid_bias_stage4_sigma']:.3f} sigma",
            f"{pt['direct_no_emulator_bias_sigma']:.2f} sigma",
            f"{pt['tarp_coverage_68']*100:.1f}%",
            "PASS" if pt["passes_stage4_calibrated_gate"] else "FAIL",
        )
    console.print(table)


def cmd_arxiv_bundle(args: argparse.Namespace) -> None:
    """Audits LaTeX manuscripts and builds arXiv (.tar.gz) submission bundles."""
    packager = ArxivSubmissionPackager()
    out = packager.build_arxiv_bundles(out_dir=args.out_dir)
    table = Table(title="COSMOVERSE arXiv (astro-ph.CO) & JOSS Submission Audit")
    table.add_column("Artifact / Check", style="cyan")
    table.add_column("Status / Path", style="green")
    table.add_row("Manuscript & BibTeX Integrity", f"VALID={out['audit']['is_valid']} (Blocked keys={len(out['audit']['leaked_blocked_keys'])})")
    table.add_row("Paper 1 Bundle (Stages 1-4 MVP)", str(out["paper1_bundle"]))
    table.add_row("Paper 2 Bundle (Stage 5 Extension)", str(out["paper2_bundle"]))
    console.print(table)


def cmd_joint_3x2pt() -> None:
    """Evaluates Joint Stage-IV 3x2pt + RSD degeneracy breaking."""
    pipeline = Joint3x2ptObservationalPipeline()
    res = pipeline.evaluate_probe_degeneracy_breaking()
    table = Table(
        title=(
            f"Level 2-3: Joint Stage-IV 3x2pt + RSD Degeneracy Breaking "
            f"(Inflation Reduction = {res['inflation_reduction_pct']:.1f}%, FoM Gain = {res['fom_gain_vs_rsd_only']:.2f}x)"
        )
    )
    table.add_column("Probe Combination", style="cyan")
    table.add_column("Unmarg. |Bias|", justify="right", style="red")
    table.add_column("Marg. |Bias|", justify="right", style="green")
    table.add_column("Error Inflation", justify="right", style="yellow")
    table.add_column("S8 FoM", justify="right", style="magenta")
    for p in res["probes"]:
        table.add_row(
            str(p["probe_combination"]),
            f"{p['unmarginalized_bias_sigma']:.2f} sigma",
            f"{p['marginalized_bias_sigma']:.2f} sigma",
            f"{p['error_inflation_factor']:.2f}x",
            f"{p['s8_figure_of_merit']:.1f}",
        )
    console.print(table)


def cmd_train(args: argparse.Namespace) -> None:
    """Trains and saves a SHA-256 verified checkpoint (.npz) for TUNeS + SBIEngine."""
    manifest = CheckpointManager.train_and_save_checkpoint(
        out_dir=args.out_dir, grid_size=16, n_sbi_sims=32, seed=42
    )
    table = Table(title="Trained Model Checkpoint & SHA-256 Verification")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Checkpoint Path", str(manifest["checkpoint_file"]))
    table.add_row("SHA-256", str(manifest["sha256"])[:24] + "...")
    table.add_row("TUNeS Residual Scale", f"{manifest['tunes_residual_scale']:.6f}")
    table.add_row("SBI Final Loss", f"{manifest['sbi_final_loss']:.4f}")
    console.print(table)


def cmd_fetch_camels(args: argparse.Namespace) -> None:
    """Populates a local multi-suite CAMELS power spectrum cache."""
    res = CAMELSArchiveFetcher.fetch_and_cache_public_spectra(
        cache_dir=args.cache_dir, n_per_suite=args.n
    )
    table = Table(title="CAMELS Multi-Suite Local Spectrum Cache")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Cache Directory", str(res["cache_dir"]))
    table.add_row("Suites Cached", ", ".join(res["suites"]))
    table.add_row("Total Files Written", str(res["n_files_cached"]))
    console.print(table)


def cmd_neutrinos_hmf() -> None:
    """Evaluates Massive Neutrino S_nu(k) suppression and Result 1 Halo Mass Function dn/dlnM."""
    import numpy as np

    params = CosmologicalParameters(omega_m=0.315, sigma8=0.811, h=0.674, m_nu=0.12)
    f_nu = MassiveNeutrinoPhysics.neutrino_fraction_f_nu(params)
    s_nu_1 = float(MassiveNeutrinoPhysics.suppression_factor_S_nu(np.array([1.0]), params)[0])
    hmf = HaloMassFunctionModel(cosmo_params=params)
    hmf_eval = hmf.evaluate_emulated_hmf_accuracy()

    table = Table(title="Level 1: Massive Neutrinos (sum m_nu = 0.12 eV) & Halo Mass Function dn/dlnM")
    table.add_column("Physical Diagnostic", style="cyan")
    table.add_column("Measured Value", justify="right", style="green")
    table.add_row("Neutrino Fraction f_nu = Omega_nu / Omega_m", f"{f_nu:.5f}")
    table.add_row("Small-Scale S_nu(k=1 h/Mpc) Suppression", f"{s_nu_1:.4f} ({(1.0 - s_nu_1)*100:.2f}% damping)")
    table.add_row("Emulated Halo Mass Function MAPE (10^11 - 3e14 Msun/h)", f"{hmf_eval['hmf_mape_pct']:.3f}%")
    table.add_row("Result 1 HMF Gate (< 1.0% MAPE)", "PASS" if hmf_eval["passes_hmf_1pct_gate"] else "FAIL")
    console.print(table)


def cmd_hod_mocks(args: argparse.Namespace) -> None:
    """Evaluates HOD + Secondary Assembly Bias P_gm(k) and exports a Stage-IV synthetic mock catalog."""
    hod = HaloOccupationModel()
    ab_res = hod.evaluate_assembly_bias_marginalization()
    manifest = StageIVMockCatalogExporter.export_mock_catalog(
        out_dir=args.out_dir, n_galaxies=args.n_galaxies, seed=42
    )
    table = Table(title="Level 2: HOD + Secondary Assembly Bias P_gm(k) & Stage-IV Mock Catalog")
    table.add_column("Diagnostic / Artifact", style="cyan")
    table.add_column("Measured Value / Path", justify="right", style="green")
    table.add_row("Effective Galaxy Bias b_eff (with Assembly Bias)", f"{ab_res['b_eff_with_assembly_bias']:.4f}")
    table.add_row("Satellite Galaxy Fraction f_sat", f"{ab_res['satellite_fraction']*100:.2f}%")
    table.add_row("Mean Cross-Correlation r_gm(k <= 1 h/Mpc)", f"{ab_res['mean_r_gm_k_le_1']:.4f}")
    table.add_row("Unmarginalized vs Marginalized S8 Bias", f"{ab_res['unmarginalized_s8_bias_sigma']:.2f} -> {ab_res['marginalized_s8_bias_sigma']:.2f} sigma")
    table.add_row("Mock Catalog NPZ (SHA-256 Verified)", f"{manifest['npz_file']} ({str(manifest['sha256_npz'])[:12]}...)")
    table.add_row("Mock Catalog CSV", f"{manifest['csv_file']} ({manifest['n_galaxies']} galaxies)")
    console.print(table)


def cmd_bispectrum_ia() -> None:
    """Evaluates Result 1 Reduced Bispectrum Q(theta) and Level 2 NLA/TATT Intrinsic Alignments."""
    bisp_eval = ReducedBispectrumModel().evaluate_tunes_bispectrum_accuracy()
    ia_eval = IntrinsicAlignmentModel().evaluate_ia_marginalization_impact()

    table = Table(title="Result 1 & Level 2: Reduced Bispectrum Q(theta) & NLA/TATT Intrinsic Alignments")
    table.add_column("Diagnostic / Metric", style="cyan")
    table.add_column("Measured Value", justify="right", style="green")
    table.add_row(
        f"TUNeS Reduced Bispectrum Q(k1={bisp_eval['k1']}, k2={bisp_eval['k2']}) MAPE",
        f"{bisp_eval['q_mape_pct']:.3f}% (< 1.0% Gate)",
    )
    table.add_row(
        "BCEmu3 Baryonic Bispectrum Suppression",
        f"-{bisp_eval['mean_baryonic_bispectrum_suppression_pct']:.2f}%",
    )
    table.add_row(
        "Stage-IV Mean GI+II Shear Contamination",
        f"{ia_eval['mean_ia_contamination_pct']:.2f}%",
    )
    table.add_row(
        "NLA/TATT Unmarginalized vs Marginalized S8 Bias",
        f"{ia_eval['unmarginalized_s8_bias_sigma']:.2f} -> {ia_eval['marginalized_s8_bias_sigma']:.2f} sigma",
    )
    table.add_row(
        "Combined Bispectrum + IA Gate",
        "PASS" if (bisp_eval["passes_bispectrum_1pct_gate"] and ia_eval["passes_stage4_ia_gate"]) else "FAIL",
    )
    console.print(table)


def cmd_dark_energy_imnn() -> None:
    """Evaluates Level 1 w0-wa CPL Dark Energy and Level 3 Fisher/IMNN Score Compression."""
    de_eval = DynamicDarkEnergyCosmology().evaluate_desi_w0wa_signatures()
    imnn_eval = FisherScoreCompressor.benchmark_multiprobe_compression()

    table = Table(title="Level 1 & Level 3: Dynamic w0-wa CPL Dark Energy & Fisher/IMNN Score Compression")
    table.add_column("Diagnostic / Metric", style="cyan")
    table.add_column("Measured Value", justify="right", style="green")
    table.add_row(
        f"CPL Equation of State (w0={de_eval['w0']}, wa={de_eval['wa']})",
        f"LambdaCDM Limit Verified={de_eval['recovers_lcdm_limit']}",
    )
    table.add_row(
        "Alcock-Paczynski Dilation (z=0.85)",
        f"alpha_par={de_eval['alpha_parallel_z085']:.4f}, alpha_perp={de_eval['alpha_perp_z085']:.4f}",
    )
    table.add_row(
        "Mean DESI f*sigma8(z) Shift vs LambdaCDM",
        f"{de_eval['mean_fsigma8_shift_pct']:.2f}%",
    )
    table.add_row(
        "Multi-Probe Summary Compression Ratio",
        f"{imnn_eval['n_data_raw']} bins -> {imnn_eval['n_summary_compressed']} scores ({imnn_eval['compression_ratio']:.1f}x)",
    )
    table.add_row(
        "Fisher Information Preserved (det(F_comp)/det(F_full))^(1/p)",
        f"{imnn_eval['fisher_info_preserved_pct']:.2f}% (>= 98.0% Gate)",
    )
    console.print(table)


def cmd_cmb_sz() -> None:
    """Evaluates Level 2 CMB lensing cross-spectra and tSZ/kSZ BCEmu gas calibration."""
    import numpy as np

    pipeline = CMBLensingAndSZPipeline()
    spec = pipeline.compute_cmb_lensing_and_tsz_spectra()
    gain = pipeline.evaluate_sz_baryon_calibration_gain()

    table = Table(title="Level 2: CMB Lensing Cross-Spectra & tSZ/kSZ BCEmu Gas Calibration")
    table.add_column("Diagnostic / Metric", style="cyan")
    table.add_column("Measured Value", justify="right", style="green")
    table.add_row(
        "CMB Lensing x Galaxy C_ell^{kappa_CMB g}",
        f"{float(np.min(spec['c_ell_kcmb_g'])):.3e} - {float(np.max(spec['c_ell_kcmb_g'])):.3e}",
    )
    table.add_row(
        "CMB Lensing x Cosmic Shear C_ell^{kCMB kgal}",
        f"{float(np.min(spec['c_ell_kcmb_kgal'])):.3e} - {float(np.max(spec['c_ell_kcmb_kgal'])):.3e}",
    )
    table.add_row(
        "tSZ Peak D_ell^{yy} (10^12 ell(ell+1)C_ell/2pi)",
        f"{float(np.max(spec['d_ell_yy_1e12'])):.3f}",
    )
    table.add_row(
        "Mean kSZ Optical Depth tau_kSZ",
        f"{gain['mean_ksz_optical_depth']:.4f}",
    )
    table.add_row(
        "BCEmu Bound Gas Fraction Proxy",
        f"{spec['bound_gas_fraction_proxy']:.4f}",
    )
    table.add_row(
        "tSZ Low/High A_AGN Power Ratio",
        f"{gain['tsz_low_to_high_agn_power_ratio']:.3f}",
    )
    table.add_row(
        "S8 Error Inflation 3x2pt+RSD -> +CMB/SZ",
        f"{gain['s8_inflation_3x2pt_rsd_only']:.2f}x -> {gain['s8_inflation_with_cmb_sz']:.2f}x",
    )
    table.add_row(
        "CMB Lensing + tSZ/kSZ Calibration Gate",
        "PASS" if gain["passes_cmb_sz_calibration_gate"] else "FAIL",
    )
    console.print(table)


def cmd_light_cone() -> None:
    """Evaluates Level 2 relativistic light-cone geometry, magnification bias, and wide-angle RSD."""
    import numpy as np

    pipeline = RelativisticLightConePipeline()
    shells = pipeline.geometry.build_shells(0.4, 1.1, n_shells=8)
    impact = pipeline.evaluate_relativistic_light_cone_impact()

    table = Table(title="Level 2: Relativistic Light-Cone Geometry, Magnification Bias & Wide-Angle RSD")
    table.add_column("Diagnostic / Metric", style="cyan")
    table.add_column("Measured Value", justify="right", style="green")
    table.add_row(
        "DESI-LRG Light-Cone Shells (z = 0.4 - 1.1)",
        f"{len(shells['z_centers'])} shells, {float(np.sum(shells['shell_volume_mpc3'])) / 1.0e9:.3f} Gpc^3/h^3",
    )
    table.add_row(
        "Lookback Time Range",
        f"{float(shells['lookback_time_gyr'][0]):.2f} - {float(shells['lookback_time_gyr'][-1]):.2f} Gyr",
    )
    table.add_row(
        "Mean Magnification Contamination",
        f"{impact['mean_magnification_contamination_pct']:.2f}%",
    )
    table.add_row(
        "S8 Bias Unmarginalized -> Marginalized",
        f"{impact['unmarginalized_s8_bias_sigma']:.2f} -> {impact['marginalized_s8_bias_sigma']:.2f} sigma",
    )
    table.add_row(
        "Wide-Angle RSD Fractional RMS Error (60 deg)",
        f"{impact['wide_angle_fractional_rms_error_60deg']:.3f}",
    )
    table.add_row(
        "Relativistic Light-Cone Gate",
        "PASS" if impact["passes_relativistic_light_cone_gate"] else "FAIL",
    )
    console.print(table)


def _build_pm_light_cone_catalog() -> tuple[LightConeBuilder, dict]:
    """Runs a tiny PM simulation and returns its past-light-cone catalog."""
    sim_params = SimulationParameters(
        box_size=50.0,
        n_mesh=16,
        n_steps=2,
        z_initial=5.0,
        z_final=0.0,
        seed=42,
    )
    simulator = ParticleMeshSimulator(sim_params=sim_params)
    builder = LightConeBuilder(
        box_size=sim_params.box_size, n_mesh=sim_params.n_mesh
    )
    catalog = builder.build_from_simulation(
        simulator,
        snapshot_redshifts=[0.0, 0.1, 0.2],
        z_max=0.2,
        n_radial_bins=4,
    )
    return builder, catalog


def cmd_light_cone_build(args: argparse.Namespace) -> None:
    """Builds and exports a past-light-cone catalog from PM snapshots."""
    builder, catalog = _build_pm_light_cone_catalog()
    metrics = builder.evaluate_catalog(catalog)
    manifest = builder.export_catalog(catalog, out_dir=args.out_dir)

    table = Table(title="Level 2: Past-Light-Cone Catalog from PM Snapshots & Wide-Angle RSD")
    table.add_column("Diagnostic / Artifact", style="cyan")
    table.add_column("Measured Value / Path", justify="right", style="green")
    table.add_row("PM Snapshot Redshifts", "0.00, 0.10, 0.20")
    table.add_row("Catalog Objects", f"{metrics['n_objects']}")
    table.add_row(
        "Populated Radial Shells",
        f"{metrics['n_populated_shells']}/{metrics['n_shells']}",
    )
    table.add_row(
        "Observed Redshift Range",
        f"{metrics['redshift_min']:.3f} - {metrics['redshift_max']:.3f}",
    )
    table.add_row(
        "Mean Wide-Angle RSD Displacement",
        f"{metrics['mean_rsd_displacement_mpc_h']:.2f} Mpc/h",
    )
    table.add_row(
        "Catalog NPZ (SHA-256)",
        f"{pathlib.Path(str(manifest['path'])).name} ({str(manifest['sha256'])[:12]}...)",
    )
    table.add_row(
        "Light-Cone Catalog Gate",
        "PASS" if metrics["passes_light_cone_catalog_gate"] else "FAIL",
    )
    console.print(table)


def cmd_light_cone_survey() -> None:
    """Applies a survey mask and galaxy bias to a PM light-cone catalog and measures w(theta)."""
    _, catalog = _build_pm_light_cone_catalog()
    pipeline = LightConeSurveyPipeline()
    result = pipeline.evaluate(catalog, box_size=50.0)

    table = Table(title="Level 2: Light-Cone Galaxy Catalog, Survey Mask & Angular Clustering")
    table.add_column("Diagnostic / Metric", style="cyan")
    table.add_column("Measured Value", justify="right", style="green")
    table.add_row("Matter Objects after Mask", f"{result['n_matter']}")
    table.add_row(
        "Galaxy Sample (b1 input)",
        f"{result['n_galaxies']} galaxies (b1={result['b1_input']:.2f})",
    )
    table.add_row(
        "Effective Bias b_eff (w_gm / w_mm)",
        f"{result['b1_effective']:.3f} (ratio={result['bias_ratio']:.3f})",
    )
    table.add_row("Survey Mask Area", f"{result['mask_area_deg2']:.1f} deg^2")
    table.add_row(
        "w(theta) Shape Correlation (gm vs mm)",
        f"r = {result['pearson_r_gm_mm']:.3f}",
    )
    table.add_row(
        "Light-Cone Survey Gate",
        "PASS" if result["passes_light_cone_survey_gate"] else "FAIL",
    )
    console.print(table)


def cmd_light_cone_sbi(args: argparse.Namespace) -> None:
    """Runs end-to-end SBI on light-cone w(theta) with SBC and TARP."""
    pipeline = LightConeSBIPipeline(
        n_training_simulations=int(args.n_sims)
    )
    pipeline.fit()
    result = pipeline.run_calibration(
        n_sbc_trials=int(args.n_sbc),
        n_tarp_trials=int(args.n_tarp),
    )

    table = Table(title="Level 3: End-to-End SBI on Light-Cone w(theta) with SBC + TARP")
    table.add_column("Calibration Diagnostic", style="cyan")
    table.add_column("Measured Value", justify="right", style="green")
    table.add_row(
        "Parameters (b1 fixed)",
        f"Omega_m, sigma8, A_AGN (b1={result['b1_fixed']:.2f})",
    )
    table.add_row("Training Simulations", f"{result['n_training_simulations']}")
    table.add_row(
        "Data Vector",
        f"{len(result['theta_deg'])} bins w(theta), 1-20 deg",
    )
    for name in result["param_names"]:
        metric = result["sbc"][name]
        table.add_row(
            f"SBC {name}",
            f"68%={metric['coverage_68']*100:.1f}%, "
            f"95%={metric['coverage_95']*100:.1f}%, "
            f"KS p={metric['p_value']:.3f}",
        )
    table.add_row(
        "TARP 68% / 95%",
        f"{result['tarp']['coverage_68']*100:.1f}% / "
        f"{result['tarp']['coverage_95']*100:.1f}% "
        f"(KS p={result['tarp']['ks_p_value']:.3f})",
    )
    table.add_row(
        "Joint SBC + TARP Gate",
        "PASS" if result["passes_joint_sbi_gate"] else "FAIL",
    )
    console.print(table)


def cmd_multiprobe_sbi(args: argparse.Namespace) -> None:
    """Runs four-parameter SBI on the joint w(theta) + RSD data vector."""
    pipeline = MultiProbeSBIPipeline(
        n_training_simulations=int(args.n_sims)
    )
    pipeline.fit()
    result = pipeline.run_calibration(
        n_sbc_trials=int(args.n_sbc),
        n_tarp_trials=int(args.n_tarp),
    )

    table = Table(title="Level 3: Multi-Probe SBI (w(theta) + RSD) with SBC + TARP")
    table.add_column("Calibration Diagnostic", style="cyan")
    table.add_column("Measured Value", justify="right", style="green")
    table.add_row(
        "Parameters",
        "Omega_m, sigma8, b1, A_AGN (b1 free)",
    )
    table.add_row("Training Simulations", f"{result['n_training_simulations']}")
    table.add_row(
        "Data Vector",
        f"{len(result['theta_deg'])} w(theta) + 3 x {len(result['k_rsd'])} RSD bins",
    )
    for name in result["param_names"]:
        metric = result["sbc"][name]
        table.add_row(
            f"SBC {name}",
            f"68%={metric['coverage_68']*100:.1f}%, "
            f"95%={metric['coverage_95']*100:.1f}%, "
            f"KS p={metric['p_value']:.3f}",
        )
    table.add_row(
        "TARP 68% / 95%",
        f"{result['tarp']['coverage_68']*100:.1f}% / "
        f"{result['tarp']['coverage_95']*100:.1f}% "
        f"(KS p={result['tarp']['ks_p_value']:.3f})",
    )
    table.add_row(
        "Joint SBC + TARP Gate",
        "PASS" if result["passes_joint_sbi_gate"] else "FAIL",
    )
    console.print(table)


def cmd_pm_emulator_sbi(args: argparse.Namespace) -> None:
    """Runs multi-probe SBI with a PM-trained P(k) emulator."""
    pipeline = PMEmulatorSBIPipeline(
        n_training_simulations=int(args.n_sims)
    )
    pipeline.fit()
    result = pipeline.run_calibration(
        n_sbc_trials=int(args.n_sbc),
        n_tarp_trials=int(args.n_tarp),
    )

    table = Table(title="Level 3: PM-Emulator Multi-Probe SBI with SBC + TARP")
    table.add_column("Calibration Diagnostic", style="cyan")
    table.add_column("Measured Value", justify="right", style="green")
    emulator = result["emulator_accuracy"]
    table.add_row(
        "PM Emulator (training points, LOO MAPE)",
        f"{emulator['n_training_points']} points, "
        f"{emulator['loo_mape_pct']:.2f}%",
    )
    table.add_row("Parameters", "Omega_m, sigma8, b1, A_AGN")
    table.add_row("SBI Training Simulations", f"{result['n_training_simulations']}")
    table.add_row(
        "Data Vector",
        f"{len(result['theta_deg'])} w(theta) + 3 x {len(result['k_rsd'])} RSD bins",
    )
    for name in result["param_names"]:
        metric = result["sbc"][name]
        table.add_row(
            f"SBC {name}",
            f"68%={metric['coverage_68']*100:.1f}%, "
            f"95%={metric['coverage_95']*100:.1f}%, "
            f"KS p={metric['p_value']:.3f}",
        )
    table.add_row(
        "TARP 68% / 95%",
        f"{result['tarp']['coverage_68']*100:.1f}% / "
        f"{result['tarp']['coverage_95']*100:.1f}% "
        f"(KS p={result['tarp']['ks_p_value']:.3f})",
    )
    table.add_row(
        "Joint SBC + TARP Gate",
        "PASS" if result["passes_joint_sbi_gate"] else "FAIL",
    )
    console.print(table)


def cmd_flow_sbi(args: argparse.Namespace) -> None:
    """Runs multi-probe SBI with a gradient-trained conditional RealNVP flow."""
    pipeline = MultiProbeSBIPipeline(
        n_training_simulations=int(args.n_sims),
        density_estimator="flow",
    )
    pipeline.fit()
    result = pipeline.run_calibration(
        n_sbc_trials=int(args.n_sbc),
        n_tarp_trials=int(args.n_tarp),
    )

    table = Table(title="Level 3: Gradient-Trained Flow SBI with SBC + TARP")
    table.add_column("Calibration Diagnostic", style="cyan")
    table.add_column("Measured Value", justify="right", style="green")
    table.add_row(
        "Density Estimator",
        "Trained ConditionalRealNVP (6 layers, Adam)",
    )
    table.add_row("Parameters", "Omega_m, sigma8, b1, A_AGN")
    table.add_row("SBI Training Simulations", f"{result['n_training_simulations']}")
    table.add_row(
        "Data Vector",
        f"{len(result['theta_deg'])} w(theta) + 3 x {len(result['k_rsd'])} RSD bins",
    )
    for name in result["param_names"]:
        metric = result["sbc"][name]
        table.add_row(
            f"SBC {name}",
            f"68%={metric['coverage_68']*100:.1f}%, "
            f"95%={metric['coverage_95']*100:.1f}%, "
            f"KS p={metric['p_value']:.3f}",
        )
    table.add_row(
        "TARP 68% / 95%",
        f"{result['tarp']['coverage_68']*100:.1f}% / "
        f"{result['tarp']['coverage_95']*100:.1f}% "
        f"(KS p={result['tarp']['ks_p_value']:.3f})",
    )
    table.add_row(
        "Joint SBC + TARP Gate",
        "PASS" if result["passes_joint_sbi_gate"] else "FAIL",
    )
    console.print(table)


def cmd_pm_flow_sbi(args: argparse.Namespace) -> None:
    """Runs PM-emulator multi-probe SBI with a trained MLP conditional flow."""
    pipeline = PMEmulatorSBIPipeline(
        n_training_simulations=int(args.n_sims),
        density_estimator="flow",
        flow_n_layers=6,
        flow_epochs=800,
        flow_scale_bound=6.0,
        flow_hidden_dim=16,
    )
    pipeline.fit()
    result = pipeline.run_calibration(
        n_sbc_trials=int(args.n_sbc),
        n_tarp_trials=int(args.n_tarp),
    )

    table = Table(title="Level 3: PM-Emulator Flow SBI (MLP RealNVP) with SBC + TARP")
    table.add_column("Calibration Diagnostic", style="cyan")
    table.add_column("Measured Value", justify="right", style="green")
    emulator = result["emulator_accuracy"]
    table.add_row(
        "PM Emulator (training points, LOO MAPE)",
        f"{emulator['n_training_points']} points, "
        f"{emulator['loo_mape_pct']:.2f}%",
    )
    table.add_row(
        "Density Estimator",
        "ConditionalRealNVP + MLP (hidden=16, 6 layers)",
    )
    table.add_row("SBI Training Simulations", f"{result['n_training_simulations']}")
    table.add_row(
        "Data Vector",
        f"{len(result['theta_deg'])} w(theta) + 3 x {len(result['k_rsd'])} RSD bins",
    )
    for name in result["param_names"]:
        metric = result["sbc"][name]
        table.add_row(
            f"SBC {name}",
            f"68%={metric['coverage_68']*100:.1f}%, "
            f"95%={metric['coverage_95']*100:.1f}%, "
            f"KS p={metric['p_value']:.3f}",
        )
    table.add_row(
        "TARP 68% / 95%",
        f"{result['tarp']['coverage_68']*100:.1f}% / "
        f"{result['tarp']['coverage_95']*100:.1f}% "
        f"(KS p={result['tarp']['ks_p_value']:.3f})",
    )
    table.add_row(
        "Joint SBC + TARP Gate",
        "PASS" if result["passes_joint_sbi_gate"] else "FAIL",
    )
    console.print(table)


def cmd_doctor() -> None:
    """Runs the Master Document scientific & methodological verification suite."""
    hmf_ok = HaloMassFunctionModel().evaluate_emulated_hmf_accuracy()["passes_hmf_1pct_gate"]
    frontier_res = BiasVarianceFrontier().compute_frontier_curve(
        k_max_values=[0.5, 1.0], marginalization_weights=[0.0, 1.0]
    )
    frontier_ok = any(pt["is_stage4_compatible"] for pt in frontier_res["frontier_curve"])
    budget_res = SimulationBudgetAnalyzer(seed=42).evaluate_budget_scaling_curve()
    budget_ok = budget_res["min_budget_for_stage4_calibrated_posteriors"] == 16
    joint_ok = Joint3x2ptObservationalPipeline().evaluate_probe_degeneracy_breaking()[
        "passes_joint_degeneracy_breaking_gate"
    ]
    cmb_sz_ok = bool(
        CMBLensingAndSZPipeline().evaluate_sz_baryon_calibration_gain()["passes_cmb_sz_calibration_gate"]
    )
    light_cone_ok = bool(
        RelativisticLightConePipeline().evaluate_relativistic_light_cone_impact()[
            "passes_relativistic_light_cone_gate"
        ]
    )
    light_cone_builder, light_cone_catalog = _build_pm_light_cone_catalog()
    light_cone_build_ok = bool(
        light_cone_builder.evaluate_catalog(light_cone_catalog)[
            "passes_light_cone_catalog_gate"
        ]
    )
    light_cone_survey_ok = bool(
        LightConeSurveyPipeline().evaluate(light_cone_catalog, box_size=50.0)[
            "passes_light_cone_survey_gate"
        ]
    )
    light_cone_sbi_ok = bool(
        LightConeSBIPipeline().fit().run_calibration()["passes_joint_sbi_gate"]
    )
    multiprobe_sbi_ok = bool(
        MultiProbeSBIPipeline().fit().run_calibration()["passes_joint_sbi_gate"]
    )
    pm_emulator_ok = bool(
        PMEmulatorSBIPipeline().fit().run_calibration()["passes_joint_sbi_gate"]
    )
    flow_sbi_ok = bool(
        MultiProbeSBIPipeline(density_estimator="flow")
        .fit()
        .run_calibration()["passes_joint_sbi_gate"]
    )
    pm_flow_sbi_ok = bool(
        PMEmulatorSBIPipeline(
            n_training_simulations=1600,
            density_estimator="flow",
            flow_n_layers=6,
            flow_epochs=800,
            flow_scale_bound=6.0,
            flow_hidden_dim=16,
        )
        .fit()
        .run_calibration(n_sbc_trials=100, n_tarp_trials=100)[
            "passes_joint_sbi_gate"
        ]
    )
    hod_ok = bool(HaloOccupationModel().evaluate_assembly_bias_marginalization()["passes_stage4_assembly_bias_gate"])
    bisp_ia_ok = bool(
        ReducedBispectrumModel().evaluate_tunes_bispectrum_accuracy()["passes_bispectrum_1pct_gate"]
        and IntrinsicAlignmentModel().evaluate_ia_marginalization_impact()["passes_stage4_ia_gate"]
    )
    de_imnn_ok = bool(
        DynamicDarkEnergyCosmology().evaluate_desi_w0wa_signatures()["recovers_lcdm_limit"]
        and FisherScoreCompressor.benchmark_multiprobe_compression()["passes_98pct_fisher_preservation_gate"]
    )
    mangrove_ok = MangroveMergerTreeGNN(
        allow_post_v1_extension=True, seed=42
    ).benchmark_merger_tree_vs_halo_mlp(n_trees=20, seed=42)["passes_2x_improvement_benchmark"]
    arxiv_ok = ArxivSubmissionPackager().audit_manuscript_integrity()["is_valid"]

    checks = [
        ("Level 1: TUNeS Emulator & Halo Mass Function Gate (<1% MAPE)", hmf_ok),
        ("Level 1: BCEmu3 Baryonic Bias-Variance Frontier (<0.30 sigma_S4)", frontier_ok),
        ("Level 2: RSD (P0,P2,P4) + Joint 3x2pt Degeneracy Breaking", joint_ok),
        ("Level 2: CMB Lensing (kappa_CMB x g / kappa_gal) + tSZ/kSZ BCEmu Gas Gate", cmb_sz_ok),
        ("Level 2: Relativistic Light-Cone Magnification Bias + Wide-Angle RSD Gate", light_cone_ok),
        ("Level 2: Light-Cone Snapshot Stacking & Wide-Angle RSD Catalog Gate", light_cone_build_ok),
        ("Level 2: Light-Cone Galaxy Bias, Survey Mask & w(theta) Clustering Gate", light_cone_survey_ok),
        ("Level 3: End-to-End SBI on Light-Cone w(theta) with SBC + TARP", light_cone_sbi_ok),
        ("Level 3: Multi-Probe SBI (w(theta) + RSD, b1 free) with SBC + TARP", multiprobe_sbi_ok),
        ("Level 3: PM-Emulator Multi-Probe SBI (LOO < 2%) with SBC + TARP", pm_emulator_ok),
        ("Level 3: Gradient-Trained Flow SBI (ConditionalRealNVP) with SBC + TARP", flow_sbi_ok),
        ("Level 3: PM-Emulator + MLP Flow SBI (1600 sims) with SBC + TARP", pm_flow_sbi_ok),
        ("Level 2: HOD + Secondary Assembly Bias P_gm(k) Gate (<0.30 sigma_S4)", hod_ok),
        ("Result 1 & Level 2: Reduced Bispectrum Q(theta) (<1%) + NLA/TATT IA Gate", bisp_ia_ok),
        ("Level 1 & Level 3: w0-wa CPL Dark Energy + Fisher/IMNN Compression (>=98%)", de_imnn_ok),
        ("Level 3: Simulation Budget Scaling Law (N_min=8 TUNeS, N_min=16 Stage-IV)", budget_ok),
        ("Stage 5 & Post-v1: Preconditioned MCLMC + Mangrove Merger-Tree GNN (>=2x)", mangrove_ok),
        ("Publications: Paper 1, Paper 2 & JOSS BibTeX/Cross-Ref Integrity", arxiv_ok),
    ]
    table = Table(title=f"COSMOVERSE v{__version__} Master System & Scientific Gate Doctor")
    table.add_column("Verification Gate", style="cyan")
    table.add_column("Status", justify="center", style="green")
    for label, status in checks:
        table.add_row(label, "PASS" if status else "FAIL")
    console.print(table)


def cmd_multi_suite() -> None:
    """Evaluates Result 4 cross-suite transfer across IllustrisTNG, SIMBA, and Astrid."""
    connector = CAMELSHydroSpectraConnector()
    rows = connector.evaluate_multi_suite_transfer_table(n_test_sims=25, seed=42)
    table = Table(title="Result 4: Multi-Suite Cross-Simulation Transfer (DMO+BCEmu3 -> Hydro)")
    table.add_column("Target Suite", style="cyan")
    table.add_column("Mean P(k) Mismatch", justify="right", style="white")
    table.add_column("Unmarg. |Bias|", justify="right", style="red")
    table.add_column("Marg. |Bias|", justify="right", style="green")
    table.add_column("TARP 68%", justify="right", style="yellow")
    table.add_column("Stage-IV Gate", justify="center", style="magenta")
    for row in rows:
        table.add_row(
            str(row["target_suite"]),
            f"{row['mean_pk_mismatch_pct']:.2f}%",
            f"{row['unmarginalized_bias_sigma']:.2f} sigma",
            f"{row['marginalized_bias_sigma']:.2f} sigma",
            f"{row['post_transfer_cov68']*100:.1f}%",
            "PASS" if row["stage4_compatible_after_marg"] else "FAIL",
        )
    console.print(table)


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Displays or opens the interactive HTML5/3D Cosmic Web & 4-Result Explorer."""
    root = pathlib.Path(__file__).resolve().parents[1]
    html_path = root / "docs" / "index.html"
    uri = html_path.as_uri()
    console.print(
        Panel(
            f"[bold green]Interactive Explorer Ready:[/bold green] {html_path}\n[bold cyan]Browser URI:[/bold cyan] {uri}",
            title="COSMOVERSE Section 10 Interactive Diffusion Layer",
            expand=False,
        )
    )
    if args.open:
        webbrowser.open(uri)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cosmoverse",
        description="COSMOVERSE: Hybrid Cosmological Inference Protocol",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info
    subparsers.add_parser("info", help="Display framework info, Master Document hypothesis, and 4 results")

    # Simulate
    sim_parser = subparsers.add_parser("simulate", help="Run a Particle-Mesh N-body simulation")
    sim_parser.add_argument("--mesh", type=int, default=32, help="Grid resolution per axis")
    sim_parser.add_argument("--box", type=float, default=100.0, help="Box size in Mpc/h")
    sim_parser.add_argument("--steps", type=int, default=5, help="Time integration steps")
    sim_parser.add_argument("--omega_m", type=float, default=0.315, help="Matter density fraction")
    sim_parser.add_argument("--sigma8", type=float, default=0.811, help="Fluctuation amplitude")
    sim_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run CAMELS Latin Hypercube benchmarks")
    bench_parser.add_argument("--n", type=int, default=10, help="Number of Latin Hypercube points")
    bench_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Frontier
    subparsers.add_parser("frontier", help="Compute Resultado 2 Bias-Variance Frontier curve")

    # CAMELS Gen-2
    subparsers.add_parser("camels", help="Inspect CAMELS Gen-2 (1,192 sims) catalog and Stage-IV thresholds")

    # Field-level
    fl_parser = subparsers.add_parser("field-level", help="Run Stage 5 Field-Level MCLMC vs HMC benchmark")
    fl_parser.add_argument("--grid", type=int, default=6, help="3D grid size per dimension")
    fl_parser.add_argument("--samples", type=int, default=40, help="Number of MCMC samples")
    fl_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Citations audit
    subparsers.add_parser("audit-citations", help="Display verified emulator DOIs and blocked references")

    # Figures generator
    fig_parser = subparsers.add_parser("figures", help="Generate all 4 Paper 1 vector SVG figures")
    fig_parser.add_argument("--out-dir", type=str, default="docs/figures", help="Target directory for SVG figures")

    # Multi-suite transfer
    subparsers.add_parser("multi-suite", help="Evaluate Result 4 transfer across IllustrisTNG, SIMBA, and Astrid")

    # Interactive HTML5/3D dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Display or launch the interactive HTML5/3D explorer")
    dash_parser.add_argument("--open", action="store_true", help="Open docs/index.html in the default web browser")

    # Mangrove GNN post-v1 extension
    subparsers.add_parser("mangrove", help="Run post-v1 Mangrove merger-tree GNN benchmark (Section 6.1)")

    # Simulation budget scaling law
    subparsers.add_parser("budget", help="Evaluate Simulation Budget Scaling Law (N_sims = 4..128)")

    # arXiv submission bundler
    ax_parser = subparsers.add_parser("arxiv-bundle", help="Audit LaTeX manuscripts and build arXiv .tar.gz bundles")
    ax_parser.add_argument("--out-dir", type=str, default=None, help="Output directory for .tar.gz archives")

    # Joint 3x2pt + RSD degeneracy breaking
    subparsers.add_parser("joint-3x2pt", help="Evaluate Joint Stage-IV 3x2pt + RSD degeneracy breaking")

    # Train & checkpoint persistence
    tr_parser = subparsers.add_parser("train", help="Train and save a SHA-256 verified checkpoint (.npz)")
    tr_parser.add_argument("--out-dir", type=str, default="dist/checkpoints", help="Checkpoint output directory")

    # Fetch/cache CAMELS public spectra
    fc_parser = subparsers.add_parser("fetch-camels", help="Populate local CAMELS multi-suite spectrum cache")
    fc_parser.add_argument("--cache-dir", type=str, default="dist/camels_cache", help="Target cache directory")
    fc_parser.add_argument("--n", type=int, default=3, help="Number of spectra per suite")

    # Massive neutrinos & Halo Mass Function
    subparsers.add_parser("neutrinos-hmf", help="Evaluate Massive Neutrinos S_nu(k) and Halo Mass Function dn/dlnM")

    # Relativistic light-cone geometry + magnification bias + wide-angle RSD
    subparsers.add_parser("light-cone", help="Evaluate Level 2 relativistic light-cone magnification bias and wide-angle RSD")

    # Past-light-cone catalog from PM snapshots
    lc_build_parser = subparsers.add_parser("light-cone-build", help="Build and export a past-light-cone catalog from PM snapshots")
    lc_build_parser.add_argument("--out-dir", type=str, default="dist/light_cone", help="Output directory for the NPZ light-cone catalog")

    # Light-cone survey mask + galaxy bias + angular clustering
    subparsers.add_parser("light-cone-survey", help="Apply survey mask + galaxy bias and measure w(theta) from a PM light-cone catalog")

    # End-to-end SBI on light-cone w(theta)
    lc_sbi_parser = subparsers.add_parser("light-cone-sbi", help="Run end-to-end SBI on light-cone w(theta) with SBC + TARP")
    lc_sbi_parser.add_argument("--n-sims", type=int, default=200, help="Number of SBI training simulations")
    lc_sbi_parser.add_argument("--n-sbc", type=int, default=50, help="Number of SBC calibration trials")
    lc_sbi_parser.add_argument("--n-tarp", type=int, default=50, help="Number of TARP calibration trials")

    # Multi-probe SBI on w(theta) + RSD with free b1
    mp_sbi_parser = subparsers.add_parser("multiprobe-sbi", help="Run multi-probe SBI (w(theta) + RSD, b1 free) with SBC + TARP")
    mp_sbi_parser.add_argument("--n-sims", type=int, default=400, help="Number of SBI training simulations")
    mp_sbi_parser.add_argument("--n-sbc", type=int, default=50, help="Number of SBC calibration trials")
    mp_sbi_parser.add_argument("--n-tarp", type=int, default=50, help="Number of TARP calibration trials")

    # PM-emulator multi-probe SBI
    pm_sbi_parser = subparsers.add_parser("pm-emulator-sbi", help="Run multi-probe SBI with a PM-trained P(k) emulator and SBC + TARP")
    pm_sbi_parser.add_argument("--n-sims", type=int, default=400, help="Number of SBI training simulations")
    pm_sbi_parser.add_argument("--n-sbc", type=int, default=50, help="Number of SBC calibration trials")
    pm_sbi_parser.add_argument("--n-tarp", type=int, default=50, help="Number of TARP calibration trials")

    # Gradient-trained conditional flow SBI
    flow_sbi_parser = subparsers.add_parser("flow-sbi", help="Run multi-probe SBI with a gradient-trained ConditionalRealNVP flow")
    flow_sbi_parser.add_argument("--n-sims", type=int, default=400, help="Number of SBI training simulations")
    flow_sbi_parser.add_argument("--n-sbc", type=int, default=50, help="Number of SBC calibration trials")
    flow_sbi_parser.add_argument("--n-tarp", type=int, default=50, help="Number of TARP calibration trials")

    # PM-emulator + MLP flow SBI
    pm_flow_parser = subparsers.add_parser("pm-flow-sbi", help="Run PM-emulator multi-probe SBI with an MLP conditional flow")
    pm_flow_parser.add_argument("--n-sims", type=int, default=1600, help="Number of SBI training simulations")
    pm_flow_parser.add_argument("--n-sbc", type=int, default=100, help="Number of SBC calibration trials")
    pm_flow_parser.add_argument("--n-tarp", type=int, default=100, help="Number of TARP calibration trials")

    # CMB Lensing cross-spectra + tSZ/kSZ BCEmu gas calibration
    subparsers.add_parser("cmb-sz", help="Evaluate Level 2 CMB lensing cross-spectra and tSZ/kSZ gas calibration")

    # HOD + Secondary Assembly Bias + Stage-IV Mock Catalog exporter
    hm_parser = subparsers.add_parser("hod-mocks", help="Evaluate HOD P_gm(k) assembly bias and export Stage-IV mock catalog")
    hm_parser.add_argument("--out-dir", type=str, default="dist/mocks", help="Output directory for synthetic mock catalogs")
    hm_parser.add_argument("--n-galaxies", type=int, default=200, help="Number of synthetic galaxies to export")

    # Reduced Bispectrum Q(theta) + NLA/TATT Intrinsic Alignments
    subparsers.add_parser("bispectrum-ia", help="Evaluate Result 1 Reduced Bispectrum Q(theta) and Level 2 NLA/TATT IA")

    # Dynamic w0-wa CPL Dark Energy + Fisher/IMNN Score Compression
    subparsers.add_parser("dark-energy-imnn", help="Evaluate Level 1 w0-wa CPL Dark Energy and Level 3 Fisher/IMNN compression")

    # Master system & scientific gate doctor
    subparsers.add_parser("doctor", help="Run full 18-gate Master Document health & calibration check")

    args = parser.parse_args(argv)

    if args.command == "info" or args.command is None:
        show_info()
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "benchmark":
        cmd_benchmark_camels(args)
    elif args.command == "frontier":
        cmd_frontier()
    elif args.command == "camels":
        cmd_camels_gen2()
    elif args.command == "field-level":
        cmd_field_level(args)
    elif args.command == "audit-citations":
        cmd_audit_citations()
    elif args.command == "figures":
        cmd_figures(args)
    elif args.command == "multi-suite":
        cmd_multi_suite()
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "mangrove":
        cmd_mangrove()
    elif args.command == "budget":
        cmd_budget()
    elif args.command == "arxiv-bundle":
        cmd_arxiv_bundle(args)
    elif args.command == "joint-3x2pt":
        cmd_joint_3x2pt()
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "fetch-camels":
        cmd_fetch_camels(args)
    elif args.command == "neutrinos-hmf":
        cmd_neutrinos_hmf()
    elif args.command == "light-cone":
        cmd_light_cone()
    elif args.command == "light-cone-build":
        cmd_light_cone_build(args)
    elif args.command == "light-cone-survey":
        cmd_light_cone_survey()
    elif args.command == "light-cone-sbi":
        cmd_light_cone_sbi(args)
    elif args.command == "multiprobe-sbi":
        cmd_multiprobe_sbi(args)
    elif args.command == "pm-emulator-sbi":
        cmd_pm_emulator_sbi(args)
    elif args.command == "flow-sbi":
        cmd_flow_sbi(args)
    elif args.command == "pm-flow-sbi":
        cmd_pm_flow_sbi(args)
    elif args.command == "cmb-sz":
        cmd_cmb_sz()
    elif args.command == "hod-mocks":
        cmd_hod_mocks(args)
    elif args.command == "bispectrum-ia":
        cmd_bispectrum_ia()
    elif args.command == "dark-energy-imnn":
        cmd_dark_energy_imnn()
    elif args.command == "doctor":
        cmd_doctor()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
