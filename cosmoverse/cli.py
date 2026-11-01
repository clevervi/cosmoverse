"""
COSMOVERSE Command Line Interface (CLI).
Provides commands for running simulations, neural emulators (TUNeS), BCEmu bias-variance
frontiers, observational forward pipelines, SBI inference, and SBC/TARP/Transfer validation.
"""

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cosmoverse import __author__, __email__, __version__
from cosmoverse.baryons.frontier import BiasVarianceFrontier
from cosmoverse.core.cosmology import Cosmology
from cosmoverse.core.parameters import CosmologicalParameters, SimulationParameters
from cosmoverse.simulation.nbody import ParticleMeshSimulator
from cosmoverse.validation.benchmarks import camels_latin_hypercube_benchmark

console = Console()


def show_info() -> None:
    """Displays COSMOVERSE framework metadata, Master Document hypothesis, and roadmap stages."""
    title = "[bold cyan]COSMOVERSE[/bold cyan]: Hybrid Cosmological Inference Protocol"
    subtitle = "TUNeS Emulation · BCEmu Baryonic Marginalization · Stage-IV Realism · Calibrated SBI & MCLMC"
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
 • [bold]Result 1[/bold]: Emulator precision in Gaussian (P(k) MAPE < 1% at k < 1 h/Mpc) & non-Gaussian (Peaks, 3D Minkowski)
 • [bold]Result 2[/bold]: Complete Bias-Variance Frontier under BCEmu + observational marginalization vs Stage-IV precision
 • [bold]Result 3[/bold]: SBI posterior calibration via SBC + TARP + scale-dependent misspecification detection
 • [bold]Result 4[/bold]: Quantified coverage & bias degradation under Cross-Simulation Family Transfer (disjoint ICs/cosmology)
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


def main() -> None:
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

    args = parser.parse_args()

    if args.command == "info" or args.command is None:
        show_info()
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "benchmark":
        cmd_benchmark_camels(args)
    elif args.command == "frontier":
        cmd_frontier()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
