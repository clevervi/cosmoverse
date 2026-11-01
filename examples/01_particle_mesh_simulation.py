"""
Example 01: Particle-Mesh (PM) cosmological simulation.
Generates 1LPT initial conditions, evolves particles with leapfrog KDK, and measures P(k).
"""


from cosmoverse.core import CosmologicalParameters, Cosmology, SimulationParameters
from cosmoverse.simulation import ParticleMeshSimulator


def main():
    print("=" * 60)
    print("COSMOVERSE: Example 01 - Particle-Mesh Simulation")
    print("=" * 60)

    # 1. Setup cosmological and simulation parameters
    cosmo_params = CosmologicalParameters(omega_m=0.315, sigma8=0.811, h=0.674)
    sim_params = SimulationParameters(
        box_size=100.0,  # 100 Mpc/h
        n_mesh=32,       # 32^3 grid
        z_initial=50.0,
        z_final=0.0,
        n_steps=5,
        seed=42,
    )

    cosmo = Cosmology(cosmo_params)
    simulator = ParticleMeshSimulator(cosmology=cosmo, sim_params=sim_params)

    # 2. Run simulation
    print("Initializing Zel'dovich (1LPT) particles at z=50...")
    simulator.initialize()
    print(f"Total particles: {len(simulator.positions)}")

    print("\nEvolving gravitational dynamics to z=0...")
    density_grid = simulator.run(verbose=True)
    print(f"Final density grid shape: {density_grid.shape}")
    print(f"Density contrast range: min={density_grid.min():.2f}, max={density_grid.max():.2f}")

    # 3. Estimate matter power spectrum
    k, pk = simulator.compute_power_spectrum(n_bins=8)
    print("\nEstimated 3D Matter Power Spectrum:")
    print(f"{'k [h/Mpc]':>12} | {'P(k) [(Mpc/h)^3]':>18}")
    print("-" * 34)
    for kv, pkv in zip(k, pk, strict=False):
        print(f"{kv:12.4f} | {pkv:18.2e}")


if __name__ == "__main__":
    main()
