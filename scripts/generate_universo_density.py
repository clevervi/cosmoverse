"""Generate the embedded 80^3 cosmic-web density field for docs/universo.html.

The field is a fast Zel'dovich-approximation realisation of a periodic CDM box,
not a hydrodynamical simulation. It replaces the hand-drawn node/filament graph
with a physically motivated density field used by the volumetric and point
layers.
"""

from __future__ import annotations

import base64
import pathlib

import numpy as np

N = 96
BOX = 200.0
N_PARTICLES = 520_000
SEED = 20261102


def periodic_sample(field: np.ndarray, coords: np.ndarray) -> np.ndarray:
    grid = coords / BOX * N
    i0 = np.floor(grid).astype(np.int64)
    f = grid - i0
    i0 %= N
    i1 = (i0 + 1) % N
    result = np.zeros(coords.shape[0], dtype=np.float64)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1.0 - f[:, 0]
        ix = i1[:, 0] if dx else i0[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1.0 - f[:, 1]
            iy = i1[:, 1] if dy else i0[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1.0 - f[:, 2]
                iz = i1[:, 2] if dz else i0[:, 2]
                result += field[ix, iy, iz] * wx * wy * wz
    return result


def deposit_cic(positions: np.ndarray) -> np.ndarray:
    grid = positions / BOX * N
    i0 = np.floor(grid).astype(np.int64)
    f = grid - i0
    i0 %= N
    i1 = (i0 + 1) % N
    counts = np.zeros((N, N, N), dtype=np.float64)
    for dx in (0, 1):
        wx = f[:, 0] if dx else 1.0 - f[:, 0]
        ix = i1[:, 0] if dx else i0[:, 0]
        for dy in (0, 1):
            wy = f[:, 1] if dy else 1.0 - f[:, 1]
            iy = i1[:, 1] if dy else i0[:, 1]
            for dz in (0, 1):
                wz = f[:, 2] if dz else 1.0 - f[:, 2]
                iz = i1[:, 2] if dz else i0[:, 2]
                np.add.at(counts, (ix, iy, iz), wx * wy * wz)
    return counts


def main() -> None:
    rng = np.random.default_rng(SEED)
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=BOX / N)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=BOX / N)
    kz = 2.0 * np.pi * np.fft.fftfreq(N, d=BOX / N)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    k = np.sqrt(k2)
    k[0, 0, 0] = 1.0
    k2[0, 0, 0] = 1.0

    # CDM-like broad-band power spectrum with a smooth small-scale cutoff.
    power = (k / 0.085) ** -2.10 * np.exp(-((k / 0.55) ** 2))
    power[0, 0, 0] = 0.0

    delta_k = (
        rng.normal(size=(N, N, N)) + 1j * rng.normal(size=(N, N, N))
    ) * np.sqrt(power / 2.0)
    delta = np.fft.ifftn(delta_k).real
    delta /= delta.std()
    delta -= delta.mean()

    # Zel'dovich displacement: Psi_k = i k / k^2 delta_k.
    psi_k = 1j * np.stack([kx, ky, kz], axis=0) / k2 * delta_k
    psi = np.fft.ifftn(psi_k, axes=(1, 2, 3)).real
    psi *= 4.2 / psi.std()

    q = rng.uniform(0.0, BOX, size=(N_PARTICLES, 3))
    displacement = np.stack(
        [
            periodic_sample(psi[0], q),
            periodic_sample(psi[1], q),
            periodic_sample(psi[2], q),
        ],
        axis=1,
    )
    positions = (q + displacement) % BOX

    counts = deposit_cic(positions)
    density = counts / (N_PARTICLES / N**3) - 1.0

    # Mild smoothing removes CIC shot noise without erasing filaments.
    density_k = np.fft.fftn(density)
    sigma = 0.90 * BOX / N
    density_k *= np.exp(-0.5 * k2 * sigma**2)
    density = np.fft.ifftn(density_k).real

    low, high = np.percentile(density, [1.0, 99.5])
    field = np.clip((density - low) / (high - low), 0.0, 1.0)
    field = field**0.85

    data = (field * 255.0).astype(np.uint8)
    encoded = base64.b64encode(data.tobytes()).decode("ascii")
    out = pathlib.Path(__file__).resolve().parents[1] / "docs" / "universo_density.js"
    out.write_text(
        "window.COSMOVerseDensity = {\n"
        f'  size: {N},\n'
        '  source: "Zel\'dovich N-body 80^3, periodic 200 Mpc/h",\n'
        f'  data: "{encoded}"\n'
        "};\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"field shape={field.shape} min={field.min():.4f} max={field.max():.4f}")
    print(f"percentiles 50/90/99 = {np.percentile(field, [50, 90, 99])}")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
