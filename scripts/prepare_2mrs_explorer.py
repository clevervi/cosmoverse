"""Prepare the all-sky 2MRS galaxy layer for docs/explorador.html.

Reads the HEASARC TDAT version of the 2MASS Redshift Survey (Huchra et al.
2012), keeps galaxies with positive radial velocity, converts velocities to
comoving distances and writes a compact base64 point layer. Radial distances
are gently compressed so the full celestial sphere and the CMB horizon fit in
one normalized scene.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import pathlib

import numpy as np
from scipy.integrate import cumulative_trapezoid

OMEGA_M = 0.3
OMEGA_L = 0.7
H0 = 70.0
C_KMS = 299792.458
C_OVER_H0 = C_KMS / H0
V_MAX = 30000.0
EXTENT = 1.9


def parse_tdat(path: pathlib.Path) -> tuple[np.ndarray, ...]:
    ra, dec, velocity, kmag = [], [], [], []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        in_data = False
        for line in handle:
            if not in_data:
                if line.strip() == "<DATA>":
                    in_data = True
                continue
            fields = line.rstrip("\n").split("|")
            if len(fields) < 25:
                continue
            try:
                v = float(fields[24])
                if v <= 0 or v > V_MAX:
                    continue
                ra.append(float(fields[1]))
                dec.append(float(fields[2]))
                velocity.append(v)
                kmag.append(float(fields[5]))
            except (ValueError, IndexError):
                continue
    return (
        np.asarray(ra),
        np.asarray(dec),
        np.asarray(velocity),
        np.asarray(kmag),
    )


def comoving_distance(z: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.0, max(float(z.max()), 0.1), 512)
    e = np.sqrt(OMEGA_M * (1.0 + grid) ** 3 + OMEGA_L)
    dc = C_OVER_H0 * cumulative_trapezoid(1.0 / e, grid, initial=0.0)
    return np.interp(z, grid, dc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tdat", type=pathlib.Path, required=True)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("docs/explorador_2mrs.js"),
    )
    args = parser.parse_args()

    ra, dec, velocity, kmag = parse_tdat(args.tdat)
    z = velocity / C_KMS
    distance = comoving_distance(z)
    dmax = float(distance.max())
    radius = 0.55 + 1.15 * np.power(distance / dmax, 0.72)

    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = radius * np.cos(dec_rad) * np.cos(ra_rad)
    y = radius * np.cos(dec_rad) * np.sin(ra_rad)
    zc = radius * np.sin(dec_rad)

    brightness = np.clip((11.75 - kmag) / 4.0, 0.15, 1.0)
    blue = np.array([0.42, 0.62, 0.95])
    warm = np.array([1.0, 0.70, 0.34])
    colors = blue[None, :] * (1.0 - brightness[:, None]) + warm[None, :] * brightness[:, None]

    positions = np.stack([x, y, zc], axis=1)
    pos_u16 = np.round(
        ((positions + EXTENT) / (2.0 * EXTENT) * 65535.0)
    ).astype(np.uint16)
    col_u8 = np.round(np.clip(colors, 0, 1) * 255.0).astype(np.uint8)
    vel_u16 = np.round(velocity / V_MAX * 65535.0).astype(np.uint16)

    payload = bytearray()
    for i in range(len(ra)):
        payload += pos_u16[i].tobytes()
        payload += col_u8[i].tobytes()
        payload += vel_u16[i].tobytes()
    encoded = base64.b64encode(bytes(payload)).decode("ascii")
    args.out.write_text(
        "window.COSMOVerse2MRS = {\n"
        f"  count: {len(ra)},\n"
        '  source: "2MASS Redshift Survey (2MRS), Huchra et al. 2012",\n'
        f"  vMax: {V_MAX},\n"
        f"  extent: {EXTENT},\n"
        f"  maxDistanceMpc: {dmax:.2f},\n"
        '  radialNote: "radial distances compressed for a single normalized view",\n'
        f'  data: "{encoded}"\n'
        "};\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"kept {len(ra)} galaxies; vmax={velocity.max():.0f} km/s; dmax={dmax:.1f} Mpc")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
