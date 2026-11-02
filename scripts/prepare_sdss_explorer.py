"""Prepare the real SDSS galaxy layer for docs/explorador.html.

Reads a CSV produced by the SDSS SkyServer query (ra, dec, z, modelMag_u,
modelMag_g, modelMag_r), converts redshifts to comoving Cartesian positions,
builds observation-like colours from g-r/r and writes a compact base64 layer.

Query used (SDSS DR18):
  SELECT TOP 45000 s.ra, s.dec, s.z, p.modelMag_u, p.modelMag_g, p.modelMag_r
  FROM SpecObj s JOIN PhotoObj p ON s.bestObjID=p.objID
  WHERE s.class='GALAXY' AND s.z BETWEEN 0.005 AND 0.15 AND s.zWarning=0
"""

from __future__ import annotations

import argparse
import base64
import csv
import pathlib

import numpy as np
from scipy.integrate import cumulative_trapezoid

OMEGA_M = 0.3
OMEGA_L = 0.7
H0 = 70.0
C_KMS = 299792.458
C_OVER_H0 = C_KMS / H0  # Mpc
Z_MAX = 0.15
EXTENT_MPC = 750.0


def load_rows(csv_path: pathlib.Path) -> tuple[np.ndarray, ...]:
    rows = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        for row in reader:
            try:
                rows.append(
                    (
                        float(row["ra"]),
                        float(row["dec"]),
                        float(row["z"]),
                        float(row["modelMag_u"]),
                        float(row["modelMag_g"]),
                        float(row["modelMag_r"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    data = np.asarray(rows, dtype=np.float64)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]


def comoving_distance(z: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.0, max(float(z.max()), Z_MAX), 512)
    e = np.sqrt(OMEGA_M * (1.0 + grid) ** 3 + OMEGA_L)
    dc = C_OVER_H0 * cumulative_trapezoid(1.0 / e, grid, initial=0.0)
    return np.interp(z, grid, dc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=pathlib.Path, required=True)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("docs/explorador_sdss.js"),
    )
    args = parser.parse_args()

    ra, dec, z, mag_u, mag_g, mag_r = load_rows(args.csv)
    distance = comoving_distance(z)
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = distance * np.cos(dec_rad) * np.cos(ra_rad)
    y = distance * np.cos(dec_rad) * np.sin(ra_rad)
    zc = distance * np.sin(dec_rad)

    # SDSS wedge: keep a cube large enough for the z<=0.15 sample.
    positions = np.stack([x, y, zc], axis=1)
    positions = np.clip(positions, -EXTENT_MPC, EXTENT_MPC)

    color_index = np.clip((mag_g - mag_r + 0.4) / 1.4, 0.0, 1.0)
    blue = np.array([0.42, 0.66, 1.0])
    red = np.array([1.0, 0.55, 0.32])
    colors = (
        blue[None, :] * (1.0 - color_index[:, None])
        + red[None, :] * color_index[:, None]
    )
    brightness = np.clip((19.5 - mag_r) / 4.0, 0.25, 1.0)
    colors *= brightness[:, None]

    pos_u16 = np.round(
        ((positions + EXTENT_MPC) / (2.0 * EXTENT_MPC) * 65535.0)
    ).astype(np.uint16)
    col_u8 = np.round(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    z_u16 = np.round(z / Z_MAX * 65535.0).astype(np.uint16)

    count = len(ra)
    payload = bytearray()
    for i in range(count):
        payload += pos_u16[i].tobytes()
        payload += col_u8[i].tobytes()
        payload += z_u16[i].tobytes()
    encoded = base64.b64encode(bytes(payload)).decode("ascii")
    args.out.write_text(
        "window.COSMOVerseSDSS = {\n"
        f"  count: {count},\n"
        '  source: "SDSS DR18 spectroscopic galaxies",\n'
        '  selection: "0.005 < z < 0.15, zWarning=0",\n'
        f"  zMax: {Z_MAX},\n"
        f"  extentMpc: {EXTENT_MPC},\n"
        f'  data: "{encoded}"\n'
        "};\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"kept {count} galaxies")
    print(f"z range {z.min():.4f}..{z.max():.4f}, distance {distance.min():.1f}..{distance.max():.1f} Mpc")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
