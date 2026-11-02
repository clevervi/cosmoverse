"""Prepare the embedded CAMELS-SIMBA gas field for docs/universo.html.

Downloads (if needed) the public CMD 3D grids for gas density (Mgas) and
mass-weighted gas temperature (T) for the SIMBA EX set at z=0, takes the
fiducial realisation (grid 0), downsamples 128^3 -> 96^3 and writes a compact
two-channel base64 field for the WebGL page.

Data: CAMELS Multifield Dataset, Villasenor et al. 2023.
URL:  https://camels-multifield-dataset.readthedocs.io/
"""

from __future__ import annotations

import argparse
import base64
import pathlib
import urllib.request

import numpy as np
from scipy.ndimage import zoom

BASE_URL = (
    "https://users.flatironinstitute.org/~fvillaescusa/priv/"
    "DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/3D_grids/data/SIMBA"
)
FILES = {
    "mgas": "Grids_Mgas_SIMBA_EX_128_z=0.0.npy",
    "temperature": "Grids_T_SIMBA_EX_128_z=0.0.npy",
}
GRID = 96
REALIZATION = 1
PARAMETERS = [
    "Omega_m=0.3, sigma8=0.8, A_SN1=1, A_AGN1=1, A_SN2=1, A_AGN2=1",
    "Omega_m=0.3, sigma8=0.8, A_SN1=1, A_AGN1=100, A_SN2=1, A_AGN2=1",
    "Omega_m=0.3, sigma8=0.8, A_SN1=100, A_AGN1=1, A_SN2=1, A_AGN2=1",
    "Omega_m=0.3, sigma8=0.8, A_SN1=0, A_AGN1=0, A_SN2=0, A_AGN2=0",
]


def download(cache_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, pathlib.Path] = {}
    for key, name in FILES.items():
        path = cache_dir / name
        if not path.exists():
            print(f"downloading {name}...")
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)
        paths[key] = path
    return paths


def normalize(values: np.ndarray, low: float, high: float) -> np.ndarray:
    lo, hi = np.percentile(values, [low, high])
    normalized = np.clip((values - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path("dist/camels"))
    parser.add_argument("--mgas", type=pathlib.Path, default=None)
    parser.add_argument("--temperature", type=pathlib.Path, default=None)
    parser.add_argument("--realization", type=int, default=REALIZATION)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("docs/universo_camels.js"),
    )
    args = parser.parse_args()

    if args.mgas and args.temperature:
        paths = {"mgas": args.mgas, "temperature": args.temperature}
    else:
        paths = download(args.cache_dir)

    mgas = np.load(paths["mgas"], mmap_mode="r")
    temperature = np.load(paths["temperature"], mmap_mode="r")
    density = np.asarray(mgas[args.realization], dtype=np.float64)
    temp = np.asarray(temperature[args.realization], dtype=np.float64)

    density = np.log10(np.maximum(density, 1e-6))
    temp = np.log10(np.maximum(temp, 1e2))

    factor = GRID / density.shape[0]
    density = zoom(density, factor, order=1)
    temp = zoom(temp, factor, order=1)

    density = normalize(density, 1.0, 99.8)
    temp = normalize(temp, 2.0, 99.5)
    density = density**0.85

    channels = np.concatenate(
        [
            (density * 255.0).astype(np.uint8).ravel(),
            (temp * 255.0).astype(np.uint8).ravel(),
        ]
    )
    encoded = base64.b64encode(channels.tobytes()).decode("ascii")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "window.COSMOVerseCAMELS = {\n"
        f"  size: {GRID},\n"
        '  source: "CAMELS-SIMBA EX, Mgas + T, 128^3 -> 96^3, z=0",\n'
        '  simulation: "SIMBA",\n'
        '  channels: ["Mgas", "T"],\n'
        f'  params: "{PARAMETERS[args.realization]}",\n'
        f'  data: "{encoded}"\n'
        "};\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"density range {density.min():.3f}..{density.max():.3f}")
    print(f"temperature range {temp.min():.3f}..{temp.max():.3f}")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
