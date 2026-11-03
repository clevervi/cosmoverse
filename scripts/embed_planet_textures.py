"""Embed the planet texture atlas as local data URIs for the explorer."""

from __future__ import annotations

import base64
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "planetas"
OUT = ROOT / "docs" / "explorador_planetas.js"

KEYS = (
    "mercury",
    "venus",
    "earth",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
    "moon",
    "sun",
)


def main() -> None:
    entries = []
    for key in KEYS:
        path = SOURCE / f"{key}.jpg"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        entries.append(f'    {key}: "data:image/jpeg;base64,{payload}"')
    OUT.write_text(
        "window.COSMOVersePlanets = {\n"
        '  source: "Solar System Scope planet textures, CC BY 4.0",\n'
        "  textures: {\n"
        + ",\n".join(entries)
        + "\n  }\n};\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
