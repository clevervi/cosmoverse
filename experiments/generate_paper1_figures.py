"""
Generates the 4 publication-ready SVG figures for COSMOVERSE Paper 1
(Results 1, 2, 3, and 4) into `docs/figures/`.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cosmoverse.validation.plotting import Paper1FigureGenerator  # noqa: E402
from experiments.run_paper1_mvp_protocol import run_mvp_protocol  # noqa: E402


def main() -> None:
    report = run_mvp_protocol()
    out_dir = ROOT / "docs" / "figures"
    saved = Paper1FigureGenerator.generate_all_paper1_figures(out_dir, protocol_report=report)
    print("\nGenerated Paper 1 Scientific Figures:")
    for key, path in saved.items():
        print(f"  * {key}: {path}")


if __name__ == "__main__":
    main()
