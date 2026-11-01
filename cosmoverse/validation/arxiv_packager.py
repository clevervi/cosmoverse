"""
Automated arXiv (`astro-ph.CO`) Submission Bundler & LaTeX Cross-Reference Validator
for COSMOVERSE Paper 1 (Stages 1-4 MVP) and Paper 2 (Stage 5 Field-Level Extension).

Performs strict pre-submission checks:
  1. Verifies that all `\\input{...}` tables exist on disk.
  2. Verifies that every `\\cite{...}` key in `.tex` exists in `paper/references.bib`.
  3. Verifies that no blocked/unverified citations (e.g. `CosmoRGNet` as a bibliographic key)
     appear in `paper/references.bib`.
  4. Packages clean `.tar.gz` bundles ready for arXiv upload.
"""

import io
import pathlib
import re
import tarfile
from typing import Any

from cosmoverse.emulators.fno import EMULATOR_BIBLIOGRAPHY_REGISTRY


class ArxivSubmissionPackager:
    """
    Validates LaTeX manuscripts, tables, and BibTeX references, and builds deterministic
    `.tar.gz` submission bundles for arXiv (`astro-ph.CO`).
    """

    def __init__(self, repo_root: str | pathlib.Path | None = None):
        self.repo_root = (
            pathlib.Path(repo_root)
            if repo_root is not None
            else pathlib.Path(__file__).resolve().parents[2]
        )
        self.paper_dir = self.repo_root / "paper"
        self.figures_dir = self.repo_root / "docs" / "figures"

    def audit_manuscript_integrity(self) -> dict[str, Any]:
        """
        Audits `paper/main.tex`, `paper/paper2_field_level.tex`, and `paper/references.bib`
        for missing citations, missing table inputs, or blocked references.
        """
        bib_path = self.paper_dir / "references.bib"
        bib_text = bib_path.read_text(encoding="utf-8")
        defined_bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_text))

        # Verify blocked emulator keys are not present as BibTeX entries
        blocked_keys = {
            k
            for k, meta in EMULATOR_BIBLIOGRAPHY_REGISTRY.items()
            if meta.get("status") == "UNVERIFIED_BLOCKED"
        }
        leaked_blocked = sorted(
            k for k in blocked_keys if any(k.lower() in bk.lower() for bk in defined_bib_keys)
        )

        manuscripts = {
            "paper1": self.paper_dir / "main.tex",
            "paper2": self.paper_dir / "paper2_field_level.tex",
        }
        missing_citations: dict[str, list[str]] = {}
        missing_inputs: dict[str, list[str]] = {}
        cited_by_paper: dict[str, list[str]] = {}

        for key, tex_path in manuscripts.items():
            tex = tex_path.read_text(encoding="utf-8")
            raw_cites = re.findall(r"\\cite\{([^}]+)\}", tex)
            cites: set[str] = set()
            for group in raw_cites:
                for item in group.split(","):
                    if item.strip():
                        cites.add(item.strip())
            cited_by_paper[key] = sorted(cites)
            missing_citations[key] = sorted(cites - defined_bib_keys)

            inputs = re.findall(r"\\input\{([^}]+)\}", tex)
            missing_in = []
            for rel_in in inputs:
                candidate = self.paper_dir / rel_in
                if not candidate.exists():
                    missing_in.append(rel_in)
            missing_inputs[key] = missing_in

        # Also validate JOSS paper (`paper/paper.md`) if present
        joss_path = self.paper_dir / "paper.md"
        joss_missing: list[str] = []
        if joss_path.exists():
            joss_text = joss_path.read_text(encoding="utf-8")
            joss_cites = set(re.findall(r"@([A-Za-z0-9_]+)", joss_text))
            cited_by_paper["joss"] = sorted(joss_cites)
            joss_missing = sorted(joss_cites - defined_bib_keys)
            missing_citations["joss"] = joss_missing

        is_valid = (
            len(leaked_blocked) == 0
            and all(len(v) == 0 for v in missing_citations.values())
            and all(len(v) == 0 for v in missing_inputs.values())
        )

        return {
            "is_valid": is_valid,
            "defined_bib_keys": sorted(defined_bib_keys),
            "cited_by_paper": cited_by_paper,
            "missing_citations": missing_citations,
            "missing_inputs": missing_inputs,
            "leaked_blocked_keys": leaked_blocked,
        }

    def build_arxiv_bundles(
        self, out_dir: str | pathlib.Path | None = None
    ) -> dict[str, Any]:
        """
        Audits manuscripts and builds `cosmoverse_paper1_arxiv.tar.gz` and
        `cosmoverse_paper2_arxiv.tar.gz` in `out_dir` (defaults to `dist/arxiv`).
        """
        audit = self.audit_manuscript_integrity()
        if not audit["is_valid"]:
            raise ValueError(f"LaTeX manuscript audit failed: {audit}")

        target_dir = (
            pathlib.Path(out_dir)
            if out_dir is not None
            else (self.repo_root / "dist" / "arxiv")
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        p1_tar = target_dir / "cosmoverse_paper1_arxiv.tar.gz"
        p2_tar = target_dir / "cosmoverse_paper2_arxiv.tar.gz"

        self._write_tar_bundle(
            p1_tar,
            main_tex=self.paper_dir / "main.tex",
            bib_file=self.paper_dir / "references.bib",
            tables=[
                self.paper_dir / "tables" / "table_results_summary.tex",
                self.paper_dir / "tables" / "table_suite_transfer.tex",
                self.paper_dir / "tables" / "table_simulation_budget.tex",
            ],
            figures=[
                self.figures_dir / "fig1_result1_emulator.svg",
                self.figures_dir / "fig2_result2_frontier.svg",
                self.figures_dir / "fig3_result3_calibration.svg",
                self.figures_dir / "fig4_result4_transfer.svg",
                self.figures_dir / "fig7_simulation_budget_scaling.svg",
            ],
        )

        self._write_tar_bundle(
            p2_tar,
            main_tex=self.paper_dir / "paper2_field_level.tex",
            bib_file=self.paper_dir / "references.bib",
            tables=[
                self.paper_dir / "tables" / "table_paper2_field_level.tex",
            ],
            figures=[
                self.figures_dir / "fig5_paper2_ic_reconstruction.svg",
                self.figures_dir / "fig6_paper2_mclmc_vs_hmc.svg",
            ],
        )

        return {
            "audit": audit,
            "paper1_bundle": str(p1_tar),
            "paper2_bundle": str(p2_tar),
        }

    @staticmethod
    def _write_tar_bundle(
        tar_path: pathlib.Path,
        main_tex: pathlib.Path,
        bib_file: pathlib.Path,
        tables: list[pathlib.Path],
        figures: list[pathlib.Path],
    ) -> None:
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(main_tex, arcname="main.tex")
            tf.add(bib_file, arcname="references.bib")
            for tbl in tables:
                if tbl.exists():
                    tf.add(tbl, arcname=f"tables/{tbl.name}")
            for fig in figures:
                if fig.exists():
                    tf.add(fig, arcname=f"figures/{fig.name}")
            readme_bytes = (
                "COSMOVERSE arXiv (astro-ph.CO) Submission Bundle\n"
                "Compile with: pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex\n"
            ).encode("utf-8")
            info = tarfile.TarInfo(name="00README.txt")
            info.size = len(readme_bytes)
            tf.addfile(info, io.BytesIO(readme_bytes))
