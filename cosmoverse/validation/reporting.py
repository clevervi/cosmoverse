"""
Structured JSON and Markdown Benchmark Report Generator for all 4 Demonstrable Results
(Master Document Section 4 & Section 13).
"""

import json
import pathlib
from typing import Any


class ScientificBenchmarkReporter:
    """
    Aggregates, validates, and exports the 4 Demonstrable Results of COSMOVERSE
    into auditable JSON and publication-ready Markdown summary tables.
    """

    def __init__(
        self,
        result1_emulator: dict[str, Any],
        result2_frontier: dict[str, Any],
        result3_calibration: dict[str, Any],
        result4_transfer: dict[str, Any],
        partition_verification: dict[str, Any],
    ):
        self.result1 = result1_emulator
        self.result2 = result2_frontier
        self.result3 = result3_calibration
        self.result4 = result4_transfer
        self.partition = partition_verification

    def evaluate_all_acceptance_gates(self) -> dict[str, bool]:
        """Checks every quantitative acceptance criterion from Section 13 of MASTER_DOCUMENT.md."""
        gate1 = bool(self.result1.get("passes_pk_threshold", False))
        gate2 = bool(self.result2.get("has_stage4_compatible_regime", False))
        gate3 = bool(self.result3.get("tarp", {}).get("passes_acceptance_gate", False)) and (
            not bool(self.result3.get("misspecification", {}).get("is_misspecified", True))
        )
        gate4 = bool(self.partition.get("is_leakage_free", False)) and bool(
            self.result4.get("is_transfer_bounded", False)
        )
        return {
            "result1_emulator_gate": gate1,
            "result2_frontier_gate": gate2,
            "result3_calibration_gate": gate3,
            "result4_transfer_gate": gate4,
            "all_gates_passed": bool(gate1 and gate2 and gate3 and gate4),
        }

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable dictionary of all 4 Demonstrable Results and gate status."""
        return {
            "acceptance_gates": self.evaluate_all_acceptance_gates(),
            "partition_verification": self.partition,
            "result1_emulator": self.result1,
            "result2_bias_variance_frontier": self.result2,
            "result3_sbi_calibration": self.result3,
            "result4_cross_simulation_transfer": self.result4,
        }

    def format_markdown_report(self) -> str:
        """Generates a Markdown report summarizing all 4 Demonstrable Results."""
        gates = self.evaluate_all_acceptance_gates()
        opt = self.result2["optimal_operating_point"]
        tarp = self.result3["tarp"]
        pre = self.result4["pre_transfer"]
        post = self.result4["post_transfer"]
        deg = self.result4["degradation"]

        lines = [
            "# COSMOVERSE — Reporte de Validación Científica (Paper 1 MVP)",
            "",
            "## Resumen de Criterios de Aceptación Bloqueantes (Sección 13)",
            "",
            "| Resultado | Criterio Cuantitativo | Valor Obtenido | Estado |",
            "| :--- | :--- | :--- | :--- |",
            (
                f"| **Resultado 1 (Emulador TUNeS)** | MAPE en $P(k) < 1\\%$ ($k \\le 1\\,h\\,\\mathrm{{Mpc}}^{{-1}}$) "
                f"| `{self.result1['pk_mape_pct']:.4f}%` | `{'PASS' if gates['result1_emulator_gate'] else 'FAIL'}` |"
            ),
            (
                f"| **Resultado 2 (Frontera Sesgo–Varianza)** | $|\\Delta S_8| \\le 0.30\\,\\sigma_{{\\text{{Stage-IV}}}}$ "
                f"| `{opt['bias_in_stage4_sigma']:.3f} sigma` (inflación `{opt['error_inflation_factor']:.2f}x`) "
                f"| `{'PASS' if gates['result2_frontier_gate'] else 'FAIL'}` |"
            ),
            (
                f"| **Resultado 3 (Calibración SBI: SBC + TARP)** | Cobertura 68%/95% nominal y KS $p > 0.05$ "
                f"| `Cov68={tarp['coverage_68']*100:.1f}%`, `Cov95={tarp['coverage_95']*100:.1f}%` "
                f"| `{'PASS' if gates['result3_calibration_gate'] else 'FAIL'}` |"
            ),
            (
                f"| **Resultado 4 (Transferencia entre Familias)** | Fuga cero (`ic_seed`, `cosmo_id`) + degradación acotada "
                f"| `DeltaBias={deg['delta_bias_in_sigma']:+.3f} sigma`, `DeltaCov68={deg['delta_coverage_68']*100:+.1f}%` "
                f"| `{'PASS' if gates['result4_transfer_gate'] else 'FAIL'}` |"
            ),
            "",
            "## Detalle de Transferencia entre Familias (Resultado 4)",
            "",
            f"- **Pareja de transferencia**: `{self.result4['transfer_pair']}`",
            f"- **Antes de transferir (In-Family)**: Sesgo = `{pre['mean_bias_in_sigma']:.3f} sigma`, Cobertura 68% = `{pre['coverage_68']*100:.1f}%`",
            f"- **Después de transferir (Cross-Sim)**: Sesgo = `{post['mean_bias_in_sigma']:.3f} sigma`, Cobertura 68% = `{post['coverage_68']*100:.1f}%`",
            "",
        ]
        return "\n".join(lines)

    def save_json_report(self, path: str | pathlib.Path) -> pathlib.Path:
        """Writes the structured JSON benchmark report to disk."""
        out_path = pathlib.Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return out_path

    def save_markdown_report(self, path: str | pathlib.Path) -> pathlib.Path:
        """Writes the Markdown benchmark summary table to disk."""
        out_path = pathlib.Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.format_markdown_report(), encoding="utf-8", newline="\n")
        return out_path
