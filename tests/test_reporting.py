"""
Unit tests for ScientificBenchmarkReporter (JSON & Markdown export for all 4 Demonstrable Results).
"""

import json
import pathlib

from cosmoverse.validation.reporting import ScientificBenchmarkReporter


def test_scientific_benchmark_reporter_gates_and_export(tmp_path: pathlib.Path):
    reporter = ScientificBenchmarkReporter(
        result1_emulator={
            "pk_mape_pct": 0.42,
            "passes_pk_threshold": True,
            "one_point_pdf_l1_error": 0.012,
            "peak_counts_mae": 0.003,
            "minkowski_mae": {"V0": 0.001, "V1": 0.002, "V2": 0.002, "V3": 0.003},
        },
        result2_frontier={
            "has_stage4_compatible_regime": True,
            "optimal_operating_point": {
                "k_max": 0.5,
                "marginalization_weight": 1.0,
                "bias_in_stage4_sigma": 0.14,
                "error_inflation_factor": 1.45,
            },
        },
        result3_calibration={
            "tarp": {
                "coverage_68": 0.69,
                "coverage_95": 0.95,
                "ks_p_value": 0.62,
                "passes_acceptance_gate": True,
            },
            "misspecification": {"is_misspecified": False},
        },
        result4_transfer={
            "transfer_pair": "CAMELS-DMO+BCEmu -> CAMELS-IllustrisTNG",
            "pre_transfer": {"mean_bias_in_sigma": 0.05, "coverage_68": 0.68},
            "post_transfer": {"mean_bias_in_sigma": -0.22, "coverage_68": 0.63},
            "degradation": {"delta_bias_in_sigma": 0.17, "delta_coverage_68": -0.05},
            "is_transfer_bounded": True,
        },
        partition_verification={
            "is_leakage_free": True,
            "shared_ic_seeds": 0,
            "shared_cosmo_ids": 0,
        },
    )

    gates = reporter.evaluate_all_acceptance_gates()
    assert gates["all_gates_passed"] is True

    md_text = reporter.format_markdown_report()
    assert "Resultado 1 (Emulador TUNeS)" in md_text
    assert "PASS" in md_text

    json_file = reporter.save_json_report(tmp_path / "report.json")
    md_file = reporter.save_markdown_report(tmp_path / "report.md")
    loaded = json.loads(json_file.read_text(encoding="utf-8"))
    assert loaded["acceptance_gates"]["all_gates_passed"] is True
    assert md_file.exists()
