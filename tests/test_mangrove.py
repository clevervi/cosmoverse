"""
Unit tests for the optional post-v1 Mangrove halo merger-tree GNN emulator
(Master Document Section 6.1) and Paper 2 SVG figure generation.
"""

import pathlib

import pytest

from cosmoverse.emulators.mangrove import MangroveMergerTreeGNN
from cosmoverse.validation.plotting import Paper1FigureGenerator


def test_mangrove_v1_exclusivity_guard():
    with pytest.raises(RuntimeError, match="v1 enforces a single neural emulator"):
        MangroveMergerTreeGNN(allow_post_v1_extension=False)


def test_mangrove_merger_tree_predictions_and_2x_benchmark():
    gnn = MangroveMergerTreeGNN(allow_post_v1_extension=True, seed=42)
    tree = gnn.synthesize_merger_tree(n_progenitors=12, seed=42)
    preds = gnn.forward_tree(tree)
    assert set(preds.keys()) == set(MangroveMergerTreeGNN.TARGET_PROPERTIES)

    bench = gnn.benchmark_merger_tree_vs_halo_mlp(n_trees=30, seed=42)
    assert bench["box_size_mpc_h"] == 75.0
    assert bench["rms_improvement_factor"] >= 2.0
    assert bench["passes_2x_improvement_benchmark"] is True


def test_paper2_svg_figures_generation(tmp_path: pathlib.Path):
    saved = Paper1FigureGenerator.generate_all_paper2_figures(tmp_path)
    assert saved["fig5_paper2_ic_reconstruction"].exists()
    assert saved["fig6_paper2_mclmc_vs_hmc"].exists()
    content5 = saved["fig5_paper2_ic_reconstruction"].read_text(encoding="utf-8")
    assert "<svg" in content5 and "r(k)" in content5
