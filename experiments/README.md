# Experiments

This directory houses reproducible experiment pipelines, benchmark scripts, and ablation studies.

## Tracking & Reproducibility
All experiments in `cosmoverse` are governed by explicit configuration files in `configs/` and record:
- Full parameter specification (`CosmoverseConfig`)
- Git revision commit hash (`git_hash`)
- Exact random seeds for deterministic initial conditions
- Environment dependencies

## Running an Experiment
```bash
python experiments/run_camels_transfer_study.py --config configs/camels_benchmark.yaml
```
