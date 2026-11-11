# Paper 1 — Federated Learning Experiments

Single source of truth for Flat FL vs HFL matched-pair experiments.

## Layout
- `COMMON/fl_core/` — implementation
- `COLAB/` — CIFAR-100 execution notebook
- `KAGGLE/` — Wheat execution notebook
- `ANALYSIS/` — final analysis notebook
- `tests/` — unit tests
- `ARCHIVE/`, `RUNS/`, `AUDIT/` — governance and audit
- Runtime only (gitignored): `RESULTS/`, `CHECKPOINTS/`, `LOGS/`, `PARTITIONS/`, `DATA/`

## Protocol
[`COMMON/fl_core/CONTRACT.md`](COMMON/fl_core/CONTRACT.md)

## Setup
    pip install -e ".[dev]"
    pytest tests/ -v
