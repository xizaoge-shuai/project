#!/usr/bin/env bash
set -e
DATASET=${1:-gsm8k}
python experiments/analysis/compute_metrics.py --predictions outputs/predictions/${DATASET}_cot.jsonl
python experiments/analysis/compute_metrics.py --predictions outputs/predictions/${DATASET}_pce_threshold.jsonl
