#!/usr/bin/env bash
set -e
DATASET=${1:-gsm8k}
bash scripts/train_pce.sh ${DATASET} step
python experiments/run_baseline.py --dataset ${DATASET} --method cot
python experiments/run_pce.py --dataset ${DATASET} --controller threshold
python experiments/run_pce.py --dataset ${DATASET} --controller budget_aware
