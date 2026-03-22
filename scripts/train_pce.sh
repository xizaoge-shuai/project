#!/usr/bin/env bash
set -e
DATASET=${1:-gsm8k}
LEVEL=${2:-step}
python data/build_trajectories.py --dataset ${DATASET} --generator dummy --num_samples 4
python data/build_prefixes.py --dataset ${DATASET} --level ${LEVEL}
python data/build_labels.py --dataset ${DATASET} --level ${LEVEL}
python pce/train.py --config configs/model/pce_mlp.yaml --dataset ${DATASET} --level ${LEVEL}
