#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GEN_CFG="configs/model/generator_llama_local_rewrite.yaml"

for DATASET in svamp asdiv math500 mathqa
do
  INPUT="data/processed/labels_cross_table_coarse/${DATASET}/val.jsonl"
  OUTDIR="data/processed/rollout_smoke/${DATASET}_p08"

  MAXTOK=384
  [ "$DATASET" = "mathqa" ] && MAXTOK=768
  [ "$DATASET" = "math500" ] && MAXTOK=1024

  test -s "$INPUT" || {
    echo "[ERROR] empty input: $INPUT"
    exit 1
  }

  rm -rf "$OUTDIR"
  mkdir -p "$OUTDIR"

  echo
  echo "============================================================"
  echo "[SMOKE] $DATASET"
  echo "input=$INPUT"
  echo "============================================================"

  python -m data.build_atom_rollout_labels \
    --input "$INPUT" \
    --output_dir "$OUTDIR" \
    --generator_config "$GEN_CFG" \
    --dataset "$DATASET" \
    --num_rollouts 1 \
    --success_hi 0.67 \
    --success_lo 0.33 \
    --min_prefix_progress 0.8 \
    --max_examples 5 \
    --max_new_tokens "$MAXTOK" \
    --seed 42

  find "$OUTDIR" -type f -exec wc -l {} \; -print
done
