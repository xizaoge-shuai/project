#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GEN_CFG="configs/model/generator_llama_local_rewrite.yaml"

run_one () {
  local DATASET="$1"
  local SPLIT="$2"
  local PROGRESS="$3"
  local MAXEX="$4"
  local MAXTOK="$5"
  local OUTROOT="$6"

  local INPUT="data/processed/labels_cross_table_coarse/${DATASET}/${SPLIT}.jsonl"

  test -s "$INPUT" || {
    echo "[ERROR] missing or empty input: $INPUT"
    exit 1
  }

  echo
  echo "======================================================================"
  echo "[ROLLOUT] dataset=$DATASET split=$SPLIT progress=$PROGRESS"
  echo "max_examples=$MAXEX max_new_tokens=$MAXTOK"
  echo "======================================================================"

  python -m data.build_atom_rollout_labels \
    --input "$INPUT" \
    --output_dir "$OUTROOT" \
    --generator_config "$GEN_CFG" \
    --dataset "$DATASET" \
    --num_rollouts 3 \
    --success_hi 0.67 \
    --success_lo 0.33 \
    --min_prefix_progress "$PROGRESS" \
    --max_examples "$MAXEX" \
    --max_new_tokens "$MAXTOK" \
    --seed 42
}

for DATASET in svamp asdiv math500 mathqa
do
  case "$DATASET" in
    svamp|asdiv)
      MAXTOK=384
      ;;
    mathqa)
      MAXTOK=768
      ;;
    math500)
      MAXTOK=1024
      ;;
  esac

  for SPLIT in train val test
  do
    case "$SPLIT" in
      train)
        MAXEX=500
        ;;
      val)
        MAXEX=200
        ;;
      test)
        MAXEX=500
        ;;
    esac

    run_one \
      "$DATASET" "$SPLIT" 0.9 "$MAXEX" "$MAXTOK" \
      data/processed/labels_cross_table_rollout_p09

    run_one \
      "$DATASET" "$SPLIT" 0.8 "$MAXEX" "$MAXTOK" \
      data/processed/labels_cross_table_rollout_p08
  done
done

echo
echo "===== OUTPUT COUNTS ====="

find \
  data/processed/labels_cross_table_rollout_p09 \
  data/processed/labels_cross_table_rollout_p08 \
  -type f -name "*.jsonl" \
  -exec wc -l {} \; | sort
