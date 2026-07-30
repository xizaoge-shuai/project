#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
unset OPENBLAS_NUM_THREADS

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p outputs/metrics/cross_table/table2_valid
mkdir -p outputs/predictions/cross_table/table2_valid

for DATASET in svamp asdiv math500 mathqa
do
  TRAJ="data/processed/trajectories_cross_table/${DATASET}/test.jsonl"
  PRED="outputs/predictions/cross_table/rollout_fulltest/${DATASET}_rollout_p08_fulltest.jsonl"

  OUT_JSON="outputs/metrics/cross_table/table2_valid/${DATASET}_pce_weight.json"
  OUT_JSONL="outputs/predictions/cross_table/table2_valid/${DATASET}_pce_weight.jsonl"

  echo
  echo "======================================================================"
  echo "[TABLE2 PCE WEIGHT] dataset=${DATASET}"
  echo "======================================================================"

  test -s "$TRAJ" || {
    echo "[ERROR] missing trajectories: $TRAJ"
    exit 1
  }

  test -s "$PRED" || {
    echo "[ERROR] missing PCE predictions: $PRED"
    exit 1
  }

  echo "trajectory rows: $(wc -l < "$TRAJ")"
  echo "prediction rows: $(wc -l < "$PRED")"

  python experiments/eval_cross_pce_weighted_selection.py \
    --predictions "$PRED" \
    --trajectories "$TRAJ" \
    --dataset "$DATASET" \
    --tail_k 5 \
    --out_json "$OUT_JSON" \
    --out_jsonl "$OUT_JSONL"

  python - "$OUT_JSON" <<'PY'
import json
import sys

fp = sys.argv[1]
data = json.load(open(fp, encoding="utf-8"))

required = [
    "majority_acc",
    "pce_top1_tail_acc",
    "weighted_tail_acc",
]

missing = [key for key in required if key not in data]
if missing:
    raise SystemExit(
        f"[ERROR] missing fields in {fp}: {missing}; "
        f"available={sorted(data)}"
    )

majority = float(data["majority_acc"])
top1 = float(data["pce_top1_tail_acc"])
weighted = float(data["weighted_tail_acc"])
gain = float(
    data.get(
        "weighted_gain_vs_majority",
        weighted - majority,
    )
)

print(f"n_samples:        {data.get('n_samples')}")
print(f"majority_acc:     {majority:.6f}")
print(f"pce_top1_acc:     {top1:.6f}")
print(f"pce_weighted_acc: {weighted:.6f}")
print(f"weighted_gain:    {gain:+.6f}")

if not 0.0 <= majority <= 1.0:
    raise SystemExit("[ERROR] invalid majority accuracy")
if not 0.0 <= top1 <= 1.0:
    raise SystemExit("[ERROR] invalid top-1 accuracy")
if not 0.0 <= weighted <= 1.0:
    raise SystemExit("[ERROR] invalid weighted accuracy")
PY

  echo "[DONE] ${DATASET}"
done

echo
echo "======================================================================"
echo "ALL CROSS-DATASET PCE-WEIGHT RESULTS COMPLETED"
echo "======================================================================"
