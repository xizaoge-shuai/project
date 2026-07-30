#!/usr/bin/env bash
set -u

cd /root/pce_reasoning_project/project || exit 1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}

mkdir -p outputs/logs/baselines_candidates
mkdir -p outputs/metrics/baselines_candidates

get_base_acc() {
  local DS="$1"
  local TAG="$2"
  python - "$DS" "$TAG" <<'PY'
import sys, json, glob
from pathlib import Path

ds, tag = sys.argv[1], sys.argv[2]

metric_dirs = [
    "outputs/metrics/model_ablation",
    "outputs/metrics/model_ablation_parallel_qwen3b",
    "outputs/metrics/model_ablation_boost",
    "outputs/metrics/model_ablation_mathqa_optionmap",
    "outputs/metrics/model_ablation_mathqa_optionmap_qwen3b",
    "outputs/metrics/model_ablation_mathqa_choiceboost",
    "outputs/metrics/model_ablation_mathqa_mixedboost",
]

cands = []

# 优先找 base metric
for root in metric_dirs:
    cands += glob.glob(f"{root}/{ds}_{tag}_base.json")

# MathQA 的 base_acc 不能用原始 0，要从 optionmap/mixedboost 里取 base_acc
if ds == "mathqa":
    for root in metric_dirs:
        cands += glob.glob(f"{root}/mathqa_{tag}_optionmap*.json")
        cands += glob.glob(f"{root}/mathqa_{tag}_mixedboost*.json")
        cands += glob.glob(f"{root}/mathqa_{tag}_choiceboost*.json")

for fp in cands:
    try:
        x = json.load(open(fp, encoding="utf-8"))
    except Exception:
        continue
    v = x.get("base_acc", x.get("majority_acc", x.get("acc")))
    if isinstance(v, (int, float)):
        print(v)
        raise SystemExit

print("NA")
PY
}

get_best_ours_json() {
  local DS="$1"
  local TAG="$2"
  python - "$DS" "$TAG" <<'PY'
import sys, json, glob
from pathlib import Path

ds, tag = sys.argv[1], sys.argv[2]
roots = [
    "outputs/metrics/model_ablation",
    "outputs/metrics/model_ablation_parallel_qwen3b",
    "outputs/metrics/model_ablation_boost",
    "outputs/metrics/model_ablation_mathqa_optionmap",
    "outputs/metrics/model_ablation_mathqa_optionmap_qwen3b",
    "outputs/metrics/model_ablation_mathqa_choiceboost",
    "outputs/metrics/model_ablation_mathqa_mixedboost",
]

fps = []
for root in roots:
    fps += glob.glob(f"{root}/{ds}_{tag}_*.json")
    if ds == "mathqa":
        fps += glob.glob(f"{root}/mathqa_{tag}_optionmap*.json")
        fps += glob.glob(f"{root}/mathqa_{tag}_mixedboost*.json")
        fps += glob.glob(f"{root}/mathqa_{tag}_choiceboost*.json")

best = None
for fp in fps:
    if fp.endswith("_base.json"):
        continue
    try:
        x = json.load(open(fp, encoding="utf-8"))
    except Exception:
        continue
    final = x.get("estimated_global_acc", x.get("final_acc", x.get("accuracy")))
    if not isinstance(final, (int, float)):
        continue
    if best is None or final > best[0]:
        best = (final, fp)

print(best[1] if best else "")
PY
}

make_target_ids_if_missing() {
  local TARGET_IDS="$1"
  local TARGET_JSONL="$2"

  if [ -f "$TARGET_IDS" ]; then
    return 0
  fi

  if [ ! -f "$TARGET_JSONL" ]; then
    return 0
  fi

  mkdir -p "$(dirname "$TARGET_IDS")"

  python - "$TARGET_JSONL" "$TARGET_IDS" <<'PY'
import sys, json, re
from pathlib import Path

src, out = sys.argv[1], sys.argv[2]
ids = []

def sample_key(r):
    x = r.get("sample_id") or r.get("question_id") or r.get("qid") or r.get("problem_id") or r.get("id")
    if x is None:
        x = r.get("question") or r.get("problem")
    x = str(x)
    x = re.sub(r"_traj_\d+$", "", x)
    return x

for line in open(src, encoding="utf-8"):
    if not line.strip():
        continue
    ids.append(sample_key(json.loads(line)))

Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for x in ids:
        f.write(str(x) + "\n")

print("[WRITE]", out, len(ids))
PY
}

run_one() {
  local TAG="$1"
  local ROOT_NAME="$2"
  local DS="$3"
  local TASK_TYPE="$4"
  local SCOPE_DIR="$5"
  local TRAJ_DIR="$6"
  local PRED_DIR="$7"
  shift 7
  local EXTRA_PATTERNS=("$@")

  local PREFIX="${DS}_${TAG}"
  local BASE_DETAILS="${PRED_DIR}/${DS}_${TAG}_base_details.jsonl"
  local SCOPE="${SCOPE_DIR}/${DS}_scope.jsonl"
  local TARGET_JSONL="${SCOPE_DIR}/${DS}_${TAG}_has_disagreement.jsonl"
  local TARGET_IDS="outputs/targets/${ROOT_NAME}/${DS}_${TAG}_has_disagreement_ids.txt"
  local OUT_DIR="outputs/metrics/baselines_candidates/${TAG}"
  local LOG_DIR="outputs/logs/baselines_candidates/${TAG}"

  mkdir -p "$OUT_DIR" "$LOG_DIR"

  echo
  echo "================================================================================"
  echo "[DATASET] ${DS}"
  echo "[MODEL]   ${TAG}"
  echo "[TASK]    ${TASK_TYPE}"
  echo "[BASE]    ${BASE_DETAILS}"
  echo "[SCOPE]   ${SCOPE}"
  echo "[TARGET]  ${TARGET_IDS}"
  echo "================================================================================"

  if [ ! -f "$BASE_DETAILS" ]; then
    echo "[SKIP] missing baseline details: $BASE_DETAILS"
    return 0
  fi

  if [ ! -f "$SCOPE" ]; then
    echo "[SKIP] missing scope: $SCOPE"
    return 0
  fi

  make_target_ids_if_missing "$TARGET_IDS" "$TARGET_JSONL"

  if [ ! -f "$TARGET_IDS" ]; then
    echo "[SKIP] missing target ids and cannot create: $TARGET_IDS"
    return 0
  fi

  local BASE_ACC
  BASE_ACC=$(get_base_acc "$DS" "$TAG")
  if [ "$BASE_ACC" = "NA" ] || [ -z "$BASE_ACC" ]; then
    echo "[SKIP] cannot infer base_acc for ${DS}/${TAG}"
    return 0
  fi

  local N_SAMPLES
  N_SAMPLES=$(wc -l < "$SCOPE")

  local EXTRAS=()
  for pat in "${EXTRA_PATTERNS[@]}"; do
    while IFS= read -r f; do
      [ -f "$f" ] && EXTRAS+=("$f")
    done < <(ls $pat 2>/dev/null | sort || true)
  done

  if [ "${#EXTRAS[@]}" -eq 0 ]; then
    echo "[SKIP] no extra candidates found for ${DS}/${TAG}"
    return 0
  fi

  local OURS_JSON
  OURS_JSON=$(get_best_ours_json "$DS" "$TAG")

  echo "[INFO] base_acc=${BASE_ACC}"
  echo "[INFO] n_samples=${N_SAMPLES}"
  echo "[INFO] extras=${#EXTRAS[@]}"
  printf '  %s\n' "${EXTRAS[@]}"
  echo "[INFO] ours_json=${OURS_JSON}"

  local EXTRA_ARGS=()
  if [ -n "$OURS_JSON" ] && [ -f "$OURS_JSON" ]; then
    EXTRA_ARGS+=(--ours_json "$OURS_JSON" --ours_name "Confirm")
  fi

  python scripts/eval_tts_baselines_from_candidates.py \
    --baseline_details "$BASE_DETAILS" \
    --extra_jsonls "${EXTRAS[@]}" \
    --target_ids "$TARGET_IDS" \
    --task_type "$TASK_TYPE" \
    --base_acc "$BASE_ACC" \
    --n_samples "$N_SAMPLES" \
    --out_dir "$OUT_DIR" \
    --prefix "$PREFIX" \
    --max_candidates 4 8 12 \
    --esc_windows 2 3 4 \
    --cisc_temps 0.2 0.5 1.0 2.0 \
    --gg_lambdas "1,0" "1,0.5" "1,1" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${PREFIX}_candidate_baselines.log"
}

echo "========== START candidate baselines =========="
date

# ---------------- DeepSeek7B ----------------
run_one "deepseek7b" "model_ablation" "gsm8k" "numeric" \
  "data/processed/unified/model_ablation" \
  "data/processed/trajectories/model_ablation" \
  "outputs/predictions/model_ablation" \
  "data/processed/trajectories/model_ablation/gsm8k_deepseek7b_extra_seed*.jsonl"

run_one "deepseek7b" "model_ablation" "svamp" "numeric" \
  "data/processed/unified/model_ablation" \
  "data/processed/trajectories/model_ablation" \
  "outputs/predictions/model_ablation" \
  "data/processed/trajectories/model_ablation/svamp_deepseek7b_extra_seed*.jsonl"

run_one "deepseek7b" "model_ablation" "asdiv" "numeric" \
  "data/processed/unified/model_ablation" \
  "data/processed/trajectories/model_ablation" \
  "outputs/predictions/model_ablation" \
  "data/processed/trajectories/model_ablation/asdiv_deepseek7b_extra_seed*.jsonl"

run_one "deepseek7b" "model_ablation" "math500" "numeric" \
  "data/processed/unified/model_ablation" \
  "data/processed/trajectories/model_ablation" \
  "outputs/predictions/model_ablation" \
  "data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed*.jsonl" \
  "data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed*.jsonl"

run_one "deepseek7b" "model_ablation" "mathqa" "choice" \
  "data/processed/unified/model_ablation" \
  "data/processed/trajectories/model_ablation" \
  "outputs/predictions/model_ablation" \
  "data/processed/trajectories/model_ablation/mathqa_deepseek7b_extra_seed*.jsonl" \
  "data/processed/trajectories/model_ablation_boost/mathqa_deepseek7b_choice_extra_seed*.jsonl"

run_one "deepseek7b" "model_ablation" "bbh_logical_deduction_five_objects" "choice" \
  "data/processed/unified/model_ablation" \
  "data/processed/trajectories/model_ablation" \
  "outputs/predictions/model_ablation" \
  "data/processed/trajectories/model_ablation/bbh_logical_deduction_five_objects_deepseek7b_extra_seed*.jsonl"

run_one "deepseek7b" "model_ablation" "bbh_formal_fallacies" "choice" \
  "data/processed/unified/model_ablation" \
  "data/processed/trajectories/model_ablation" \
  "outputs/predictions/model_ablation" \
  "data/processed/trajectories/model_ablation/bbh_formal_fallacies_deepseek7b_extra_seed*.jsonl"

# ---------------- Qwen3B ----------------
run_one "qwen3b" "model_ablation_parallel_qwen3b" "gsm8k" "numeric" \
  "data/processed/unified/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b" \
  "outputs/predictions/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b/gsm8k_qwen3b_extra_seed*.jsonl"

run_one "qwen3b" "model_ablation_parallel_qwen3b" "svamp" "numeric" \
  "data/processed/unified/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b" \
  "outputs/predictions/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b/svamp_qwen3b_extra_seed*.jsonl"

run_one "qwen3b" "model_ablation_parallel_qwen3b" "asdiv" "numeric" \
  "data/processed/unified/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b" \
  "outputs/predictions/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b/asdiv_qwen3b_extra_seed*.jsonl"

run_one "qwen3b" "model_ablation_parallel_qwen3b" "math500" "numeric" \
  "data/processed/unified/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b" \
  "outputs/predictions/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed*.jsonl" \
  "data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed*.jsonl"

run_one "qwen3b" "model_ablation_parallel_qwen3b" "mathqa" "choice" \
  "data/processed/unified/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b" \
  "outputs/predictions/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b/mathqa_qwen3b_extra_seed*.jsonl"

run_one "qwen3b" "model_ablation_parallel_qwen3b" "bbh_logical_deduction_five_objects" "choice" \
  "data/processed/unified/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b" \
  "outputs/predictions/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b/bbh_logical_deduction_five_objects_qwen3b_extra_seed*.jsonl"

run_one "qwen3b" "model_ablation_parallel_qwen3b" "bbh_formal_fallacies" "choice" \
  "data/processed/unified/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b" \
  "outputs/predictions/model_ablation_parallel_qwen3b" \
  "data/processed/trajectories/model_ablation_parallel_qwen3b/bbh_formal_fallacies_qwen3b_extra_seed*.jsonl"

echo "========== DONE candidate baselines =========="
date
