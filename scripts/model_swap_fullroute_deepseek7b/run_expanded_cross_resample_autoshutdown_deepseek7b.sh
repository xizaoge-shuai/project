#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

GEN_CFG=${GEN_CFG:-configs/model/generator_deepseek_r1_distill_qwen7b_original.yaml}

AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-1}
NO_SHUTDOWN_FILE=${NO_SHUTDOWN_FILE:-/tmp/NO_AUTOSHUTDOWN}

mkdir -p outputs/logs outputs/metrics outputs/predictions outputs/logs/model_swap_fullroute/deepseek7b/final_summaries /root/autodl-tmp/pce_backups

MASTER_LOG=outputs/logs/model_swap_fullroute/deepseek7b/run_expanded_cross_resample_autoshutdown.master.log

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

is_done() {
  local metric="$1"
  local pred="$2"
  local expected="$3"

  python - "$metric" "$pred" "$expected" <<'PY'
import json, sys
from pathlib import Path

metric = Path(sys.argv[1])
pred = Path(sys.argv[2])
expected = int(sys.argv[3])

if not metric.exists() or not pred.exists():
    sys.exit(1)

try:
    m = json.load(open(metric, encoding="utf-8"))
except Exception:
    sys.exit(1)

rows = sum(1 for x in open(pred, encoding="utf-8") if x.strip())

target = m.get("target_samples")
completed = m.get("completed_samples")

if target == completed and rows >= expected:
    sys.exit(0)

sys.exit(1)
PY
}

run_resample_one() {
  local ds="$1"
  local tag="$2"
  local traj="$3"
  local expected="$4"
  local seed="$5"

  local prefix=""
  if [ "$ds" = "svamp" ]; then
    prefix="cross_svamp_full300_has_disagreement_extra4_seed${seed}"
  elif [ "$ds" = "asdiv" ]; then
    prefix="cross_asdiv_500_has_disagreement_extra4_seed${seed}"
  else
    echo "[ERROR] unknown dataset: $ds"
    exit 1
  fi

  local pred="outputs/predictions/model_swap_fullroute/deepseek7b/${prefix}.jsonl"
  local metric="outputs/metrics/model_swap_fullroute/deepseek7b/${prefix}.json"
  local logfp="outputs/logs/model_swap_fullroute/deepseek7b/${prefix}.log"

  if is_done "$metric" "$pred" "$expected"; then
    log "[SKIP] $ds $tag seed=$seed already done: $pred"
    return 0
  fi

  if [ -s "$pred" ] && [ ! -s "$metric" ]; then
    local backup="${pred}.partial_before_autorun_$(date '+%Y%m%d_%H%M%S').bak"
    cp "$pred" "$backup"
    log "[PARTIAL] found partial jsonl, backed up to $backup"
  fi

  log "[RUN] $ds $tag seed=$seed expected_targets=$expected"

  python experiments/run_cross_selective_resampling.py \
    --trajectories "$traj" \
    --generator_config "$GEN_CFG" \
    --dataset "$ds" \
    --trigger has_disagreement \
    --n_extra 4 \
    --max_new_tokens 384 \
    --temperature 0.95 \
    --top_p 0.95 \
    --sampling_seed "$seed" \
    --out_jsonl "$pred" \
    --out_json "$metric" \
    2>&1 | tee "$logfp"

  if is_done "$metric" "$pred" "$expected"; then
    log "[DONE] $ds $tag seed=$seed"
  else
    log "[ERROR] $ds $tag seed=$seed did not complete as expected"
    exit 1
  fi
}

run_ensemble_svamp() {
  log "[ENSEMBLE] SVAMP full300"

  python experiments/apply_resample_confirm_ensemble_seedaware.py \
    --resample_jsonls \
      outputs/predictions/model_swap_fullroute/deepseek7b/cross_svamp_full300_has_disagreement_extra4_seed42.jsonl \
      outputs/predictions/model_swap_fullroute/deepseek7b/cross_svamp_full300_has_disagreement_extra4_seed101.jsonl \
      outputs/predictions/model_swap_fullroute/deepseek7b/cross_svamp_full300_has_disagreement_extra4_seed202.jsonl \
    --min_total_support 3 \
    --min_seed_support 2 \
    --base_acc 0.9000 \
    --n_samples 300 \
    --out_json outputs/metrics/model_swap_fullroute/deepseek7b/cross_svamp_full300_extra4_seedaware_total3_seed2.json \
    --out_jsonl outputs/predictions/model_swap_fullroute/deepseek7b/cross_svamp_full300_extra4_seedaware_total3_seed2.jsonl

  python experiments/apply_current_support_guard.py \
    --input_jsonl outputs/predictions/model_swap_fullroute/deepseek7b/cross_svamp_full300_extra4_seedaware_total3_seed2.jsonl \
    --current_total_support_threshold 2 \
    --base_acc 0.9000 \
    --n_samples 300 \
    --out_json outputs/metrics/model_swap_fullroute/deepseek7b/cross_svamp_full300_extra4_seedaware_total3_seed2_currentkeep2.json \
    --out_jsonl outputs/predictions/model_swap_fullroute/deepseek7b/cross_svamp_full300_extra4_seedaware_total3_seed2_currentkeep2.jsonl

  python experiments/apply_orig_majority_guard.py \
    --input_jsonl outputs/predictions/model_swap_fullroute/deepseek7b/cross_svamp_full300_extra4_seedaware_total3_seed2_currentkeep2.jsonl \
    --orig_majority_threshold 2 \
    --base_acc 0.9000 \
    --n_samples 300 \
    --out_json outputs/metrics/model_swap_fullroute/deepseek7b/cross_svamp_full300_extra4_seedaware_total3_seed2_currentkeep2_origmaj2.json \
    --out_jsonl outputs/predictions/model_swap_fullroute/deepseek7b/cross_svamp_full300_extra4_seedaware_total3_seed2_currentkeep2_origmaj2.jsonl
}

run_ensemble_asdiv() {
  log "[ENSEMBLE] ASDiv 500"

  python experiments/apply_resample_confirm_ensemble_seedaware.py \
    --resample_jsonls \
      outputs/predictions/model_swap_fullroute/deepseek7b/cross_asdiv_500_has_disagreement_extra4_seed42.jsonl \
      outputs/predictions/model_swap_fullroute/deepseek7b/cross_asdiv_500_has_disagreement_extra4_seed101.jsonl \
      outputs/predictions/model_swap_fullroute/deepseek7b/cross_asdiv_500_has_disagreement_extra4_seed202.jsonl \
    --min_total_support 3 \
    --min_seed_support 2 \
    --base_acc 0.9400 \
    --n_samples 500 \
    --out_json outputs/metrics/model_swap_fullroute/deepseek7b/cross_asdiv_500_extra4_seedaware_total3_seed2.json \
    --out_jsonl outputs/predictions/model_swap_fullroute/deepseek7b/cross_asdiv_500_extra4_seedaware_total3_seed2.jsonl

  python experiments/apply_current_support_guard.py \
    --input_jsonl outputs/predictions/model_swap_fullroute/deepseek7b/cross_asdiv_500_extra4_seedaware_total3_seed2.jsonl \
    --current_total_support_threshold 2 \
    --base_acc 0.9400 \
    --n_samples 500 \
    --out_json outputs/metrics/model_swap_fullroute/deepseek7b/cross_asdiv_500_extra4_seedaware_total3_seed2_currentkeep2.json \
    --out_jsonl outputs/predictions/model_swap_fullroute/deepseek7b/cross_asdiv_500_extra4_seedaware_total3_seed2_currentkeep2.jsonl

  python experiments/apply_orig_majority_guard.py \
    --input_jsonl outputs/predictions/model_swap_fullroute/deepseek7b/cross_asdiv_500_extra4_seedaware_total3_seed2_currentkeep2.jsonl \
    --orig_majority_threshold 2 \
    --base_acc 0.9400 \
    --n_samples 500 \
    --out_json outputs/metrics/model_swap_fullroute/deepseek7b/cross_asdiv_500_extra4_seedaware_total3_seed2_currentkeep2_origmaj2.json \
    --out_jsonl outputs/predictions/model_swap_fullroute/deepseek7b/cross_asdiv_500_extra4_seedaware_total3_seed2_currentkeep2_origmaj2.jsonl
}

write_summary() {
  log "[SUMMARY] expanded cross-dataset results"

  python - <<'PY' | tee outputs/logs/model_swap_fullroute/deepseek7b/final_summaries/expanded_cross_resampling_summary.md
import json
from pathlib import Path

items = [
    ("SVAMP full300 majority", 300, 0.9000, None),
    ("SVAMP full300 currentkeep2", None, None, "outputs/metrics/model_swap_fullroute/deepseek7b/cross_svamp_full300_extra4_seedaware_total3_seed2_currentkeep2.json"),
    ("SVAMP full300 currentkeep2+origmaj2", None, None, "outputs/metrics/model_swap_fullroute/deepseek7b/cross_svamp_full300_extra4_seedaware_total3_seed2_currentkeep2_origmaj2.json"),
    ("ASDiv 500 majority", 500, 0.9400, None),
    ("ASDiv 500 currentkeep2", None, None, "outputs/metrics/model_swap_fullroute/deepseek7b/cross_asdiv_500_extra4_seedaware_total3_seed2_currentkeep2.json"),
    ("ASDiv 500 currentkeep2+origmaj2", None, None, "outputs/metrics/model_swap_fullroute/deepseek7b/cross_asdiv_500_extra4_seedaware_total3_seed2_currentkeep2_origmaj2.json"),
    ("MultiArith full180 majority", 180, 0.9833, None),
]

print("# Expanded cross-dataset resampling summary\n")
print("| Setting | n | Acc | fixed | broken | net | changed |")
print("|---|---:|---:|---:|---:|---:|---:|")

for name, n, acc, fp in items:
    if fp is None:
        print(f"| {name} | {n} | {acc:.4f} | - | - | - | - |")
        continue

    p = Path(fp)
    if not p.exists():
        print(f"| {name} | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
        continue

    x = json.load(open(p, encoding="utf-8"))
    print(
        f"| {name} | {x.get('n_samples', x.get('n_resampled'))} | "
        f"{float(x['estimated_global_acc']):.4f} | "
        f"{x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} |"
    )
PY
}

backup_results() {
  local status="$1"
  local backup="/root/autodl-tmp/pce_backups/pce_expanded_cross_${status}_$(date '+%Y%m%d_%H%M%S').tar.gz"

  log "[BACKUP] $backup"

  tar --ignore-failed-read -czf "$backup" \
    outputs/logs/model_swap_fullroute/deepseek7b/final_summaries/expanded_cross_resampling_summary.md \
    outputs/metrics/model_swap_fullroute/deepseek7b/cross_svamp_full300*.json \
    outputs/predictions/model_swap_fullroute/deepseek7b/cross_svamp_full300*.jsonl \
    outputs/logs/model_swap_fullroute/deepseek7b/cross_svamp_full300*.log \
    outputs/metrics/model_swap_fullroute/deepseek7b/cross_asdiv_500*.json \
    outputs/predictions/model_swap_fullroute/deepseek7b/cross_asdiv_500*.jsonl \
    outputs/logs/model_swap_fullroute/deepseek7b/cross_asdiv_500*.log \
    experiments/run_expanded_cross_resample_autoshutdown.sh \
    experiments/run_cross_selective_resampling.py \
    experiments/apply_resample_confirm_ensemble_seedaware.py \
    experiments/apply_current_support_guard.py \
    experiments/apply_orig_majority_guard.py \
    2>&1 | tee -a "$MASTER_LOG" || true

  ls -lh "$backup" | tee -a "$MASTER_LOG" || true
}

do_shutdown() {
  if [ "$AUTO_SHUTDOWN" != "1" ]; then
    log "[NO SHUTDOWN] AUTO_SHUTDOWN=$AUTO_SHUTDOWN"
    return 0
  fi

  if [ -f "$NO_SHUTDOWN_FILE" ]; then
    log "[NO SHUTDOWN] found $NO_SHUTDOWN_FILE"
    return 0
  fi

  log "[SHUTDOWN] sync and power off now"
  sync

  if command -v shutdown >/dev/null 2>&1; then
    echo "[skip inner shutdown]" && return 0 || true
  fi

  if command -v echo "[skip inner poweroff]" >/dev/null 2>&1; then
    echo "[skip inner poweroff]" && return 0 || true
  fi

  if command -v echo "[skip inner halt]" >/dev/null 2>&1; then
    echo "[skip inner halt]" && return 0 || true
  fi

  log "[WARN] no shutdown/echo "[skip inner poweroff]"/echo "[skip inner halt]" command succeeded. Please stop the instance manually."
}

on_exit() {
  local code=$?
  if [ "$code" = "0" ]; then
    backup_results "success"
  else
    backup_results "failed"
  fi
  do_shutdown
}

trap on_exit EXIT

log "[START] expanded cross resampling auto-run"
log "AUTO_SHUTDOWN=$AUTO_SHUTDOWN"
log "GEN_CFG=$GEN_CFG"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

SVAMP_TRAJ=data/processed/trajectories/model_swap_fullroute/deepseek7b/svamp/test_local_3traj_full300.jsonl
ASDIV_TRAJ=data/processed/trajectories/model_swap_fullroute/deepseek7b/asdiv/test_local_3traj_500.jsonl

for SEED in 42 101 202
do
  run_resample_one "svamp" "full300" "$SVAMP_TRAJ" 124 "$SEED"
done

for SEED in 42 101 202
do
  run_resample_one "asdiv" "500" "$ASDIV_TRAJ" 146 "$SEED"
done

run_ensemble_svamp
run_ensemble_asdiv
write_summary

log "[FINISHED] expanded cross resampling completed successfully"
