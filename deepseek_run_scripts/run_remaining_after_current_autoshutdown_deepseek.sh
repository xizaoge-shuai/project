#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

GEN_CFG=configs/model/generator_deepseek_r1_distill_qwen7b_original.yaml
NO_SHUTDOWN_FILE=/tmp/NO_REMAINING_AUTOSHUTDOWN

mkdir -p outputs/logs outputs/logs/final_summaries outputs/metrics outputs/predictions outputs/targets
mkdir -p /root/autodl-tmp/pce_backups

trap 'echo "========== final backup =========="; \
  tar --ignore-failed-read -czf /root/autodl-tmp/pce_backups/pce_remaining_after_current_$(date +%Y%m%d_%H%M%S).tar.gz \
    outputs/logs/asdiv_numeric_smoke300_extra_confirm_summary.md \
    outputs/logs/asdiv_numeric_full_extra_confirm_summary.md \
    outputs/logs/mathqa_500_extra_confirm_summary.md \
    outputs/logs/bbh_logic_extra_confirm_smoke_summary.md \
    outputs/logs/final_summaries/strategyqa_negative_smoke_summary.md \
    outputs/logs/final_summaries/hotpotqa_smoke_not_mainline_summary.md \
    outputs/metrics/asdiv_numeric_extra_confirm \
    outputs/predictions/asdiv_numeric_extra_confirm \
    outputs/metrics/mathqa_scale_extra_confirm \
    outputs/predictions/mathqa_scale_extra_confirm \
    outputs/metrics/bbh_logic_extra_confirm_smoke \
    outputs/predictions/bbh_logic_extra_confirm_smoke \
    data/processed/trajectories/asdiv/extra_numeric_*_seed*.jsonl \
    data/processed/trajectories/mathqa/mathqa_500_* \
    data/processed/trajectories/bbh_logic/*extra_smoke100_has_disagreement_seed*.jsonl \
    experiments/run_remaining_after_current_autoshutdown.sh || true; \
  if [ ! -f "$NO_SHUTDOWN_FILE" ]; then \
    echo "[SHUTDOWN] all remaining jobs finished; shutting down."; \
    sync; echo "[skip inner shutdown]" || echo "[skip inner poweroff]" || echo "[skip inner halt]" || true; \
  else \
    echo "[NO SHUTDOWN] found $NO_SHUTDOWN_FILE"; \
  fi' EXIT

echo "========== Step 0: wait for current foreground/background jobs =========="

# 等你当前正在跑的 ASDiv numeric smoke300 或其它生成任务结束
while pgrep -f "run_asdiv_numeric_extra_confirm_v2.sh|generate_numeric_trajectories_local.py|generate_bbh_logic_trajectories_vllm.py|run_mathqa_scale_extra_confirm.sh|generate_reasoning_qa_trajectories_vllm.py" >/dev/null 2>&1
do
  echo "[WAIT] current generation job still running..."
  ps -ef | grep -E "run_asdiv_numeric_extra_confirm_v2|generate_numeric_trajectories_local|generate_bbh_logic_trajectories_vllm|run_mathqa_scale_extra_confirm|generate_reasoning_qa_trajectories_vllm" | grep -v grep || true
  sleep 60
done

echo "========== Step 1: ensure ASDiv numeric smoke300 result =========="

if [ ! -s outputs/logs/asdiv_numeric_smoke300_extra_confirm_summary.md ]; then
  echo "[RUN] ASDiv numeric smoke300 was not summarized; running it now."
  if [ -f experiments/run_asdiv_numeric_extra_confirm_v2.sh ]; then
    MAX_TARGETS=300 bash experiments/run_asdiv_numeric_extra_confirm_v2.sh \
      2>&1 | tee outputs/logs/run_asdiv_numeric_smoke300_extra_confirm_from_remaining.log
  else
    echo "[ERROR] missing experiments/run_asdiv_numeric_extra_confirm_v2.sh"
  fi
else
  echo "[SKIP] ASDiv numeric smoke300 summary already exists."
  cat outputs/logs/asdiv_numeric_smoke300_extra_confirm_summary.md || true
fi

echo "========== Step 2: decide whether to run ASDiv numeric full =========="

ASDIV_SMOKE_NET=$(python - <<'PY'
import json
from pathlib import Path

fps = list(Path("outputs/metrics/asdiv_numeric_extra_confirm").glob("numeric_smoke300_*.json"))
if not fps:
    print(-999)
else:
    best = None
    for fp in fps:
        x = json.load(open(fp, encoding="utf-8"))
        if best is None or (x["estimated_numeric_acc"], -x["broken"], x["net"]) > (best["estimated_numeric_acc"], -best["broken"], best["net"]):
            best = x
    print(best["net"])
PY
)

echo "ASDIV_SMOKE_NET=$ASDIV_SMOKE_NET"

if python - <<PY
net = float("$ASDIV_SMOKE_NET")
raise SystemExit(0 if net > 0 else 1)
PY
then
  if [ ! -s outputs/logs/asdiv_numeric_full_extra_confirm_summary.md ]; then
    echo "[RUN] ASDiv numeric full997 because smoke net is positive."
    if [ -f experiments/run_asdiv_numeric_extra_confirm_v2.sh ]; then
      MAX_TARGETS=0 bash experiments/run_asdiv_numeric_extra_confirm_v2.sh \
        2>&1 | tee outputs/logs/run_asdiv_numeric_full_extra_confirm_from_remaining.log
    else
      echo "[ERROR] missing experiments/run_asdiv_numeric_extra_confirm_v2.sh"
    fi
  else
    echo "[SKIP] ASDiv numeric full summary already exists."
    cat outputs/logs/asdiv_numeric_full_extra_confirm_summary.md || true
  fi
else
  echo "[SKIP] ASDiv numeric full because smoke net is not positive."
fi

echo "========== Step 3: run MathQA-500 =========="

if [ ! -s outputs/logs/mathqa_500_extra_confirm_summary.md ]; then
  if [ -f experiments/run_mathqa_scale_extra_confirm.sh ]; then
    echo "[RUN] MathQA-500 extra-confirm."
    MAX_SAMPLES=500 bash experiments/run_mathqa_scale_extra_confirm.sh \
      2>&1 | tee outputs/logs/run_mathqa_500_extra_confirm_from_remaining.log
  else
    echo "[WARN] missing experiments/run_mathqa_scale_extra_confirm.sh, skip MathQA-500."
  fi
else
  echo "[SKIP] MathQA-500 summary already exists."
  cat outputs/logs/mathqa_500_extra_confirm_summary.md || true
fi

echo "========== Step 4: run BBH logic boolean/formal extra-confirm =========="

TASKS="boolean_expressions formal_fallacies"
SEEDS="303 404 505 606 707 808 909 1001 1102 1203 1304 1405"

# 4.1 build target files
for TASK in $TASKS
do
  if [ ! -s outputs/targets/bbh_logic_${TASK}_smoke100_has_disagreement_ids.txt ]; then
    echo "[BUILD TARGET] $TASK"
    python - <<PY
import json
from pathlib import Path

task = "$TASK"
DETAIL = f"outputs/predictions/bbh_logic_{task}_smoke100_baseline_details.jsonl"
UNIFIED = f"data/processed/unified/bbh_logic/{task}.jsonl"

details = [json.loads(x) for x in open(DETAIL, encoding="utf-8") if x.strip()]
target_ids = []
for r in details:
    vals = [str(a).strip() for a in r.get("answers_norm", []) if str(a).strip()]
    if len(set(vals)) >= 2:
        target_ids.append(r["sample_id"])

target_set = set(target_ids)
Path("outputs/targets").mkdir(parents=True, exist_ok=True)

with open(f"outputs/targets/bbh_logic_{task}_smoke100_has_disagreement_ids.txt", "w", encoding="utf-8") as f:
    for sid in target_ids:
        f.write(sid + "\\n")

unified = [json.loads(x) for x in open(UNIFIED, encoding="utf-8") if x.strip()]
subset = [r for r in unified if (r.get("sample_id") or r.get("id")) in target_set]

out = Path(f"data/processed/unified/bbh_logic/{task}_smoke100_has_disagreement.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for r in subset:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")

print("task:", task)
print("target_ids:", len(target_ids))
print("subset rows:", len(subset))
PY
  fi
done

# 4.2 generate extra
for TASK in $TASKS
do
  EXPECTED=$(wc -l < outputs/targets/bbh_logic_${TASK}_smoke100_has_disagreement_ids.txt)

  for SEED in $SEEDS
  do
    OUT=data/processed/trajectories/bbh_logic/${TASK}_extra_smoke100_has_disagreement_seed${SEED}.jsonl
    ROWS=0
    if [ -s "$OUT" ]; then
      ROWS=$(grep -cve '^[[:space:]]*$' "$OUT" || true)
    fi

    if [ "$ROWS" = "$EXPECTED" ]; then
      echo "[SKIP] $TASK seed=$SEED already complete: $ROWS/$EXPECTED"
    else
      echo "[RUN] BBH $TASK extra seed=$SEED, current rows=$ROWS expected=$EXPECTED"
      python scripts/generate_bbh_logic_trajectories_vllm.py \
        --input data/processed/unified/bbh_logic/${TASK}_smoke100_has_disagreement.jsonl \
        --output "$OUT" \
        --generator_config "$GEN_CFG" \
        --n_traj 1 \
        --max_samples 0 \
        --max_new_tokens 512 \
        --temperature 0.9 \
        --top_p 0.95 \
        --seed "$SEED" \
        --batch_size 4 \
        2>&1 | tee outputs/logs/generate_bbh_${TASK}_extra_seed${SEED}_from_remaining.log
    fi
  done
done

# 4.3 sweep BBH confirmation
if [ -f experiments/apply_bbh_logic_resample_confirm.py ]; then
  mkdir -p outputs/metrics/bbh_logic_extra_confirm_smoke
  mkdir -p outputs/predictions/bbh_logic_extra_confirm_smoke

  for TASK in $TASKS
  do
    BASE_ACC=$(python - <<PY
import json
x=json.load(open("outputs/metrics/bbh_logic_${TASK}_smoke100_baseline.json", encoding="utf-8"))
print(x["majority_acc"])
PY
)

    EXTRA_FILES=""
    for SEED in $SEEDS
    do
      EXTRA_FILES="$EXTRA_FILES data/processed/trajectories/bbh_logic/${TASK}_extra_smoke100_has_disagreement_seed${SEED}.jsonl"
    done

    for TOTAL in 2 3 4 5 6 7 8
    do
      for SEEDSUP in 2 3 4 5
      do
        for MARGIN in 0 1 2 3
        do
          python experiments/apply_bbh_logic_resample_confirm.py \
            --baseline_details outputs/predictions/bbh_logic_${TASK}_smoke100_baseline_details.jsonl \
            --extra_jsonls $EXTRA_FILES \
            --target_ids outputs/targets/bbh_logic_${TASK}_smoke100_has_disagreement_ids.txt \
            --subtask "$TASK" \
            --min_total_support "$TOTAL" \
            --min_seed_support "$SEEDSUP" \
            --min_margin "$MARGIN" \
            --base_acc "$BASE_ACC" \
            --n_samples 100 \
            --out_json outputs/metrics/bbh_logic_extra_confirm_smoke/${TASK}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.json \
            --out_jsonl outputs/predictions/bbh_logic_extra_confirm_smoke/${TASK}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.jsonl
        done
      done
    done
  done

  python - <<'PY' | tee outputs/logs/bbh_logic_extra_confirm_smoke_summary.md
import json
from pathlib import Path

rows = []
for fp in Path("outputs/metrics/bbh_logic_extra_confirm_smoke").glob("*.json"):
    x = json.load(open(fp, encoding="utf-8"))
    rows.append(x)

rows = sorted(
    rows,
    key=lambda x: (
        x["subtask"],
        -x["estimated_global_acc"],
        x["broken"],
        -x["fixed"],
        x["changed"],
    ),
)

print("| Task | total | seed | margin | Acc | target_acc | fixed | broken | net | changed |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for x in rows[:120]:
    print(
        f"| {x['subtask']} | {x['min_total_support']} | {x['min_seed_support']} | {x['min_margin']} | "
        f"{x['estimated_global_acc']:.4f} | {x['final_acc_on_eval']:.4f} | "
        f"{x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} |"
    )
PY
else
  echo "[WARN] missing experiments/apply_bbh_logic_resample_confirm.py, skip BBH sweep."
fi

echo "========== Step 5: save StrategyQA negative smoke summary =========="

python - <<'PY' | tee outputs/logs/final_summaries/strategyqa_negative_smoke_summary.md
import json
from pathlib import Path

base_fp = Path("outputs/metrics/strategyqa_smoke100_multiseed_baseline.json")
if not base_fp.exists():
    print("MISSING strategyqa baseline")
    raise SystemExit(0)

base = json.load(open(base_fp, encoding="utf-8"))
rows = []
for fp in Path("outputs/metrics/reasoning_qa_extra_confirm_smoke").glob("strategyqa_*.json"):
    x = json.load(open(fp, encoding="utf-8"))
    rows.append(x)

rows = sorted(rows, key=lambda x: (-x["estimated_global_acc"], x["broken"], -x["fixed"], x["changed"]))

print("# StrategyQA negative smoke summary\n")
print("| Setting | Acc | fixed | broken | net | changed |")
print("|---|---:|---:|---:|---:|---:|")
print(f"| majority | {base['majority_acc']:.4f} | - | - | - | - |")
for x in rows[:10]:
    name = f"total{x['min_total_support']}_seed{x['min_seed_support']}_margin{x['min_margin']}"
    print(f"| {name} | {x['estimated_global_acc']:.4f} | {x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} |")
PY

echo "========== Step 6: save HotpotQA not-mainline summary =========="

python - <<'PY' | tee outputs/logs/final_summaries/hotpotqa_smoke_not_mainline_summary.md
import json
from pathlib import Path

fp = Path("outputs/metrics/hotpotqa_smoke100_multiseed_baseline.json")
if not fp.exists():
    print("MISSING hotpotqa baseline")
    raise SystemExit(0)

x = json.load(open(fp, encoding="utf-8"))

print("# HotpotQA smoke summary\n")
print("| Dataset | n | first | majority | oracle_any | has_disagreement | all_disagree |")
print("|---|---:|---:|---:|---:|---:|---:|")
print(
    f"| HotpotQA smoke100 | {x['n_samples']} | {x['first_acc']:.4f} | "
    f"{x['majority_acc']:.4f} | {x['oracle_any_acc']:.4f} | "
    f"{x['has_disagreement']} | {x['all_disagree']} |"
)
PY

echo "========== ALL REMAINING JOBS DONE =========="
