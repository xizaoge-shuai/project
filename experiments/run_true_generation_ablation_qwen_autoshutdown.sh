#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

GEN_CFG=configs/model/generator_llama_local_rewrite.yaml
NO_SHUTDOWN_FILE=/tmp/NO_TRUE_ABLATION_SHUTDOWN

mkdir -p outputs/logs/true_ablation
mkdir -p outputs/metrics/true_ablation
mkdir -p outputs/predictions/true_ablation
mkdir -p outputs/logs/final_summaries
mkdir -p /root/autodl-tmp/pce_backups

trap 'echo "========== backup true ablation =========="; \
  tar --ignore-failed-read -czf /root/autodl-tmp/pce_backups/true_generation_ablation_qwen_$(date +%Y%m%d_%H%M%S).tar.gz \
    outputs/logs/true_ablation \
    outputs/metrics/true_ablation \
    outputs/predictions/true_ablation \
    outputs/logs/final_summaries/true_generation_ablation_qwen_summary.md || true; \
  if [ ! -f "$NO_SHUTDOWN_FILE" ]; then \
    echo "[SHUTDOWN] true ablation finished; shutting down."; \
    sync; shutdown -h now || poweroff || halt || true; \
  else \
    echo "[NO SHUTDOWN] found $NO_SHUTDOWN_FILE"; \
  fi' EXIT

echo "========== Step 1: GSM8K cost-accuracy ablation =========="

python experiments/analyze_resampling_cost_accuracy.py \
  --n_samples 1319 \
  --out_json outputs/metrics/resampling_cost_accuracy_summary.json \
  --out_md outputs/logs/final_summaries/resampling_cost_accuracy_summary.md \
  2>&1 | tee outputs/logs/true_ablation/analyze_gsm8k_cost_accuracy.log || true

echo "========== Step 2: ASDiv numeric full budget replay =========="

cat > experiments/apply_asdiv_numeric_budget_confirm.py <<'PY'
import argparse
import json
import re
from pathlib import Path
from collections import Counter, defaultdict


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def clean(x):
    s = str(x or "").strip().replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return s
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y


def ok(a, g):
    return clean(a) == clean(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_details", required=True)
    ap.add_argument("--target_ids", required=True)
    ap.add_argument("--extra_jsonls", nargs="+", required=True)
    ap.add_argument("--per_seed_budget", type=int, required=True)
    ap.add_argument("--min_total_support", type=int, default=2)
    ap.add_argument("--min_seed_support", type=int, default=2)
    ap.add_argument("--min_margin", type=int, default=1)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    base_rows = read_jsonl(args.baseline_details)
    base_by_id = {r["sample_id"]: r for r in base_rows}
    numeric_n = len(base_rows)
    numeric_base_acc = sum(int(r.get("majority_ok", 0)) for r in base_rows) / max(1, numeric_n)

    target_ids = [x.strip() for x in open(args.target_ids, encoding="utf-8") if x.strip()]

    extras = defaultdict(list)
    for seed_idx, fp in enumerate(args.extra_jsonls):
        per_sample_count = defaultdict(int)
        for r in read_jsonl(fp):
            sid = r["sample_id"]
            if sid not in base_by_id:
                continue
            if per_sample_count[sid] >= args.per_seed_budget:
                continue
            ans = clean(r.get("final_answer", ""))
            if ans:
                extras[sid].append((seed_idx, ans))
                per_sample_count[sid] += 1

    fixed = broken = changed = 0
    cur_correct = final_correct = 0
    out_rows = []

    for sid in target_ids:
        b = base_by_id[sid]
        gold = b["gold_answer"]
        current = clean(b.get("majority_answer", ""))
        cur_ok = int(ok(current, gold))

        cnt = Counter(a for _, a in extras.get(sid, []) if a)
        seed_support = defaultdict(set)
        for seed_idx, ans in extras.get(sid, []):
            seed_support[ans].add(seed_idx)

        if cnt:
            top, top_total = cnt.most_common(1)[0]
            runner = cnt.most_common(2)[1][1] if len(cnt) >= 2 else 0
            top_seed = len(seed_support[top])
        else:
            top, top_total, runner, top_seed = "", 0, 0, 0

        margin = top_total - runner
        final = current
        reason = "keep_current"

        if (
            top and top != current
            and top_total >= args.min_total_support
            and top_seed >= args.min_seed_support
            and margin >= args.min_margin
        ):
            final = top
            reason = f"extra_total{top_total}_seed{top_seed}_margin{margin}"

        fin_ok = int(ok(final, gold))

        is_changed = int(final != current)
        is_fixed = int(cur_ok == 0 and fin_ok == 1)
        is_broken = int(cur_ok == 1 and fin_ok == 0)

        cur_correct += cur_ok
        final_correct += fin_ok
        fixed += is_fixed
        broken += is_broken
        changed += is_changed

        out_rows.append({
            "sample_id": sid,
            "gold_answer": gold,
            "current_answer": current,
            "final_answer": final,
            "current_ok": cur_ok,
            "final_ok": fin_ok,
            "fixed": is_fixed,
            "broken": is_broken,
            "changed": is_changed,
            "reason": reason,
            "top": top,
            "top_total": top_total,
            "top_seed": top_seed,
            "runner_total": runner,
            "margin": margin,
            "extra_support": dict(cnt),
            "per_seed_budget": args.per_seed_budget,
        })

    net = fixed - broken
    summary = {
        "dataset": "asdiv_numeric",
        "ablation": "per_seed_extra_budget",
        "per_seed_budget": args.per_seed_budget,
        "n_eval": len(target_ids),
        "numeric_base_acc": numeric_base_acc,
        "numeric_n": numeric_n,
        "min_total_support": args.min_total_support,
        "min_seed_support": args.min_seed_support,
        "min_margin": args.min_margin,
        "current_acc_on_eval": cur_correct / max(1, len(target_ids)),
        "final_acc_on_eval": final_correct / max(1, len(target_ids)),
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "changed": changed,
        "estimated_numeric_acc": numeric_base_acc + net / numeric_n,
        "estimated_numeric_gain": net / numeric_n,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
PY

python -m py_compile experiments/apply_asdiv_numeric_budget_confirm.py

ASDIV_EXTRA_FILES="data/processed/trajectories/asdiv/extra_numeric_full_seed42.jsonl data/processed/trajectories/asdiv/extra_numeric_full_seed101.jsonl data/processed/trajectories/asdiv/extra_numeric_full_seed202.jsonl"

for BUDGET in 1 2 4
do
  for TOTAL in 2 3 4 5 6
  do
    for SEEDSUP in 1 2 3
    do
      for MARGIN in 0 1 2
      do
        python experiments/apply_asdiv_numeric_budget_confirm.py \
          --baseline_details outputs/predictions/asdiv_full2305_numeric_baseline_details.jsonl \
          --target_ids outputs/targets/asdiv_full2305_numeric_has_disagreement_ids.txt \
          --extra_jsonls $ASDIV_EXTRA_FILES \
          --per_seed_budget "$BUDGET" \
          --min_total_support "$TOTAL" \
          --min_seed_support "$SEEDSUP" \
          --min_margin "$MARGIN" \
          --out_json outputs/metrics/true_ablation/asdiv_budget${BUDGET}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.json \
          --out_jsonl outputs/predictions/true_ablation/asdiv_budget${BUDGET}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.jsonl
      done
    done
  done
done

echo "========== Step 3: optional ASDiv extra8 generation =========="

RUN_ASDIV_EXTRA8=${RUN_ASDIV_EXTRA8:-0}

if [ "$RUN_ASDIV_EXTRA8" = "1" ]; then
  for SEED in 42 101 202
  do
    OUT=data/processed/trajectories/asdiv/extra_numeric_full_n8_seed${SEED}.jsonl
    ROWS=0
    [ -s "$OUT" ] && ROWS=$(grep -cve '^[[:space:]]*$' "$OUT" || true)
    EXPECTED=$((997 * 8))

    if [ "$ROWS" = "$EXPECTED" ]; then
      echo "[SKIP] ASDiv extra8 seed=$SEED done $ROWS/$EXPECTED"
    else
      python scripts/generate_numeric_trajectories_local.py \
        --input data/processed/unified/asdiv/test_numeric_full.jsonl \
        --output "$OUT" \
        --generator_config "$GEN_CFG" \
        --dataset asdiv \
        --n_traj 8 \
        --max_samples 0 \
        --max_new_tokens 384 \
        --temperature 0.95 \
        --top_p 0.95 \
        --seed "$SEED" \
        2>&1 | tee outputs/logs/true_ablation/generate_asdiv_extra8_seed${SEED}.log
    fi
  done
fi

echo "========== Step 4: BBH logical5 seed-budget ablation =========="

TASK=logical_deduction_five_objects
ALL_SEEDS=(303 404 505 606 707 808 909 1001 1102 1203 1304 1405)

for SEED_BUDGET in 3 6 12
do
  EXTRA_FILES=""
  for ((i=0; i<SEED_BUDGET; i++))
  do
    SEED=${ALL_SEEDS[$i]}
    EXTRA_FILES="$EXTRA_FILES data/processed/trajectories/bbh_logic/${TASK}_extra_fixed_has_disagreement_seed${SEED}.jsonl"
  done

  BASE_ACC=$(python - <<PY
import json
x=json.load(open("outputs/metrics/bbh_logic_${TASK}_smoke100_baseline_fixed.json", encoding="utf-8"))
print(x["majority_acc_fixed"])
PY
)

  for TOTAL in 2 3 4 5 6
  do
    for SEEDSUP in 1 2 3 4
    do
      for MARGIN in 0 1 2 3
      do
        python experiments/apply_bbh_fixed_resample_confirm.py \
          --baseline_fixed_details outputs/predictions/bbh_logic_${TASK}_smoke100_baseline_fixed_details.jsonl \
          --extra_jsonls $EXTRA_FILES \
          --target_ids outputs/targets/bbh_logic_${TASK}_fixed_has_disagreement_ids.txt \
          --subtask "$TASK" \
          --min_total_support "$TOTAL" \
          --min_seed_support "$SEEDSUP" \
          --min_margin "$MARGIN" \
          --base_acc "$BASE_ACC" \
          --n_samples 100 \
          --out_json outputs/metrics/true_ablation/bbh_${TASK}_seedbudget${SEED_BUDGET}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.json \
          --out_jsonl outputs/predictions/true_ablation/bbh_${TASK}_seedbudget${SEED_BUDGET}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.jsonl || true
      done
    done
  done
done

echo "========== Step 5: summarize true ablation =========="

python - <<'PY' | tee outputs/logs/final_summaries/true_generation_ablation_qwen_summary.md
import json
from pathlib import Path
from collections import defaultdict

print("# True generation / budget ablation summary: Qwen\n")

def load_rows(pattern):
    rows = []
    for fp in Path(".").glob(pattern):
        try:
            x = json.load(open(fp, encoding="utf-8"))
            x["_fp"] = str(fp)
            rows.append(x)
        except Exception:
            pass
    return rows

def best(rows, key):
    if not rows:
        return None
    return sorted(rows, key=lambda x: (-x[key], x.get("broken", 999), -x.get("fixed", -999), x.get("changed", 999)))[0]

print("## GSM8K cost-accuracy\n")
fp = Path("outputs/logs/final_summaries/resampling_cost_accuracy_summary.md")
if fp.exists():
    print(fp.read_text(encoding="utf-8"))
else:
    print("MISSING GSM8K cost summary")

print("\n## ASDiv per-seed extra budget\n")
print("| budget_per_seed | best_acc | fixed | broken | net | changed | setting |")
print("|---:|---:|---:|---:|---:|---:|---|")
for b in [1,2,4]:
    rows = load_rows(f"outputs/metrics/true_ablation/asdiv_budget{b}_*.json")
    x = best(rows, "estimated_numeric_acc")
    if not x:
        print(f"| {b} | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
        continue
    setting = f"total{x['min_total_support']}_seed{x['min_seed_support']}_margin{x['min_margin']}"
    print(f"| {b} | {x['estimated_numeric_acc']:.4f} | {x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} | {setting} |")

print("\n## BBH logical5 seed budget\n")
print("| seed_budget | best_acc | fixed | broken | net | changed | setting |")
print("|---:|---:|---:|---:|---:|---:|---|")
for sb in [3,6,12]:
    rows = load_rows(f"outputs/metrics/true_ablation/bbh_logical_deduction_five_objects_seedbudget{sb}_*.json")
    x = best(rows, "estimated_global_acc")
    if not x:
        print(f"| {sb} | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |")
        continue
    setting = f"total{x['min_total_support']}_seed{x['min_seed_support']}_margin{x['min_margin']}"
    print(f"| {sb} | {x['estimated_global_acc']:.4f} | {x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} | {setting} |")
PY

echo "========== DONE true ablation qwen =========="
