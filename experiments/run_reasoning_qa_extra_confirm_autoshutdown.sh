#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

GEN_CFG=configs/model/generator_llama_local_rewrite.yaml
NO_SHUTDOWN_FILE=/tmp/NO_AUTOSHUTDOWN

# 只跑 strategyqa 和 mathqa。hotpotqa 先不跑 extra。
DATASETS="strategyqa mathqa"
SEEDS="303 404 505 606 707 808 909 1001 1102 1203 1304 1405"

mkdir -p outputs/logs outputs/metrics outputs/predictions outputs/targets
mkdir -p outputs/metrics/reasoning_qa_extra_confirm_smoke
mkdir -p outputs/predictions/reasoning_qa_extra_confirm_smoke
mkdir -p /root/autodl-tmp/pce_backups

trap 'tar --ignore-failed-read -czf /root/autodl-tmp/pce_backups/reasoning_qa_extra_confirm_$(date +%Y%m%d_%H%M%S).tar.gz \
  outputs/targets/*_smoke100_has_disagreement_ids.txt \
  data/processed/unified/strategyqa/test_smoke100_has_disagreement.jsonl \
  data/processed/unified/mathqa/test_smoke100_has_disagreement.jsonl \
  data/processed/trajectories/strategyqa/extra_smoke100_has_disagreement_seed*.jsonl \
  data/processed/trajectories/mathqa/extra_smoke100_has_disagreement_seed*.jsonl \
  outputs/metrics/reasoning_qa_extra_confirm_smoke \
  outputs/predictions/reasoning_qa_extra_confirm_smoke \
  outputs/logs/generate_strategyqa_extra_smoke100_has_disagreement_seed*.log \
  outputs/logs/generate_mathqa_extra_smoke100_has_disagreement_seed*.log \
  outputs/logs/reasoning_qa_extra_confirm_smoke_summary.md \
  experiments/apply_reasoning_qa_resample_confirm.py \
  scripts/generate_reasoning_qa_trajectories_vllm.py \
  experiments/eval_reasoning_qa_baseline.py || true; \
  if [ ! -f "$NO_SHUTDOWN_FILE" ]; then sync; shutdown -h now || poweroff || halt || true; fi' EXIT

echo "========== Step 1: build disagreement targets =========="

for DS in $DATASETS
do
  python - <<PY
import json
from pathlib import Path

ds = "$DS"
DETAIL = f"outputs/predictions/{ds}_smoke100_multiseed_baseline_details.jsonl"
UNIFIED = f"data/processed/unified/{ds}/test.jsonl"

details = [json.loads(x) for x in open(DETAIL, encoding="utf-8") if x.strip()]

target_ids = []
for r in details:
    vals = [str(a).strip() for a in r.get("answers_norm", []) if str(a).strip()]
    if len(set(vals)) >= 2:
        target_ids.append(r["sample_id"])

target_set = set(target_ids)

Path("outputs/targets").mkdir(parents=True, exist_ok=True)
with open(f"outputs/targets/{ds}_smoke100_has_disagreement_ids.txt", "w", encoding="utf-8") as f:
    for sid in target_ids:
        f.write(sid + "\\n")

unified = [json.loads(x) for x in open(UNIFIED, encoding="utf-8") if x.strip()]
subset = [r for r in unified if (r.get("sample_id") or r.get("id")) in target_set]

out = Path(f"data/processed/unified/{ds}/test_smoke100_has_disagreement.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for r in subset:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")

print(ds, "target_ids:", len(target_ids), "subset rows:", len(subset), "out:", out)
PY
done

echo "========== Step 2: generate extra trajectories =========="

for DS in $DATASETS
do
  for SEED in $SEEDS
  do
    echo "==================== $DS extra smoke100 seed=$SEED ===================="

    python scripts/generate_reasoning_qa_trajectories_vllm.py \
      --input data/processed/unified/${DS}/test_smoke100_has_disagreement.jsonl \
      --output data/processed/trajectories/${DS}/extra_smoke100_has_disagreement_seed${SEED}.jsonl \
      --generator_config "$GEN_CFG" \
      --dataset "$DS" \
      --n_traj 1 \
      --max_samples 0 \
      --max_new_tokens 512 \
      --temperature 0.9 \
      --top_p 0.95 \
      --seed "$SEED" \
      --batch_size 4 \
      2>&1 | tee outputs/logs/generate_${DS}_extra_smoke100_has_disagreement_seed${SEED}.log
  done
done

echo "========== Step 3: check completeness =========="

python - <<'PY' | tee outputs/logs/reasoning_qa_extra_generation_check.md
from pathlib import Path

datasets = ["strategyqa", "mathqa"]
seeds = [303,404,505,606,707,808,909,1001,1102,1203,1304,1405]

for ds in datasets:
    expected = sum(1 for _ in open(f"outputs/targets/{ds}_smoke100_has_disagreement_ids.txt", encoding="utf-8"))
    print(f"\n## {ds}")
    print("expected:", expected)
    print("| seed | exists | rows | done |")
    print("|---:|---|---:|---|")
    for seed in seeds:
        fp = Path(f"data/processed/trajectories/{ds}/extra_smoke100_has_disagreement_seed{seed}.jsonl")
        rows = sum(1 for x in open(fp, encoding="utf-8") if x.strip()) if fp.exists() else 0
        done = fp.exists() and rows == expected
        print(f"| {seed} | {fp.exists()} | {rows} | {done} |")
PY

echo "========== Step 4: run extra-confirmation sweep =========="

for DS in $DATASETS
do
  BASE_ACC=$(python - <<PY
import json
x=json.load(open("outputs/metrics/${DS}_smoke100_multiseed_baseline.json", encoding="utf-8"))
print(x["majority_acc"])
PY
)

  EXTRA_FILES=""
  for SEED in $SEEDS
  do
    EXTRA_FILES="$EXTRA_FILES data/processed/trajectories/${DS}/extra_smoke100_has_disagreement_seed${SEED}.jsonl"
  done

  for TOTAL in 2 3 4 5 6 7 8
  do
    for SEEDSUP in 2 3 4 5
    do
      for MARGIN in 0 1 2 3
      do
        python experiments/apply_reasoning_qa_resample_confirm.py \
          --baseline_details outputs/predictions/${DS}_smoke100_multiseed_baseline_details.jsonl \
          --extra_jsonls $EXTRA_FILES \
          --target_ids outputs/targets/${DS}_smoke100_has_disagreement_ids.txt \
          --dataset "$DS" \
          --min_total_support "$TOTAL" \
          --min_seed_support "$SEEDSUP" \
          --min_margin "$MARGIN" \
          --base_acc "$BASE_ACC" \
          --n_samples 100 \
          --out_json outputs/metrics/reasoning_qa_extra_confirm_smoke/${DS}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.json \
          --out_jsonl outputs/predictions/reasoning_qa_extra_confirm_smoke/${DS}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.jsonl
      done
    done
  done
done

echo "========== Step 5: summarize results =========="

python - <<'PY' | tee outputs/logs/reasoning_qa_extra_confirm_smoke_summary.md
import json
from pathlib import Path

rows = []
for fp in Path("outputs/metrics/reasoning_qa_extra_confirm_smoke").glob("*.json"):
    x = json.load(open(fp, encoding="utf-8"))
    rows.append(x)

rows = sorted(
    rows,
    key=lambda x: (
        x["dataset"],
        -x["estimated_global_acc"],
        x["broken"],
        -x["fixed"],
        x["changed"],
    ),
)

print("| Dataset | total | seed | margin | Acc | target_acc | fixed | broken | net | changed |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for x in rows[:120]:
    print(
        f"| {x['dataset']} | {x['min_total_support']} | {x['min_seed_support']} | {x['min_margin']} | "
        f"{x['estimated_global_acc']:.4f} | {x['final_acc_on_eval']:.4f} | "
        f"{x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} |"
    )
PY

echo "========== DONE =========="
