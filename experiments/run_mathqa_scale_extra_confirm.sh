#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

GEN_CFG=configs/model/generator_llama_local_rewrite.yaml

MAX_SAMPLES=${MAX_SAMPLES:-500}
TAG="mathqa_${MAX_SAMPLES}"
if [ "$MAX_SAMPLES" = "0" ]; then
  TAG="mathqa_full2985"
fi

BASE_SEEDS="42 101 202"
EXTRA_SEEDS="303 404 505 606 707 808 909 1001 1102 1203 1304 1405"

mkdir -p outputs/logs outputs/metrics outputs/predictions outputs/targets
mkdir -p data/processed/trajectories/mathqa data/processed/unified/mathqa
mkdir -p outputs/metrics/mathqa_scale_extra_confirm outputs/predictions/mathqa_scale_extra_confirm

echo "========== Step 1: generate MathQA base trajectories: $TAG =========="

for SEED in $BASE_SEEDS
do
  python scripts/generate_reasoning_qa_trajectories_vllm.py \
    --input data/processed/unified/mathqa/test.jsonl \
    --output data/processed/trajectories/mathqa/${TAG}_1traj_seed${SEED}.jsonl \
    --generator_config "$GEN_CFG" \
    --dataset mathqa \
    --n_traj 1 \
    --max_samples "$MAX_SAMPLES" \
    --max_new_tokens 512 \
    --temperature 0.9 \
    --top_p 0.95 \
    --seed "$SEED" \
    --batch_size 4 \
    2>&1 | tee outputs/logs/generate_${TAG}_base_seed${SEED}.log
done

echo "========== Step 2: merge base trajectories =========="

python - <<PY
import json
from pathlib import Path

tag = "$TAG"
seeds = [42, 101, 202]
out = Path(f"data/processed/trajectories/mathqa/{tag}_3traj_multiseed.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    for j, seed in enumerate(seeds):
        fp = f"data/processed/trajectories/mathqa/{tag}_1traj_seed{seed}.jsonl"
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            sid = r["sample_id"]
            r["traj_id"] = j
            r["trajectory_id"] = f"{sid}_traj_{j}_seed{seed}"
            r["sampling_seed"] = seed
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")

print("saved:", out)
PY

echo "========== Step 3: evaluate MathQA baseline =========="

python experiments/eval_reasoning_qa_baseline.py \
  --trajectories data/processed/trajectories/mathqa/${TAG}_3traj_multiseed.jsonl \
  --dataset mathqa \
  --out_json outputs/metrics/${TAG}_baseline.json \
  --out_jsonl outputs/predictions/${TAG}_baseline_details.jsonl \
  2>&1 | tee outputs/logs/eval_${TAG}_baseline.log

echo "========== Step 4: build disagreement target =========="

python - <<PY
import json
from pathlib import Path

tag = "$TAG"
detail = f"outputs/predictions/{tag}_baseline_details.jsonl"
unified = "data/processed/unified/mathqa/test.jsonl"

details = [json.loads(x) for x in open(detail, encoding="utf-8") if x.strip()]
target_ids = []
for r in details:
    vals = [str(a).strip() for a in r.get("answers_norm", []) if str(a).strip()]
    if len(set(vals)) >= 2:
        target_ids.append(r["sample_id"])

target_set = set(target_ids)

Path("outputs/targets").mkdir(parents=True, exist_ok=True)
with open(f"outputs/targets/{tag}_has_disagreement_ids.txt", "w", encoding="utf-8") as f:
    for sid in target_ids:
        f.write(sid + "\\n")

unified_rows = [json.loads(x) for x in open(unified, encoding="utf-8") if x.strip()]
subset = [r for r in unified_rows if (r.get("sample_id") or r.get("id")) in target_set]

out = Path(f"data/processed/unified/mathqa/{tag}_has_disagreement.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for r in subset:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")

print("tag:", tag)
print("target_ids:", len(target_ids))
print("subset rows:", len(subset))
print("out:", out)
PY

echo "========== Step 5: generate MathQA extra trajectories =========="

for SEED in $EXTRA_SEEDS
do
  python scripts/generate_reasoning_qa_trajectories_vllm.py \
    --input data/processed/unified/mathqa/${TAG}_has_disagreement.jsonl \
    --output data/processed/trajectories/mathqa/${TAG}_extra_seed${SEED}.jsonl \
    --generator_config "$GEN_CFG" \
    --dataset mathqa \
    --n_traj 1 \
    --max_samples 0 \
    --max_new_tokens 512 \
    --temperature 0.9 \
    --top_p 0.95 \
    --seed "$SEED" \
    --batch_size 4 \
    2>&1 | tee outputs/logs/generate_${TAG}_extra_seed${SEED}.log
done

echo "========== Step 6: sweep confirmation =========="

BASE_ACC=$(python - <<PY
import json
x=json.load(open("outputs/metrics/${TAG}_baseline.json", encoding="utf-8"))
print(x["majority_acc"])
PY
)

N_SAMPLES=$(python - <<PY
import json
x=json.load(open("outputs/metrics/${TAG}_baseline.json", encoding="utf-8"))
print(x["n_samples"])
PY
)

EXTRA_FILES=""
for SEED in $EXTRA_SEEDS
do
  EXTRA_FILES="$EXTRA_FILES data/processed/trajectories/mathqa/${TAG}_extra_seed${SEED}.jsonl"
done

for TOTAL in 2 3 4 5 6 7 8
do
  for SEEDSUP in 2 3 4 5
  do
    for MARGIN in 0 1 2 3
    do
      python experiments/apply_reasoning_qa_resample_confirm.py \
        --baseline_details outputs/predictions/${TAG}_baseline_details.jsonl \
        --extra_jsonls $EXTRA_FILES \
        --target_ids outputs/targets/${TAG}_has_disagreement_ids.txt \
        --dataset mathqa \
        --min_total_support "$TOTAL" \
        --min_seed_support "$SEEDSUP" \
        --min_margin "$MARGIN" \
        --base_acc "$BASE_ACC" \
        --n_samples "$N_SAMPLES" \
        --out_json outputs/metrics/mathqa_scale_extra_confirm/${TAG}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.json \
        --out_jsonl outputs/predictions/mathqa_scale_extra_confirm/${TAG}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.jsonl
    done
  done
done

echo "========== Step 7: summarize MathQA scale =========="

python - <<PY | tee outputs/logs/${TAG}_extra_confirm_summary.md
import json
from pathlib import Path

tag = "$TAG"

rows = []
for fp in Path("outputs/metrics/mathqa_scale_extra_confirm").glob(f"{tag}_*.json"):
    x = json.load(open(fp, encoding="utf-8"))
    x["_fp"] = str(fp)
    rows.append(x)

rows = sorted(rows, key=lambda x: (-x["estimated_global_acc"], x["broken"], -x["fixed"], x["changed"]))

base = json.load(open(f"outputs/metrics/{tag}_baseline.json", encoding="utf-8"))

print(f"# {tag} extra confirmation summary\\n")
print("| Setting | Acc | target_acc | fixed | broken | net | changed |")
print("|---|---:|---:|---:|---:|---:|---:|")
print(f"| majority | {base['majority_acc']:.4f} | - | - | - | - | - |")

for x in rows[:40]:
    name = f"total{x['min_total_support']}_seed{x['min_seed_support']}_margin{x['min_margin']}"
    print(
        f"| {name} | {x['estimated_global_acc']:.4f} | {x['final_acc_on_eval']:.4f} | "
        f"{x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} |"
    )
PY

echo "========== DONE $TAG =========="
