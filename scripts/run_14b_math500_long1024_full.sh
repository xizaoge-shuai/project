#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

DATASET=math500
MODEL_TAG=deepseek14b
TAG=${DATASET}_${MODEL_TAG}_long1024
CFG=configs/model/generator_deepseek_r1_distill_qwen14b_ablation.yaml

mkdir -p outputs/logs/model_ablation_14b
mkdir -p outputs/metrics/model_ablation_14b
mkdir -p outputs/predictions/model_ablation_14b
mkdir -p outputs/targets/model_ablation_14b
mkdir -p data/processed/unified/model_ablation_14b
mkdir -p data/processed/trajectories/model_ablation_14b

echo "========== Step 0: locate source =========="
python - <<'PY'
import json
from pathlib import Path

candidates = [
    Path("data/processed/unified/model_ablation/math500_scope.jsonl"),
    Path("data/processed/unified/model_ablation_parallel_qwen3b/math500_scope.jsonl"),
    Path("data/processed/unified/math500/test.jsonl"),
    Path("data/processed/unified/math500.jsonl"),
]

src = None
for p in candidates:
    if p.exists():
        rows = [x for x in p.open(encoding="utf-8") if x.strip()]
        if len(rows) >= 500:
            src = p
            break

if src is None:
    raise SystemExit("Cannot find MATH500 500-sample source jsonl.")

out = Path("data/processed/unified/model_ablation_14b/math500_scope.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

rows = []
with src.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            sid = r.get("sample_id") or r.get("id")
            if sid is None:
                sid = f"math500_test_{len(rows)}"
            r["sample_id"] = str(sid)
            if "gold_answer" not in r and "answer" in r:
                r["gold_answer"] = r["answer"]
            rows.append(r)
            if len(rows) >= 500:
                break

with out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("src =", src)
print("saved =", out)
print("rows =", len(rows))
PY

INPUT=data/processed/unified/model_ablation_14b/math500_scope.jsonl
BASE_TRAJ=data/processed/trajectories/model_ablation_14b/${TAG}_base_3traj.jsonl
BASE_JSON=outputs/metrics/model_ablation_14b/${TAG}_base.json
BASE_DETAILS=outputs/predictions/model_ablation_14b/${TAG}_base_details.jsonl
TARGET_IDS=outputs/targets/model_ablation_14b/${TAG}_has_disagreement_ids.txt
TARGET_SCOPE=data/processed/unified/model_ablation_14b/${TAG}_has_disagreement.jsonl

echo "========== Step 1: generate base 3 trajectories =========="
python scripts/generate_numeric_trajectories_resume.py \
  --input "$INPUT" \
  --output "$BASE_TRAJ" \
  --generator_config "$CFG" \
  --dataset math500 \
  --n_traj 3 \
  --max_samples 0 \
  --max_new_tokens 1024 \
  --temperature 0.95 \
  --top_p 0.95 \
  --seed 1 \
  2>&1 | tee outputs/logs/model_ablation_14b/generate_${TAG}_base_3traj.log

echo "========== Step 2: eval base =========="
python experiments/eval_base_model_ablation.py \
  --trajectories "$BASE_TRAJ" \
  --task_type numeric \
  --out_json "$BASE_JSON" \
  --out_jsonl "$BASE_DETAILS" \
  2>&1 | tee outputs/logs/model_ablation_14b/eval_${TAG}_base.log

echo "========== Step 3: build disagreement targets =========="
python - <<PY
import json
from pathlib import Path

input_fp = Path("$INPUT")
detail_fp = Path("$BASE_DETAILS")
target_ids_fp = Path("$TARGET_IDS")
target_scope_fp = Path("$TARGET_SCOPE")

details = [json.loads(x) for x in detail_fp.open(encoding="utf-8") if x.strip()]
target_ids = []
for r in details:
    vals = [str(x).strip() for x in r.get("answers_norm", r.get("answers", [])) if str(x).strip()]
    if len(set(vals)) >= 2:
        target_ids.append(r["sample_id"])

target_set = set(target_ids)
rows = [json.loads(x) for x in input_fp.open(encoding="utf-8") if x.strip()]
subset = [r for r in rows if str(r.get("sample_id") or r.get("id")) in target_set]

target_ids_fp.parent.mkdir(parents=True, exist_ok=True)
target_scope_fp.parent.mkdir(parents=True, exist_ok=True)

target_ids_fp.write_text("\\n".join(target_ids) + "\\n", encoding="utf-8")
with target_scope_fp.open("w", encoding="utf-8") as f:
    for r in subset:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")

print("details =", len(details))
print("target_ids =", len(target_ids))
print("subset =", len(subset))
print("saved ids =", target_ids_fp)
print("saved scope =", target_scope_fp)
PY

echo "========== Step 4: generate extra candidates =========="
for SEED in 42 101 202
do
  OUT=data/processed/trajectories/model_ablation_14b/${TAG}_extra_seed${SEED}.jsonl
  LOG=outputs/logs/model_ablation_14b/generate_${TAG}_extra_seed${SEED}.log

  python scripts/generate_numeric_trajectories_resume.py \
    --input "$TARGET_SCOPE" \
    --output "$OUT" \
    --generator_config "$CFG" \
    --dataset math500 \
    --n_traj 4 \
    --max_samples 0 \
    --max_new_tokens 1024 \
    --temperature 0.95 \
    --top_p 0.95 \
    --seed ${SEED} \
    2>&1 | tee "$LOG"

  echo "rows $(wc -l < "$OUT") $OUT"
done

echo "========== Step 5: confirm sweep =========="
BASE_ACC=$(python - <<PY
import json
d=json.load(open("$BASE_JSON", encoding="utf-8"))
print(d.get("majority_acc", d.get("first_acc", d.get("accuracy", 0))))
PY
)

echo "BASE_ACC=$BASE_ACC"

for TOTAL in 2 3 4
do
  for SEEDSUP in 1 2 3
  do
    for MARGIN in 0 1 2
    do
      OUT_JSON=outputs/metrics/model_ablation_14b/${TAG}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.json
      python experiments/apply_confirm_model_ablation.py \
        --baseline_details "$BASE_DETAILS" \
        --extra_jsonls \
          data/processed/trajectories/model_ablation_14b/${TAG}_extra_seed42.jsonl \
          data/processed/trajectories/model_ablation_14b/${TAG}_extra_seed101.jsonl \
          data/processed/trajectories/model_ablation_14b/${TAG}_extra_seed202.jsonl \
        --target_ids "$TARGET_IDS" \
        --task_type numeric \
        --base_acc "$BASE_ACC" \
        --n_samples 500 \
        --min_total_support "$TOTAL" \
        --min_seed_support "$SEEDSUP" \
        --min_margin "$MARGIN" \
        --out_json "$OUT_JSON"
    done
  done
done

echo "========== Step 6: summarize =========="
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/metrics/model_ablation_14b")
rows = []
for fp in root.glob("math500_deepseek14b_long1024_total*_seed*_margin*.json"):
    d = json.load(open(fp, encoding="utf-8"))
    base = d.get("base_acc")
    final = d.get("final_acc", d.get("accuracy", d.get("acc")))
    gain = d.get("gain", None)
    if gain is None and base is not None and final is not None:
        gain = final - base
    rows.append((final or -1, gain or -999, str(fp), d))

rows.sort(key=lambda x: (x[0], x[1]), reverse=True)

out = root / "math500_deepseek14b_long1024_summary.md"
lines = []
lines.append("# DeepSeek14B MATH500 long1024 Summary")
lines.append("")
lines.append("| file | base | final | gain | n_eval | changed | fixed | broken | net |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

def pick(d, keys, default=""):
    for k in keys:
        if k in d:
            return d[k]
    return default

for final, gain, fp, d in rows[:30]:
    base = pick(d, ["base_acc"])
    n_eval = pick(d, ["n_eval", "target_n"])
    changed = pick(d, ["changed"])
    fixed = pick(d, ["fixed"])
    broken = pick(d, ["broken"])
    net = pick(d, ["net"])
    lines.append(
        f"| `{Path(fp).name}` | {float(base):.4f} | {float(final):.4f} | {float(gain):.4f} | "
        f"{n_eval} | {changed} | {fixed} | {broken} | {net} |"
    )

out.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print("saved:", out)
print(out.read_text(encoding="utf-8"))
PY

echo "DONE"
