#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/run_own_generation_baselines_one_case.sh \
#   MODEL_TAG DATASET MODE SCOPE_JSON TARGET_IDS_OR_NONE GENERATOR_CONFIG N_SAMPLES BASE_ACC MAX_NEW_TOKENS

MODEL_TAG="$1"
DATASET="$2"
MODE="$3"              # targeted or full
SCOPE_JSON="$4"
TARGET_IDS="$5"        # file path or NONE
GENERATOR_CONFIG="$6"
N_SAMPLES="$7"
BASE_ACC="$8"
MAX_NEW_TOKENS="$9"

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

OUT_ROOT="outputs/metrics/own_generation_baselines"
PRED_ROOT="outputs/predictions/own_generation_baselines"
TRAJ_ROOT="data/processed/trajectories/own_generation_baselines"
SCOPE_ROOT="data/processed/unified/own_generation_baselines"
LOG_ROOT="outputs/logs/own_generation_baselines"

mkdir -p "$OUT_ROOT" "$PRED_ROOT" "$TRAJ_ROOT" "$SCOPE_ROOT" "$LOG_ROOT"

CASE="${MODEL_TAG}_${DATASET}_${MODE}_owngen"
CASE_SCOPE="${SCOPE_ROOT}/${CASE}_scope.jsonl"

echo "========== CASE=${CASE} =========="
echo "MODEL_TAG=${MODEL_TAG}"
echo "DATASET=${DATASET}"
echo "MODE=${MODE}"
echo "SCOPE_JSON=${SCOPE_JSON}"
echo "TARGET_IDS=${TARGET_IDS}"
echo "GENERATOR_CONFIG=${GENERATOR_CONFIG}"
echo "N_SAMPLES=${N_SAMPLES}"
echo "BASE_ACC=${BASE_ACC}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"

echo "========== Step 1: build scope =========="
python - "$SCOPE_JSON" "$TARGET_IDS" "$MODE" "$CASE_SCOPE" <<'PY'
import json, sys
from pathlib import Path

scope_fp = Path(sys.argv[1])
target_fp = sys.argv[2]
mode = sys.argv[3]
out_fp = Path(sys.argv[4])

rows = [json.loads(x) for x in scope_fp.open(encoding="utf-8") if x.strip()]

if mode == "targeted":
    if target_fp == "NONE":
        raise SystemExit("targeted mode requires target ids file")
    ids = set(x.strip() for x in open(target_fp, encoding="utf-8") if x.strip())
    rows = [r for r in rows if str(r.get("sample_id") or r.get("id")) in ids]

out_fp.parent.mkdir(parents=True, exist_ok=True)
with out_fp.open("w", encoding="utf-8") as f:
    for i,r in enumerate(rows):
        if not (r.get("sample_id") or r.get("id")):
            r["sample_id"] = f"sample_{i}"
        elif "sample_id" not in r:
            r["sample_id"] = str(r.get("id"))
        if "gold_answer" not in r and "answer" in r:
            r["gold_answer"] = r["answer"]
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("saved:", out_fp)
print("rows:", len(rows))
PY

echo "========== Step 2: generate own candidates, K=12 =========="
for SEED in 42 101 202
do
  OUT="${TRAJ_ROOT}/${CASE}_extra_seed${SEED}.jsonl"
  LOG="${LOG_ROOT}/generate_${CASE}_extra_seed${SEED}.log"

  python scripts/generate_numeric_trajectories_resume.py \
    --input "$CASE_SCOPE" \
    --output "$OUT" \
    --generator_config "$GENERATOR_CONFIG" \
    --dataset "$DATASET" \
    --n_traj 4 \
    --max_samples 0 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature 0.95 \
    --top_p 0.95 \
    --seed "$SEED" \
    2>&1 | tee "$LOG"

  echo "rows $(wc -l < "$OUT") $OUT"
done

echo "========== Step 3: build replay file =========="
REPLAY="${PRED_ROOT}/${CASE}_replay.jsonl"

python - "$CASE_SCOPE" "$REPLAY" "$DATASET" \
  "${TRAJ_ROOT}/${CASE}_extra_seed42.jsonl" \
  "${TRAJ_ROOT}/${CASE}_extra_seed101.jsonl" \
  "${TRAJ_ROOT}/${CASE}_extra_seed202.jsonl" <<'PY'
import json, re, sys
from pathlib import Path
from collections import defaultdict

scope_fp = Path(sys.argv[1])
out_fp = Path(sys.argv[2])
dataset = sys.argv[3]
extra_fps = [Path(x) for x in sys.argv[4:]]

def sid(r):
    return str(r.get("sample_id") or r.get("id") or "")

def norm_num(x):
    if x is None:
        return ""
    s = str(x).strip().replace(",", "").replace("$", "").replace("\\$", "")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", s)
    if nums:
        s = nums[-1]
    if "/" in s:
        try:
            a,b=s.split("/",1)
            v=float(a)/float(b)
            return f"{v:.10f}".rstrip("0").rstrip(".")
        except Exception:
            pass
    try:
        v=float(s)
        if abs(v-round(v))<1e-9:
            return str(int(round(v)))
        return f"{v:.10f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x).strip().lower()

def text_of(r):
    for k in ["trajectory","text","reasoning","output","completion","response"]:
        if r.get(k):
            return str(r[k])
    return ""

def extract(r):
    for k in ["answer","final_answer","pred_answer","prediction","extracted_answer"]:
        if r.get(k) is not None:
            return norm_num(r.get(k))
    t=text_of(r)
    for p in [r"Final Answer\s*[:：]\s*([^\n]+)", r"Answer\s*[:：]\s*([^\n]+)", r"答案\s*[:：]\s*([^\n]+)"]:
        m=re.findall(p,t,flags=re.I)
        if m:
            return norm_num(m[-1])
    return norm_num(t[-300:])

scope = [json.loads(x) for x in scope_fp.open(encoding="utf-8") if x.strip()]
gold = {}
for r in scope:
    gold[sid(r)] = norm_num(r.get("gold_answer") or r.get("answer"))

extra = defaultdict(list)
for fp in extra_fps:
    with fp.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            s=sid(r)
            a=extract(r)
            if s and a:
                extra[s].append(a)

out_fp.parent.mkdir(parents=True, exist_ok=True)
with out_fp.open("w", encoding="utf-8") as f:
    for r in scope:
        s=sid(r)
        # independent full SC/ESC 不依赖你的 base majority；这里 current_answer 留空，
        # eval 脚本主要用 extra_answers 聚合。
        out = {
            "sample_id": s,
            "gold_answer": gold.get(s, ""),
            "current_answer": "",
            "current_ok": 0,
            "final_answer": "",
            "final_ok": 0,
            "fixed": 0,
            "broken": 0,
            "changed": 0,
            "orig_answers": [],
            "extra_answers": extra.get(s, []),
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

lens=[len(json.loads(x)["extra_answers"]) for x in out_fp.open(encoding="utf-8") if x.strip()]
print("saved:", out_fp)
print("rows:", len(lens))
print("extra min/avg/max:", min(lens), sum(lens)/len(lens), max(lens))
PY

echo "========== Step 4: eval SC/ESC/CISC-support on own candidates =========="
python scripts/eval_old_candidate_baselines.py \
  --prediction_file "$REPLAY" \
  --dataset "$DATASET" \
  --task_type numeric \
  --n_samples "$N_SAMPLES" \
  --base_acc "$BASE_ACC" \
  --out_dir "$OUT_ROOT" \
  --prefix "$CASE"

cat "${OUT_ROOT}/${CASE}_old_candidate_baselines.md"
