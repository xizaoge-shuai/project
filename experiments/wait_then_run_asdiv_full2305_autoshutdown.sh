#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

GEN_CFG=configs/model/generator_llama_local_rewrite.yaml
ASDIV_NO_SHUTDOWN_FILE=/tmp/NO_ASDIV_AUTOSHUTDOWN

mkdir -p outputs/logs outputs/metrics outputs/predictions outputs/targets
mkdir -p data/processed/trajectories/asdiv
mkdir -p /root/autodl-tmp/pce_backups

echo "========== Step 0: wait for current reasoning_qa extra-confirm script =========="

REASON_PID="${1:-}"
if [ -z "$REASON_PID" ]; then
  REASON_PID=$(pgrep -o -f "run_reasoning_qa_extra_confirm_autoshutdown.sh" || true)
fi

if [ -n "$REASON_PID" ]; then
  echo "[INFO] detected reasoning_qa script PID=$REASON_PID"
  while kill -0 "$REASON_PID" 2>/dev/null
  do
    echo "[WAIT] reasoning_qa script still running: PID=$REASON_PID"
    sleep 60
  done
else
  echo "[WARN] no run_reasoning_qa_extra_confirm_autoshutdown.sh PID detected."
  echo "[INFO] waiting for possible generate/apply python processes to finish..."
  while pgrep -f "generate_reasoning_qa_trajectories_vllm.py|apply_reasoning_qa_resample_confirm.py" >/dev/null 2>&1
  do
    echo "[WAIT] reasoning_qa python process still running..."
    sleep 60
  done
fi

echo "========== reasoning_qa stage finished =========="

if [ -f outputs/logs/reasoning_qa_extra_confirm_smoke_summary.md ]; then
  echo "========== reasoning_qa summary =========="
  cat outputs/logs/reasoning_qa_extra_confirm_smoke_summary.md || true
fi

# 现在允许后续 ASDiv 阶段跑完后关机
rm -f /tmp/NO_AUTOSHUTDOWN

trap 'echo "========== final backup before shutdown =========="; \
  tar --ignore-failed-read -czf /root/autodl-tmp/pce_backups/asdiv_full2305_after_reasoningqa_$(date +%Y%m%d_%H%M%S).tar.gz \
    outputs/logs/reasoning_qa_extra_confirm_smoke_summary.md \
    outputs/logs/reasoning_qa_extra_generation_check.md \
    outputs/metrics/reasoning_qa_extra_confirm_smoke \
    outputs/predictions/reasoning_qa_extra_confirm_smoke \
    data/processed/trajectories/strategyqa/extra_smoke100_has_disagreement_seed*.jsonl \
    data/processed/trajectories/mathqa/extra_smoke100_has_disagreement_seed*.jsonl \
    data/processed/trajectories/asdiv/test_local_3traj_full2305.jsonl \
    outputs/metrics/asdiv_full2305_baseline.json \
    outputs/predictions/asdiv_full2305_baseline_details.jsonl \
    outputs/logs/generate_asdiv_test_local_3traj_full2305.log \
    outputs/logs/eval_asdiv_full2305_baseline.log \
    outputs/logs/wait_then_run_asdiv_full2305_autoshutdown.log || true; \
  if [ ! -f "$ASDIV_NO_SHUTDOWN_FILE" ]; then \
    echo "[SHUTDOWN] ASDiv finished; shutting down now."; \
    sync; shutdown -h now || poweroff || halt || true; \
  else \
    echo "[NO SHUTDOWN] found $ASDIV_NO_SHUTDOWN_FILE"; \
  fi' EXIT

echo "========== Step 1: generate ASDiv full2305 3 trajectories =========="

rm -f data/processed/trajectories/asdiv/test_local_3traj_full2305.jsonl

python scripts/generate_numeric_trajectories_local.py \
  --input data/processed/unified/asdiv/test.jsonl \
  --output data/processed/trajectories/asdiv/test_local_3traj_full2305.jsonl \
  --generator_config "$GEN_CFG" \
  --dataset asdiv \
  --n_traj 3 \
  --max_samples 0 \
  --max_new_tokens 384 \
  --temperature 0.95 \
  --top_p 0.95 \
  --seed 42 \
  2>&1 | tee outputs/logs/generate_asdiv_test_local_3traj_full2305.log

echo "========== Step 2: check ASDiv full2305 trajectory file =========="

ls -lh data/processed/trajectories/asdiv/test_local_3traj_full2305.jsonl
wc -l data/processed/trajectories/asdiv/test_local_3traj_full2305.jsonl

echo "========== Step 3: evaluate ASDiv full2305 baseline =========="

python - <<'PY' 2>&1 | tee outputs/logs/eval_asdiv_full2305_baseline.log
import json, re
from collections import defaultdict, Counter
from pathlib import Path

fp = "data/processed/trajectories/asdiv/test_local_3traj_full2305.jsonl"
rows = [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]

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

by = defaultdict(list)
for r in rows:
    sid = r.get("sample_id") or r.get("id")
    by[sid].append(r)

first = majority = anyok = 0
has_dis = all_dis = 0
details = []

for sid, rs in sorted(by.items()):
    rs = sorted(rs, key=lambda x: str(x.get("trajectory_id", "")))
    gold = rs[0].get("gold_answer", rs[0].get("answer", ""))
    answers = [clean(r.get("final_answer", r.get("answer", ""))) for r in rs]

    cnt = Counter([a for a in answers if a])
    maj = cnt.most_common(1)[0][0] if cnt else ""

    first_ok = int(ok(answers[0] if answers else "", gold))
    majority_ok = int(ok(maj, gold))
    any_ok = int(any(ok(a, gold) for a in answers))

    first += first_ok
    majority += majority_ok
    anyok += any_ok

    uniq = set(a for a in answers if a)
    has_dis += int(len(uniq) >= 2)
    all_dis += int(len(uniq) >= 3)

    details.append({
        "sample_id": sid,
        "gold_answer": clean(gold),
        "answers": answers,
        "first_answer": answers[0] if answers else "",
        "majority_answer": maj,
        "first_ok": first_ok,
        "majority_ok": majority_ok,
        "oracle_any_ok": any_ok,
    })

n = len(by)
summary = {
    "dataset": "asdiv",
    "n_samples": n,
    "n_trajectories": len(rows),
    "first_acc": first / n,
    "majority_acc": majority / n,
    "oracle_any_acc": anyok / n,
    "has_disagreement": has_dis,
    "all_disagree": all_dis,
}

print(json.dumps(summary, ensure_ascii=False, indent=2))

Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
Path("outputs/predictions").mkdir(parents=True, exist_ok=True)

with open("outputs/metrics/asdiv_full2305_baseline.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

with open("outputs/predictions/asdiv_full2305_baseline_details.jsonl", "w", encoding="utf-8") as f:
    for r in details:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
PY

echo "========== Step 4: build ASDiv full disagreement target =========="

python - <<'PY'
import json
from pathlib import Path

DETAIL = "outputs/predictions/asdiv_full2305_baseline_details.jsonl"
UNIFIED = "data/processed/unified/asdiv/test.jsonl"

details = [json.loads(x) for x in open(DETAIL, encoding="utf-8") if x.strip()]

target_ids = []
for r in details:
    vals = [str(a).strip() for a in r.get("answers", []) if str(a).strip()]
    if len(set(vals)) >= 2:
        target_ids.append(r["sample_id"])

target_set = set(target_ids)

Path("outputs/targets").mkdir(parents=True, exist_ok=True)
with open("outputs/targets/asdiv_full2305_has_disagreement_ids.txt", "w", encoding="utf-8") as f:
    for sid in target_ids:
        f.write(sid + "\n")

unified = [json.loads(x) for x in open(UNIFIED, encoding="utf-8") if x.strip()]
subset = [r for r in unified if (r.get("sample_id") or r.get("id")) in target_set]

out = Path("data/processed/unified/asdiv/test_has_disagreement_full2305.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for r in subset:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("target_ids:", len(target_ids))
print("subset rows:", len(subset))
print("out:", out)
print("first ids:", target_ids[:10])
PY

echo "========== ASDiv full2305 baseline finished =========="
