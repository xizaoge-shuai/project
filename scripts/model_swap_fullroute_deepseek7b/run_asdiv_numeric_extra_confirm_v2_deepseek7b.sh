#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

GEN_CFG=configs/model/generator_deepseek_r1_distill_qwen7b_original.yaml

MAX_TARGETS=${MAX_TARGETS:-300}
TAG="numeric_smoke${MAX_TARGETS}"
if [ "$MAX_TARGETS" = "0" ]; then
  TAG="numeric_full"
fi

SEEDS="42 101 202"

mkdir -p outputs/logs outputs/metrics/model_swap_fullroute/deepseek7b/asdiv_numeric_extra_confirm outputs/predictions/model_swap_fullroute/deepseek7b/asdiv_numeric_extra_confirm
mkdir -p data/processed/trajectories/model_swap_fullroute/deepseek7b/asdiv data/processed/unified/asdiv outputs/targets

echo "========== Build ASDiv numeric target subset: ${TAG} =========="

python - <<PY
import json
from pathlib import Path

max_targets = int("${MAX_TARGETS}")
tag = "${TAG}"

src = "data/processed/unified/asdiv/test_numeric_has_disagreement_full2305.jsonl"
ids_src = "outputs/targets/model_swap_fullroute/deepseek7b/asdiv_full2305_numeric_has_disagreement_ids.txt"

rows = [json.loads(x) for x in open(src, encoding="utf-8") if x.strip()]
ids = [x.strip() for x in open(ids_src, encoding="utf-8") if x.strip()]

if max_targets > 0:
    rows = rows[:max_targets]
    ids = ids[:max_targets]

out_jsonl = Path(f"data/processed/unified/asdiv/test_{tag}.jsonl")
out_ids = Path(f"outputs/targets/model_swap_fullroute/deepseek7b/asdiv_full2305_{tag}_ids.txt")

with out_jsonl.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")

with out_ids.open("w", encoding="utf-8") as f:
    for sid in ids:
        f.write(sid + "\\n")

print("tag:", tag)
print("rows:", len(rows))
print("ids:", len(ids))
print("out_jsonl:", out_jsonl)
print("out_ids:", out_ids)
PY

echo "========== Generate ASDiv numeric extra =========="

for SEED in $SEEDS
do
  echo "==================== ASDiv ${TAG} seed=$SEED ===================="

  python scripts/generate_numeric_trajectories_local.py \
    --input data/processed/unified/asdiv/test_${TAG}.jsonl \
    --output data/processed/trajectories/model_swap_fullroute/deepseek7b/asdiv/extra_${TAG}_seed${SEED}.jsonl \
    --generator_config "$GEN_CFG" \
    --dataset asdiv \
    --n_traj 4 \
    --max_samples 0 \
    --max_new_tokens 384 \
    --temperature 0.95 \
    --top_p 0.95 \
    --seed "$SEED" \
    2>&1 | tee outputs/logs/model_swap_fullroute/deepseek7b/generate_asdiv_${TAG}_seed${SEED}.log
done

echo "========== Apply ASDiv numeric confirmation =========="

python - <<PY
import json, re
from pathlib import Path
from collections import Counter, defaultdict

TAG = "${TAG}"
SEEDS = [42, 101, 202]

def clean(x):
    s = str(x or "").strip().replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\\d+(?:\\.\\d+)?", s)
    if not nums:
        return s
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y

def ok(a, g):
    return clean(a) == clean(g)

def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]

base_details = {r["sample_id"]: r for r in read_jsonl("outputs/predictions/model_swap_fullroute/deepseek7b/asdiv_full2305_numeric_baseline_details.jsonl")}
target_ids = [x.strip() for x in open(f"outputs/targets/model_swap_fullroute/deepseek7b/asdiv_full2305_{TAG}_ids.txt", encoding="utf-8") if x.strip()]

extras = defaultdict(list)
for seed_idx, seed in enumerate(SEEDS):
    fp = f"data/processed/trajectories/model_swap_fullroute/deepseek7b/asdiv/extra_{TAG}_seed{seed}.jsonl"
    for r in read_jsonl(fp):
        sid = r["sample_id"]
        ans = clean(r.get("final_answer", ""))
        if ans:
            extras[sid].append((seed_idx, ans))

configs = [
    ("total3_seed2_margin0", 3, 2, 0),
    ("total4_seed2_margin1", 4, 2, 1),
    ("total5_seed3_margin1", 5, 3, 1),
    ("total6_seed3_margin2", 6, 3, 2),
]

Path("outputs/metrics/model_swap_fullroute/deepseek7b/asdiv_numeric_extra_confirm").mkdir(parents=True, exist_ok=True)
Path("outputs/predictions/model_swap_fullroute/deepseek7b/asdiv_numeric_extra_confirm").mkdir(parents=True, exist_ok=True)

numeric_rows = read_jsonl("outputs/predictions/model_swap_fullroute/deepseek7b/asdiv_full2305_numeric_baseline_details.jsonl")
numeric_n = len(numeric_rows)
numeric_base_acc = sum(int(r.get("majority_ok", 0)) for r in numeric_rows) / numeric_n

for name, min_total, min_seed, min_margin in configs:
    out_rows = []
    fixed = broken = changed = 0
    cur_correct = final_correct = 0

    for sid in target_ids:
        b = base_details[sid]
        gold = b["gold_answer"]
        current = clean(b["majority_answer"])
        cur_ok = int(ok(current, gold))

        cnt = Counter(a for _, a in extras.get(sid, []))
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

        if top and top != current and top_total >= min_total and top_seed >= min_seed and margin >= min_margin:
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
            "extra_seed_support": {k: len(v) for k, v in seed_support.items()},
            "base_answers": b.get("answers", []),
        })

    net = fixed - broken
    summary = {
        "dataset": "asdiv_numeric",
        "tag": TAG,
        "rule": name,
        "n_eval": len(target_ids),
        "numeric_base_acc": numeric_base_acc,
        "numeric_n": numeric_n,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "changed": changed,
        "current_acc_on_eval": cur_correct / max(1, len(target_ids)),
        "final_acc_on_eval": final_correct / max(1, len(target_ids)),
        "estimated_numeric_acc": numeric_base_acc + net / numeric_n,
        "estimated_numeric_gain": net / numeric_n,
    }

    with open(f"outputs/metrics/model_swap_fullroute/deepseek7b/asdiv_numeric_extra_confirm/{TAG}_{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(f"outputs/predictions/model_swap_fullroute/deepseek7b/asdiv_numeric_extra_confirm/{TAG}_{name}.jsonl", "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "========== Summary =========="

python - <<PY | tee outputs/logs/model_swap_fullroute/deepseek7b/asdiv_${TAG}_extra_confirm_summary.md
import json
from pathlib import Path

TAG = "${TAG}"

print(f"# ASDiv {TAG} extra confirmation summary\\n")
print("| Setting | numeric_acc | eval_acc | fixed | broken | net | changed |")
print("|---|---:|---:|---:|---:|---:|---:|")

for fp in sorted(Path("outputs/metrics/model_swap_fullroute/deepseek7b/asdiv_numeric_extra_confirm").glob(f"{TAG}_*.json")):
    x = json.load(open(fp, encoding="utf-8"))
    print(
        f"| {x['rule']} | {x['estimated_numeric_acc']:.4f} | {x['final_acc_on_eval']:.4f} | "
        f"{x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} |"
    )
PY

echo "========== DONE ${TAG} =========="
