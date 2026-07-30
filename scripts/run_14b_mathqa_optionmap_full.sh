#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

CFG=configs/model/generator_deepseek_r1_distill_qwen14b_mathqa_choice.yaml
TAG=mathqa_deepseek14b

SCOPE_SRC=data/processed/unified/model_ablation/mathqa_scope.jsonl
SCOPE_DIR=data/processed/unified/model_ablation_14b
TRAJ_DIR=data/processed/trajectories/model_ablation_14b
PRED_DIR=outputs/predictions/model_ablation_14b
TARGET_DIR=outputs/targets/model_ablation_14b
METRIC_DIR=outputs/metrics/model_ablation_mathqa_optionmap_14b
LOG_DIR=outputs/logs/model_ablation_14b

mkdir -p "$SCOPE_DIR" "$TRAJ_DIR" "$PRED_DIR" "$TARGET_DIR" "$METRIC_DIR" "$LOG_DIR"

SCOPE=${SCOPE_DIR}/mathqa_scope.jsonl
BASE_TRAJ=${TRAJ_DIR}/${TAG}_base_3traj.jsonl
BASE_DETAILS=${PRED_DIR}/${TAG}_base_details.jsonl
TARGET_IDS=${TARGET_DIR}/${TAG}_has_disagreement_ids.txt
TARGET_SCOPE=${SCOPE_DIR}/${TAG}_has_disagreement.jsonl

cp -f "$SCOPE_SRC" "$SCOPE"

echo "========== Step 1: generate base 3traj =========="
python scripts/generate_numeric_trajectories_resume.py \
  --input "$SCOPE" \
  --output "$BASE_TRAJ" \
  --generator_config "$CFG" \
  --dataset mathqa \
  --n_traj 3 \
  --max_samples 0 \
  --max_new_tokens 1024 \
  --temperature 0.95 \
  --top_p 0.95 \
  --seed 42 \
  2>&1 | tee "$LOG_DIR/generate_${TAG}_base_3traj.log"

echo "========== Step 2: build base_details and target ids =========="
python - <<'PY'
import json, re
from pathlib import Path
from collections import defaultdict, Counter

SCOPE = Path("data/processed/unified/model_ablation_14b/mathqa_scope.jsonl")
BASE_TRAJ = Path("data/processed/trajectories/model_ablation_14b/mathqa_deepseek14b_base_3traj.jsonl")
BASE_DETAILS = Path("outputs/predictions/model_ablation_14b/mathqa_deepseek14b_base_details.jsonl")
TARGET_IDS = Path("outputs/targets/model_ablation_14b/mathqa_deepseek14b_has_disagreement_ids.txt")
TARGET_SCOPE = Path("data/processed/unified/model_ablation_14b/mathqa_deepseek14b_has_disagreement.jsonl")
BASE_JSON = Path("outputs/metrics/model_ablation_mathqa_optionmap_14b/mathqa_deepseek14b_base.json")

LABELS = ["a","b","c","d","e"]

def sid(r):
    return str(r.get("sample_id") or r.get("id") or "")

def norm_text(x):
    return re.sub(r"\s+", " ", str(x or "").strip().lower()).strip(" .。,:：;；")

def parse_options(r):
    opts = {}
    v = r.get("choices") or r.get("options") or r.get("answer_choices")
    if isinstance(v, dict):
        for k, val in v.items():
            lab = str(k).strip().lower()[:1]
            if lab in LABELS:
                opts[lab] = str(val)
    elif isinstance(v, list):
        for i, val in enumerate(v[:5]):
            opts[LABELS[i]] = str(val)
    elif isinstance(v, str):
        pat = re.compile(r"(?i)(?:^|[\s,;])([abcde])\s*[\)\.：:]\s*(.*?)(?=(?:[\s,;]+[abcde]\s*[\)\.：:])|$)")
        for lab, val in pat.findall(" " + v):
            opts[lab.lower()] = val.strip(" ,;")
    return opts

def ans_to_label(ans, opts):
    if ans is None:
        return ""
    s = norm_text(ans)
    m = re.match(r"^\(?\s*([abcde])\s*\)?$", s)
    if m:
        return m.group(1)
    m = re.match(r"^\(?\s*([abcde])\s*\)?[\.\):：\s]", s)
    if m:
        return m.group(1)
    for lab, val in opts.items():
        if norm_text(val) == s:
            return lab
    for lab, val in opts.items():
        nv = norm_text(val)
        if nv and (nv in s or s in nv):
            return lab
    return ""

def extract_answer(r, opts):
    for k in ["answer","final_answer","pred_answer","prediction","extracted_answer"]:
        if r.get(k) is not None:
            lab = ans_to_label(r.get(k), opts)
            if lab:
                return lab
    txt = ""
    for k in ["trajectory","text","reasoning","output","completion","response"]:
        if r.get(k):
            txt = str(r[k])
            break
    for p in [r"Final Answer\s*[:：]\s*([^\n]+)", r"Answer\s*[:：]\s*([^\n]+)", r"答案\s*[:：]\s*([^\n]+)"]:
        m = re.findall(p, txt, flags=re.I)
        if m:
            lab = ans_to_label(m[-1], opts)
            if lab:
                return lab
    return ans_to_label(txt[-300:], opts)

def majority(xs):
    xs = [x for x in xs if x]
    if not xs:
        return ""
    c = Counter(xs)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

scope_rows = [json.loads(x) for x in SCOPE.open(encoding="utf-8") if x.strip()]
row_by_sid = {sid(r): r for r in scope_rows}
opts_by_sid = {s: parse_options(r) for s, r in row_by_sid.items()}
gold_by_sid = {s: ans_to_label(row_by_sid[s].get("gold_answer") or row_by_sid[s].get("answer"), opts_by_sid[s]) for s in row_by_sid}

answers = defaultdict(list)
for line in BASE_TRAJ.open(encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    s = sid(r)
    if s not in row_by_sid:
        continue
    lab = extract_answer(r, opts_by_sid[s])
    answers[s].append(lab)

BASE_DETAILS.parent.mkdir(parents=True, exist_ok=True)
TARGET_IDS.parent.mkdir(parents=True, exist_ok=True)
TARGET_SCOPE.parent.mkdir(parents=True, exist_ok=True)
BASE_JSON.parent.mkdir(parents=True, exist_ok=True)

details = []
target = []
correct = 0

for r in scope_rows:
    s = sid(r)
    ans = answers.get(s, [])
    maj = majority(ans)
    gold = gold_by_sid.get(s, "")
    ok = int(maj == gold and gold != "")
    correct += ok
    disag = len(set([a for a in ans if a])) > 1

    out = dict(r)
    out.update({
        "sample_id": s,
        "gold_answer": gold,
        "answers": ans,
        "majority_answer": maj,
        "majority_ok": ok,
        "has_disagreement": disag,
    })
    details.append(out)
    if disag:
        target.append(out)

with BASE_DETAILS.open("w", encoding="utf-8") as f:
    for r in details:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with TARGET_IDS.open("w", encoding="utf-8") as f:
    for r in target:
        f.write(r["sample_id"] + "\n")

with TARGET_SCOPE.open("w", encoding="utf-8") as f:
    for r in target:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

base_acc = correct / len(scope_rows)
BASE_JSON.write_text(json.dumps({
    "n_samples": len(scope_rows),
    "base_acc": base_acc,
    "target_n": len(target),
    "target_rate": len(target) / len(scope_rows),
}, ensure_ascii=False, indent=2), encoding="utf-8")

print("saved:", BASE_DETAILS)
print("saved:", TARGET_IDS)
print("saved:", TARGET_SCOPE)
print("base_acc =", base_acc)
print("target_n =", len(target), "/", len(scope_rows))
PY

echo "========== Step 3: generate extra candidates =========="
for SEED in 42 101 202
do
  OUT=${TRAJ_DIR}/${TAG}_extra_seed${SEED}.jsonl
  LOG=${LOG_DIR}/generate_${TAG}_extra_seed${SEED}.log

  python scripts/generate_numeric_trajectories_resume.py \
    --input "$TARGET_SCOPE" \
    --output "$OUT" \
    --generator_config "$CFG" \
    --dataset mathqa \
    --n_traj 4 \
    --max_samples 0 \
    --max_new_tokens 1024 \
    --temperature 0.95 \
    --top_p 0.95 \
    --seed "$SEED" \
    2>&1 | tee "$LOG"

  echo "rows $(wc -l < "$OUT") $OUT"
done

echo "========== Step 4: optionmap confirm sweep =========="
python - <<'PY'
import json, re
from pathlib import Path
from collections import Counter, defaultdict

LABELS = ["a","b","c","d","e"]

BASE_DETAILS = Path("outputs/predictions/model_ablation_14b/mathqa_deepseek14b_base_details.jsonl")
TARGET_SCOPE = Path("data/processed/unified/model_ablation_14b/mathqa_deepseek14b_has_disagreement.jsonl")
METRIC_DIR = Path("outputs/metrics/model_ablation_mathqa_optionmap_14b")
PRED_DIR = Path("outputs/predictions/model_ablation_mathqa_optionmap_14b")
METRIC_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

EXTRA_FILES = [
    Path("data/processed/trajectories/model_ablation_14b/mathqa_deepseek14b_extra_seed42.jsonl"),
    Path("data/processed/trajectories/model_ablation_14b/mathqa_deepseek14b_extra_seed101.jsonl"),
    Path("data/processed/trajectories/model_ablation_14b/mathqa_deepseek14b_extra_seed202.jsonl"),
]

def sid(r):
    return str(r.get("sample_id") or r.get("id") or "")

def norm_text(x):
    return re.sub(r"\s+", " ", str(x or "").strip().lower()).strip(" .。,:：;；")

def parse_options(r):
    opts = {}
    v = r.get("choices") or r.get("options") or r.get("answer_choices")
    if isinstance(v, dict):
        for k, val in v.items():
            lab = str(k).strip().lower()[:1]
            if lab in LABELS:
                opts[lab] = str(val)
    elif isinstance(v, list):
        for i, val in enumerate(v[:5]):
            opts[LABELS[i]] = str(val)
    elif isinstance(v, str):
        pat = re.compile(r"(?i)(?:^|[\s,;])([abcde])\s*[\)\.：:]\s*(.*?)(?=(?:[\s,;]+[abcde]\s*[\)\.：:])|$)")
        for lab, val in pat.findall(" " + v):
            opts[lab.lower()] = val.strip(" ,;")
    return opts

def ans_to_label(ans, opts):
    s = norm_text(ans)
    m = re.match(r"^\(?\s*([abcde])\s*\)?$", s)
    if m:
        return m.group(1)
    m = re.match(r"^\(?\s*([abcde])\s*\)?[\.\):：\s]", s)
    if m:
        return m.group(1)
    for lab, val in opts.items():
        if norm_text(val) == s:
            return lab
    for lab, val in opts.items():
        nv = norm_text(val)
        if nv and (nv in s or s in nv):
            return lab
    return ""

def extract_answer(r, opts):
    for k in ["answer","final_answer","pred_answer","prediction","extracted_answer"]:
        if r.get(k) is not None:
            lab = ans_to_label(r.get(k), opts)
            if lab:
                return lab
    txt = ""
    for k in ["trajectory","text","reasoning","output","completion","response"]:
        if r.get(k):
            txt = str(r[k])
            break
    for p in [r"Final Answer\s*[:：]\s*([^\n]+)", r"Answer\s*[:：]\s*([^\n]+)", r"答案\s*[:：]\s*([^\n]+)"]:
        m = re.findall(p, txt, flags=re.I)
        if m:
            lab = ans_to_label(m[-1], opts)
            if lab:
                return lab
    return ans_to_label(txt[-300:], opts)

def majority_count(counter):
    if not counter:
        return "", 0
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]

base_rows = [json.loads(x) for x in BASE_DETAILS.open(encoding="utf-8") if x.strip()]
base_by_sid = {sid(r): r for r in base_rows}
target_rows = [json.loads(x) for x in TARGET_SCOPE.open(encoding="utf-8") if x.strip()]
opts_by_sid = {sid(r): parse_options(r) for r in target_rows}

extra = defaultdict(list)
extra_by_seed = defaultdict(lambda: defaultdict(list))

for seed_idx, fp in enumerate(EXTRA_FILES):
    seed_name = fp.stem.split("seed")[-1]
    for line in fp.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        s = sid(r)
        if s not in opts_by_sid:
            continue
        lab = extract_answer(r, opts_by_sid[s])
        if lab:
            extra[s].append(lab)
            extra_by_seed[s][seed_name].append(lab)

n_samples = len(base_rows)
base_acc = sum(int(r.get("majority_ok", 0)) for r in base_rows) / n_samples

summary = []

for total_thr in [2, 3, 4]:
    for seed_thr in [1, 2, 3]:
        for margin in [0, 1, 2]:
            rows = []
            fixed = broken = changed = 0

            for tr in target_rows:
                s = sid(tr)
                b = base_by_sid[s]
                current = b.get("majority_answer", "")
                gold = b.get("gold_answer", "")
                current_ok = int(current == gold and gold != "")

                cnt = Counter(extra.get(s, []))
                top, top_count = majority_count(cnt)
                current_count = cnt.get(current, 0)

                seed_support = 0
                for seed_name, vals in extra_by_seed[s].items():
                    if top and top in vals:
                        seed_support += 1

                replace = (
                    top
                    and top != current
                    and top_count >= total_thr
                    and seed_support >= seed_thr
                    and (top_count - current_count) >= margin
                )

                final = top if replace else current
                final_ok = int(final == gold and gold != "")
                chg = int(final != current)
                fx = int(chg and (not current_ok) and final_ok)
                br = int(chg and current_ok and (not final_ok))

                fixed += fx
                broken += br
                changed += chg

                rows.append({
                    "sample_id": s,
                    "gold_answer": gold,
                    "current_answer": current,
                    "final_answer": final,
                    "current_ok": current_ok,
                    "final_ok": final_ok,
                    "changed": chg,
                    "fixed": fx,
                    "broken": br,
                    "top": top,
                    "top_total": top_count,
                    "top_seed": seed_support,
                    "runner_total": cnt.most_common(2)[1][1] if len(cnt) > 1 else 0,
                    "margin": margin,
                    "extra_support": dict(cnt),
                })

            net = fixed - broken
            final_acc = base_acc + net / n_samples
            gain = final_acc - base_acc

            name = f"mathqa_deepseek14b_optionmap_total{total_thr}_seed{seed_thr}_margin{margin}"
            pred_fp = PRED_DIR / f"{name}.jsonl"
            metric_fp = METRIC_DIR / f"{name}.json"

            with pred_fp.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            metric = {
                "dataset": "mathqa",
                "model": "deepseek14b",
                "base_acc": base_acc,
                "final_acc": final_acc,
                "gain": gain,
                "n_samples": n_samples,
                "n_eval": len(target_rows),
                "target_n": len(target_rows),
                "changed": changed,
                "fixed": fixed,
                "broken": broken,
                "net": net,
                "total_thr": total_thr,
                "seed_thr": seed_thr,
                "margin": margin,
                "extra_per_target": 12,
                "extra_per_sample": 12 * len(target_rows) / n_samples,
                "prediction_file": str(pred_fp),
            }
            metric_fp.write_text(json.dumps(metric, ensure_ascii=False, indent=2), encoding="utf-8")
            summary.append((final_acc, gain, net, fixed, broken, changed, str(metric_fp)))

summary.sort(reverse=True)
out = METRIC_DIR / "ds14b_mathqa_optionmap_summary.md"
lines = []
lines.append("# DS14B MathQA OptionMap Summary")
lines.append("")
lines.append("| file | base | final | gain | changed | fixed | broken | net |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for final_acc, gain, net, fixed, broken, changed, fp in summary:
    lines.append(f"| `{fp}` | {base_acc:.4f} | {final_acc:.4f} | {gain:.4f} | {changed} | {fixed} | {broken} | {net} |")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out.read_text(encoding="utf-8"))
PY

echo "========== DONE =========="
