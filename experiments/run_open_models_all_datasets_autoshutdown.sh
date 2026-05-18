#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

NO_SHUTDOWN_FILE=/tmp/NO_OPEN_MODELS_ALL_DATASETS_SHUTDOWN
SHUTDOWN_ON_ERROR=${SHUTDOWN_ON_ERROR:-1}

# 默认先跑 smoke。确认没问题后可以改成 MODE=full。
MODE=${MODE:-smoke}

# 如果 llama 权限有问题，可以先 MODELS_TO_RUN="deepseek7b qwen3b"
MODELS_TO_RUN=${MODELS_TO_RUN:-"deepseek7b llama3b qwen3b"}

mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/pce_backups
mkdir -p configs/model
mkdir -p outputs/logs/model_ablation
mkdir -p outputs/metrics/model_ablation
mkdir -p outputs/predictions/model_ablation
mkdir -p outputs/logs/final_summaries
mkdir -p outputs/targets/model_ablation
mkdir -p data/processed/unified/model_ablation
mkdir -p data/processed/trajectories/model_ablation

on_exit() {
  status=$?
  echo "========== backup open models all datasets =========="
  tar --ignore-failed-read -czf /root/autodl-tmp/pce_backups/open_models_all_datasets_${MODE}_$(date +%Y%m%d_%H%M%S).tar.gz \
    outputs/logs/model_ablation \
    outputs/metrics/model_ablation \
    outputs/predictions/model_ablation \
    outputs/logs/final_summaries/open_models_all_datasets_${MODE}_summary.md \
    data/processed/trajectories/model_ablation \
    configs/model/generator_deepseek_r1_distill_qwen7b_ablation.yaml \
    configs/model/generator_llama32_3b_ablation.yaml \
    configs/model/generator_qwen25_3b_ablation.yaml || true

  if [ -f "$NO_SHUTDOWN_FILE" ]; then
    echo "[NO SHUTDOWN] found $NO_SHUTDOWN_FILE"
    exit $status
  fi

  if [ "$status" = "0" ] || [ "$SHUTDOWN_ON_ERROR" = "1" ]; then
    echo "[SHUTDOWN] status=$status, shutting down."
    sync
    shutdown -h now || poweroff || halt || true
  else
    echo "[NO SHUTDOWN] failed with status=$status and SHUTDOWN_ON_ERROR=0"
  fi

  exit $status
}
trap on_exit EXIT

echo "========== Step 0: install/check huggingface_hub =========="
pip show huggingface_hub >/dev/null 2>&1 || pip install -U huggingface_hub

echo "========== Step 1: download models =========="

download_model() {
  local tag="$1"
  local repo="$2"
  local out="$3"

  echo "---------- $tag ----------"
  echo "repo=$repo"
  echo "out=$out"

  if [ -f "$out/config.json" ]; then
    echo "[SKIP] $tag already exists: $out"
    return 0
  fi

  mkdir -p "$out"
  hf download "$repo" --local-dir "$out"
  test -f "$out/config.json"
}

for TAG in $MODELS_TO_RUN
do
  case "$TAG" in
    deepseek7b)
      download_model deepseek7b deepseek-ai/DeepSeek-R1-Distill-Qwen-7B /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-7B
      ;;
    llama3b)
      download_model llama3b meta-llama/Llama-3.2-3B-Instruct /root/autodl-tmp/models/Llama-3.2-3B-Instruct
      ;;
    qwen3b)
      download_model qwen3b Qwen/Qwen2.5-3B-Instruct /root/autodl-tmp/models/Qwen2.5-3B-Instruct
      ;;
    *)
      echo "[WARN] unknown model tag: $TAG"
      ;;
  esac
done

echo "========== Step 2: create yaml configs =========="

cat > scripts/make_ablation_generator_yaml.py <<'PY'
import argparse
from pathlib import Path
import yaml

def patch_model_path(obj, model_path):
    hit = False
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in {
                "model",
                "model_path",
                "model_name",
                "model_name_or_path",
                "pretrained_model_name_or_path",
            }:
                obj[k] = model_path
                hit = True
            else:
                if patch_model_path(v, model_path):
                    hit = True
    elif isinstance(obj, list):
        for v in obj:
            if patch_model_path(v, model_path):
                hit = True
    return hit

ap = argparse.ArgumentParser()
ap.add_argument("--template", default="configs/model/generator_llama_local_rewrite.yaml")
ap.add_argument("--out", required=True)
ap.add_argument("--model_path", required=True)
args = ap.parse_args()

cfg = yaml.safe_load(open(args.template, "r", encoding="utf-8")) or {}
hit = patch_model_path(cfg, args.model_path)
if not hit:
    cfg["model_name_or_path"] = args.model_path

# 沿用当前项目已经跑通的单卡配置
cfg["max_model_len"] = 2048
cfg["gpu_memory_utilization"] = 0.60
cfg["enforce_eager"] = True
cfg["trust_remote_code"] = True
cfg["tensor_parallel_size"] = 1

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

print("saved:", args.out)
print("model_path:", args.model_path)
PY

for TAG in $MODELS_TO_RUN
do
  case "$TAG" in
    deepseek7b)
      python scripts/make_ablation_generator_yaml.py \
        --template configs/model/generator_llama_local_rewrite.yaml \
        --out configs/model/generator_deepseek_r1_distill_qwen7b_ablation.yaml \
        --model_path /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-7B
      ;;
    llama3b)
      python scripts/make_ablation_generator_yaml.py \
        --template configs/model/generator_llama_local_rewrite.yaml \
        --out configs/model/generator_llama32_3b_ablation.yaml \
        --model_path /root/autodl-tmp/models/Llama-3.2-3B-Instruct
      ;;
    qwen3b)
      python scripts/make_ablation_generator_yaml.py \
        --template configs/model/generator_llama_local_rewrite.yaml \
        --out configs/model/generator_qwen25_3b_ablation.yaml \
        --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct
      ;;
  esac
done

echo "===== yaml check ====="
grep -R "model_name_or_path" -n configs/model/*ablation.yaml || true

echo "========== Step 3: write model-ablation evaluators =========="

cat > experiments/eval_numeric_baseline_model_ablation.py <<'PY'
import argparse, json, re
from pathlib import Path
from collections import defaultdict, Counter

def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]

def clean(x):
    s = str(x or "").strip().replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return s.lower().strip()
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y

def ok(a, g):
    return clean(a) == clean(g)

ap = argparse.ArgumentParser()
ap.add_argument("--trajectories", required=True)
ap.add_argument("--out_json", required=True)
ap.add_argument("--out_jsonl", required=True)
args = ap.parse_args()

rows = read_jsonl(args.trajectories)
by = defaultdict(list)
for r in rows:
    by[r["sample_id"]].append(r)

first = majority = anyok = has_dis = all_dis = 0
details = []

for sid, rs in sorted(by.items()):
    rs = sorted(rs, key=lambda x: str(x.get("trajectory_id", "")))
    gold = rs[0].get("gold_answer", rs[0].get("answer", ""))
    answers = [r.get("final_answer", "") for r in rs]
    answers_norm = [clean(a) for a in answers]
    cnt = Counter(a for a in answers_norm if a)
    maj = cnt.most_common(1)[0][0] if cnt else ""

    first_ok = int(ok(answers[0] if answers else "", gold))
    maj_ok = int(ok(maj, gold))
    any_ok = int(any(ok(a, gold) for a in answers))

    first += first_ok
    majority += maj_ok
    anyok += any_ok

    uniq = set(a for a in answers_norm if a)
    has_dis += int(len(uniq) >= 2)
    all_dis += int(len(uniq) >= 3)

    details.append({
        "sample_id": sid,
        "gold_answer": gold,
        "answers": answers,
        "answers_norm": answers_norm,
        "majority_answer": maj,
        "first_ok": first_ok,
        "majority_ok": maj_ok,
        "oracle_any_ok": any_ok,
    })

n = len(by)
summary = {
    "n_samples": n,
    "n_trajectories": len(rows),
    "first_acc": first / max(1, n),
    "majority_acc": majority / max(1, n),
    "oracle_any_acc": anyok / max(1, n),
    "has_disagreement": has_dis,
    "all_disagree": all_dis,
}
Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
with open(args.out_jsonl, "w", encoding="utf-8") as f:
    for r in details:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

cat > experiments/apply_numeric_confirm_model_ablation.py <<'PY'
import argparse, json, re
from pathlib import Path
from collections import defaultdict, Counter

def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]

def clean(x):
    s = str(x or "").strip().replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return s.lower().strip()
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y

def ok(a, g):
    return clean(a) == clean(g)

ap = argparse.ArgumentParser()
ap.add_argument("--baseline_details", required=True)
ap.add_argument("--extra_jsonls", nargs="+", required=True)
ap.add_argument("--target_ids", required=True)
ap.add_argument("--base_acc", type=float, required=True)
ap.add_argument("--n_samples", type=int, required=True)
ap.add_argument("--min_total_support", type=int, default=2)
ap.add_argument("--min_seed_support", type=int, default=2)
ap.add_argument("--min_margin", type=int, default=1)
ap.add_argument("--out_json", required=True)
ap.add_argument("--out_jsonl", required=True)
args = ap.parse_args()

base = {r["sample_id"]: r for r in read_jsonl(args.baseline_details)}
target_ids = [x.strip() for x in open(args.target_ids, encoding="utf-8") if x.strip()]

extras = defaultdict(list)
for seed_idx, fp in enumerate(args.extra_jsonls):
    for r in read_jsonl(fp):
        sid = r["sample_id"]
        ans = clean(r.get("final_answer", ""))
        if ans:
            extras[sid].append((seed_idx, ans))

fixed = broken = changed = 0
cur_correct = final_correct = 0

for sid in target_ids:
    b = base[sid]
    gold = b["gold_answer"]
    cur = clean(b["majority_answer"])
    cur_ok = int(ok(cur, gold))

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
    final = cur
    if top and top != cur and top_total >= args.min_total_support and top_seed >= args.min_seed_support and margin >= args.min_margin:
        final = top

    fin_ok = int(ok(final, gold))
    fixed += int(cur_ok == 0 and fin_ok == 1)
    broken += int(cur_ok == 1 and fin_ok == 0)
    changed += int(final != cur)
    cur_correct += cur_ok
    final_correct += fin_ok

net = fixed - broken
summary = {
    "base_acc": args.base_acc,
    "n_samples": args.n_samples,
    "n_eval": len(target_ids),
    "min_total_support": args.min_total_support,
    "min_seed_support": args.min_seed_support,
    "min_margin": args.min_margin,
    "current_acc_on_eval": cur_correct / max(1, len(target_ids)),
    "final_acc_on_eval": final_correct / max(1, len(target_ids)),
    "fixed": fixed,
    "broken": broken,
    "net": net,
    "changed": changed,
    "estimated_global_acc": args.base_acc + net / args.n_samples,
}
Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
with open(args.out_jsonl, "w", encoding="utf-8") as f:
    f.write("")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

cat > experiments/eval_bbh_fixed_baseline_model_ablation.py <<'PY'
import argparse, json, re, string
from pathlib import Path
from collections import defaultdict, Counter

def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]

def extract_option(s):
    raw = str(s or "")
    x = raw.lower().strip()
    fa = re.findall(r"final answer\s*[:：]\s*([^\n\|]+)", raw, flags=re.I)
    candidates = fa + [raw]
    for c in reversed(candidates):
        c = str(c or "").strip()
        m = re.search(r"\(([A-Ea-e])\)", c)
        if m:
            return m.group(1).lower()
        m = re.search(r"\boption\s*([A-Ea-e])\b", c, flags=re.I)
        if m:
            return m.group(1).lower()
        m = re.search(r"^\s*([A-Ea-e])[\)\.\:]\s*", c)
        if m:
            return m.group(1).lower()
    m = re.search(r"^\s*([a-e])\b", x)
    if m:
        return m.group(1)
    x = "".join(ch for ch in x if ch not in string.punctuation)
    return " ".join(x.split())

ap = argparse.ArgumentParser()
ap.add_argument("--trajectories", required=True)
ap.add_argument("--out_json", required=True)
ap.add_argument("--out_jsonl", required=True)
args = ap.parse_args()

rows = read_jsonl(args.trajectories)
by = defaultdict(list)
for r in rows:
    by[r["sample_id"]].append(r)

first = majority = anyok = has_dis = all_dis = 0
details = []

for sid, rs in sorted(by.items()):
    rs = sorted(rs, key=lambda x: str(x.get("trajectory_id", "")))
    gold = rs[0].get("gold_answer", rs[0].get("answer", ""))
    gold_norm = extract_option(gold)
    answers = [r.get("final_answer", "") for r in rs]
    answers_norm = [extract_option(a) for a in answers]
    cnt = Counter(a for a in answers_norm if a)
    maj = cnt.most_common(1)[0][0] if cnt else ""

    first_ok = int((answers_norm[0] if answers_norm else "") == gold_norm)
    maj_ok = int(maj == gold_norm)
    any_ok = int(any(a == gold_norm for a in answers_norm))

    first += first_ok
    majority += maj_ok
    anyok += any_ok

    uniq = set(a for a in answers_norm if a)
    has_dis += int(len(uniq) >= 2)
    all_dis += int(len(uniq) >= 3)

    details.append({
        "sample_id": sid,
        "gold_answer": gold,
        "gold_norm_fixed": gold_norm,
        "answers": answers,
        "answers_norm_fixed": answers_norm,
        "majority_answer_fixed": maj,
        "first_ok_fixed": first_ok,
        "majority_ok_fixed": maj_ok,
        "oracle_any_ok_fixed": any_ok,
    })

n = len(by)
summary = {
    "n_samples": n,
    "n_trajectories": len(rows),
    "first_acc_fixed": first / max(1, n),
    "majority_acc_fixed": majority / max(1, n),
    "oracle_any_acc_fixed": anyok / max(1, n),
    "has_disagreement_fixed": has_dis,
    "all_disagree_fixed": all_dis,
}
Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
with open(args.out_jsonl, "w", encoding="utf-8") as f:
    for r in details:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "========== Step 4: create dataset scopes =========="

python - <<'PY'
import json, re
from pathlib import Path

def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]

def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(fp, len(rows))

def is_numeric(r):
    s = str(r.get("gold_answer", r.get("answer", "")))
    return re.search(r"[-+]?\d+(?:\.\d+)?", s) is not None

# MODE=smoke 时控制规模；MODE=full 时尽量跑主范围
mode = "${MODE}"

# GSM8K
gsm = read_jsonl("data/processed/unified/gsm8k/test.jsonl")
if gsm:
    write_jsonl("data/processed/unified/model_ablation/gsm8k_scope.jsonl", gsm if mode == "full" else gsm[:300])

# SVAMP
svamp = read_jsonl("data/processed/unified/svamp/test.jsonl")
if svamp:
    write_jsonl("data/processed/unified/model_ablation/svamp_scope.jsonl", svamp if mode == "full" else svamp[:300])

# ASDiv numeric
asdiv = read_jsonl("data/processed/unified/asdiv/test_numeric_full.jsonl")
if not asdiv:
    asdiv = [r for r in read_jsonl("data/processed/unified/asdiv/test.jsonl") if is_numeric(r)]
write_jsonl("data/processed/unified/model_ablation/asdiv_numeric_scope.jsonl", asdiv if mode == "full" else asdiv[:300])

# MATH500
math500 = read_jsonl("data/processed/unified/math500/test.jsonl")
if math500:
    write_jsonl("data/processed/unified/model_ablation/math500_scope.jsonl", math500 if mode == "full" else math500[:100])

# MathQA
mathqa = read_jsonl("data/processed/unified/mathqa/test.jsonl")
if mathqa:
    write_jsonl("data/processed/unified/model_ablation/mathqa_scope.jsonl", mathqa[:500] if mode == "full" else mathqa[:100])

# BBH
for task in ["logical_deduction_five_objects", "formal_fallacies"]:
    bbh = read_jsonl(f"data/processed/unified/bbh_logic/{task}.jsonl")
    if bbh:
        write_jsonl(f"data/processed/unified/model_ablation/bbh_{task}_scope.jsonl", bbh[:100])
PY

echo "========== Step 5: run model × datasets =========="

get_cfg() {
  case "$1" in
    deepseek7b) echo "configs/model/generator_deepseek_r1_distill_qwen7b_ablation.yaml" ;;
    llama3b) echo "configs/model/generator_llama32_3b_ablation.yaml" ;;
    qwen3b) echo "configs/model/generator_qwen25_3b_ablation.yaml" ;;
  esac
}

run_numeric_dataset() {
  local TAG="$1"
  local CFG="$2"
  local DS="$3"
  local INPUT="$4"
  local MAX_NEW="${5:-384}"

  echo "---------- $TAG $DS base ----------"
  python scripts/generate_numeric_trajectories_local.py \
    --input "$INPUT" \
    --output data/processed/trajectories/model_ablation/${DS}_${TAG}_base_3traj.jsonl \
    --generator_config "$CFG" \
    --dataset "$DS" \
    --n_traj 3 \
    --max_samples 0 \
    --max_new_tokens "$MAX_NEW" \
    --temperature 0.7 \
    --top_p 0.95 \
    --seed 42 \
    2>&1 | tee outputs/logs/model_ablation/generate_${DS}_${TAG}_base.log

  python experiments/eval_numeric_baseline_model_ablation.py \
    --trajectories data/processed/trajectories/model_ablation/${DS}_${TAG}_base_3traj.jsonl \
    --out_json outputs/metrics/model_ablation/${DS}_${TAG}_base.json \
    --out_jsonl outputs/predictions/model_ablation/${DS}_${TAG}_base_details.jsonl

  python - <<PY
import json
from pathlib import Path
tag="$TAG"; ds="$DS"; inp="$INPUT"
rows=[json.loads(x) for x in open(f"outputs/predictions/model_ablation/{ds}_{tag}_base_details.jsonl", encoding="utf-8") if x.strip()]
target_ids=[]
for r in rows:
    vals=[str(a).strip() for a in r.get("answers_norm", []) if str(a).strip()]
    if len(set(vals))>=2:
        target_ids.append(r["sample_id"])
Path("outputs/targets/model_ablation").mkdir(parents=True, exist_ok=True)
with open(f"outputs/targets/model_ablation/{ds}_{tag}_has_disagreement_ids.txt","w",encoding="utf-8") as f:
    for sid in target_ids:
        f.write(sid+"\\n")
unified=[json.loads(x) for x in open(inp, encoding="utf-8") if x.strip()]
target=set(target_ids)
subset=[r for r in unified if (r.get("sample_id") or r.get("id")) in target]
out=Path(f"data/processed/unified/model_ablation/{ds}_{tag}_has_disagreement.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w",encoding="utf-8") as f:
    for r in subset:
        f.write(json.dumps(r, ensure_ascii=False)+"\\n")
print(ds, tag, "targets", len(target_ids))
PY

  echo "---------- $TAG $DS extra ----------"
  for SEED in 42 101 202
  do
    python scripts/generate_numeric_trajectories_local.py \
      --input data/processed/unified/model_ablation/${DS}_${TAG}_has_disagreement.jsonl \
      --output data/processed/trajectories/model_ablation/${DS}_${TAG}_extra_seed${SEED}.jsonl \
      --generator_config "$CFG" \
      --dataset "$DS" \
      --n_traj 4 \
      --max_samples 0 \
      --max_new_tokens "$MAX_NEW" \
      --temperature 0.95 \
      --top_p 0.95 \
      --seed "$SEED" \
      2>&1 | tee outputs/logs/model_ablation/generate_${DS}_${TAG}_extra_seed${SEED}.log
  done

  BASE_ACC=$(python - <<PY
import json
x=json.load(open("outputs/metrics/model_ablation/${DS}_${TAG}_base.json", encoding="utf-8"))
print(x["majority_acc"])
PY
)

  EXTRA_FILES="data/processed/trajectories/model_ablation/${DS}_${TAG}_extra_seed42.jsonl data/processed/trajectories/model_ablation/${DS}_${TAG}_extra_seed101.jsonl data/processed/trajectories/model_ablation/${DS}_${TAG}_extra_seed202.jsonl"

  for TOTAL in 2 3 4
  do
    for SEEDSUP in 1 2 3
    do
      for MARGIN in 0 1 2
      do
        python experiments/apply_numeric_confirm_model_ablation.py \
          --baseline_details outputs/predictions/model_ablation/${DS}_${TAG}_base_details.jsonl \
          --extra_jsonls $EXTRA_FILES \
          --target_ids outputs/targets/model_ablation/${DS}_${TAG}_has_disagreement_ids.txt \
          --base_acc "$BASE_ACC" \
          --n_samples $(wc -l < "$INPUT") \
          --min_total_support "$TOTAL" \
          --min_seed_support "$SEEDSUP" \
          --min_margin "$MARGIN" \
          --out_json outputs/metrics/model_ablation/${DS}_${TAG}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.json \
          --out_jsonl outputs/predictions/model_ablation/${DS}_${TAG}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.jsonl
      done
    done
  done
}

run_bbh_task() {
  local TAG="$1"
  local CFG="$2"
  local TASK="$3"
  local INPUT="data/processed/unified/model_ablation/bbh_${TASK}_scope.jsonl"

  echo "---------- $TAG BBH $TASK base ----------"
  python scripts/generate_bbh_logic_trajectories_vllm.py \
    --input "$INPUT" \
    --output data/processed/trajectories/model_ablation/bbh_${TASK}_${TAG}_base_3traj.jsonl \
    --generator_config "$CFG" \
    --n_traj 3 \
    --max_samples 0 \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.95 \
    --seed 42 \
    --batch_size 4 \
    2>&1 | tee outputs/logs/model_ablation/generate_bbh_${TASK}_${TAG}_base.log

  python experiments/eval_bbh_fixed_baseline_model_ablation.py \
    --trajectories data/processed/trajectories/model_ablation/bbh_${TASK}_${TAG}_base_3traj.jsonl \
    --out_json outputs/metrics/model_ablation/bbh_${TASK}_${TAG}_base_fixed.json \
    --out_jsonl outputs/predictions/model_ablation/bbh_${TASK}_${TAG}_base_fixed_details.jsonl

  python - <<PY
import json
from pathlib import Path
tag="$TAG"; task="$TASK"; inp="$INPUT"
rows=[json.loads(x) for x in open(f"outputs/predictions/model_ablation/bbh_{task}_{tag}_base_fixed_details.jsonl", encoding="utf-8") if x.strip()]
target_ids=[]
for r in rows:
    vals=[str(a).strip() for a in r.get("answers_norm_fixed", []) if str(a).strip()]
    if len(set(vals))>=2:
        target_ids.append(r["sample_id"])
Path("outputs/targets/model_ablation").mkdir(parents=True, exist_ok=True)
with open(f"outputs/targets/model_ablation/bbh_{task}_{tag}_has_disagreement_ids.txt","w",encoding="utf-8") as f:
    for sid in target_ids:
        f.write(sid+"\\n")
unified=[json.loads(x) for x in open(inp, encoding="utf-8") if x.strip()]
target=set(target_ids)
subset=[r for r in unified if (r.get("sample_id") or r.get("id")) in target]
out=Path(f"data/processed/unified/model_ablation/bbh_{task}_{tag}_has_disagreement.jsonl")
with out.open("w",encoding="utf-8") as f:
    for r in subset:
        f.write(json.dumps(r, ensure_ascii=False)+"\\n")
print("bbh", task, tag, "targets", len(target_ids))
PY

  for SEED in 303 404 505 606 707 808
  do
    python scripts/generate_bbh_logic_trajectories_vllm.py \
      --input data/processed/unified/model_ablation/bbh_${TASK}_${TAG}_has_disagreement.jsonl \
      --output data/processed/trajectories/model_ablation/bbh_${TASK}_${TAG}_extra_seed${SEED}.jsonl \
      --generator_config "$CFG" \
      --n_traj 1 \
      --max_samples 0 \
      --max_new_tokens 512 \
      --temperature 0.9 \
      --top_p 0.95 \
      --seed "$SEED" \
      --batch_size 4 \
      2>&1 | tee outputs/logs/model_ablation/generate_bbh_${TASK}_${TAG}_extra_seed${SEED}.log
  done

  BASE_ACC=$(python - <<PY
import json
x=json.load(open("outputs/metrics/model_ablation/bbh_${TASK}_${TAG}_base_fixed.json", encoding="utf-8"))
print(x["majority_acc_fixed"])
PY
)

  EXTRA_FILES=""
  for SEED in 303 404 505 606 707 808
  do
    EXTRA_FILES="$EXTRA_FILES data/processed/trajectories/model_ablation/bbh_${TASK}_${TAG}_extra_seed${SEED}.jsonl"
  done

  for TOTAL in 2 3 4
  do
    for SEEDSUP in 1 2 3
    do
      for MARGIN in 0 1 2
      do
        python experiments/apply_bbh_fixed_resample_confirm.py \
          --baseline_fixed_details outputs/predictions/model_ablation/bbh_${TASK}_${TAG}_base_fixed_details.jsonl \
          --extra_jsonls $EXTRA_FILES \
          --target_ids outputs/targets/model_ablation/bbh_${TASK}_${TAG}_has_disagreement_ids.txt \
          --subtask "$TASK" \
          --min_total_support "$TOTAL" \
          --min_seed_support "$SEEDSUP" \
          --min_margin "$MARGIN" \
          --base_acc "$BASE_ACC" \
          --n_samples $(wc -l < "$INPUT") \
          --out_json outputs/metrics/model_ablation/bbh_${TASK}_${TAG}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.json \
          --out_jsonl outputs/predictions/model_ablation/bbh_${TASK}_${TAG}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.jsonl
      done
    done
  done
}

for TAG in $MODELS_TO_RUN
do
  CFG=$(get_cfg "$TAG")
  echo "==================== MODEL $TAG CFG=$CFG ===================="

  run_numeric_dataset "$TAG" "$CFG" "gsm8k" "data/processed/unified/model_ablation/gsm8k_scope.jsonl" 384
  run_numeric_dataset "$TAG" "$CFG" "svamp" "data/processed/unified/model_ablation/svamp_scope.jsonl" 384
  run_numeric_dataset "$TAG" "$CFG" "asdiv" "data/processed/unified/model_ablation/asdiv_numeric_scope.jsonl" 384

  if [ -s data/processed/unified/model_ablation/math500_scope.jsonl ]; then
    run_numeric_dataset "$TAG" "$CFG" "math500" "data/processed/unified/model_ablation/math500_scope.jsonl" 512
  fi

  if [ -s data/processed/unified/model_ablation/mathqa_scope.jsonl ]; then
    run_numeric_dataset "$TAG" "$CFG" "mathqa" "data/processed/unified/model_ablation/mathqa_scope.jsonl" 512
  fi

  run_bbh_task "$TAG" "$CFG" "logical_deduction_five_objects"
  run_bbh_task "$TAG" "$CFG" "formal_fallacies"
done

echo "========== Step 6: summarize =========="

python - <<'PY' | tee outputs/logs/final_summaries/open_models_all_datasets_${MODE}_summary.md
import json
from pathlib import Path

mode = "${MODE}"

def rows(pattern):
    out = []
    for fp in Path(".").glob(pattern):
        try:
            x = json.load(open(fp, encoding="utf-8"))
            x["_fp"] = str(fp)
            out.append(x)
        except Exception:
            pass
    return out

def best(xs, key="estimated_global_acc"):
    if not xs:
        return None
    return sorted(xs, key=lambda x: (-x[key], x.get("broken",999), -x.get("fixed",-999), x.get("changed",999)))[0]

print(f"# Open model all-dataset comparison summary ({mode})\n")
print("| Model | Dataset | Base | Best | Gain | fixed | broken | net | n_eval | setting |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

models = ["deepseek7b", "llama3b", "qwen3b"]
datasets = ["gsm8k", "svamp", "asdiv", "math500", "mathqa"]

for tag in models:
    for ds in datasets:
        base_fp = Path(f"outputs/metrics/model_ablation/{ds}_{tag}_base.json")
        if not base_fp.exists():
            continue
        b = json.load(open(base_fp, encoding="utf-8"))
        xs = rows(f"outputs/metrics/model_ablation/{ds}_{tag}_total*.json")
        x = best(xs)
        if not x:
            continue
        setting = f"total{x['min_total_support']}_seed{x['min_seed_support']}_margin{x['min_margin']}"
        print(f"| {tag} | {ds} | {b['majority_acc']:.4f} | {x['estimated_global_acc']:.4f} | {x['estimated_global_acc']-b['majority_acc']:+.4f} | {x['fixed']} | {x['broken']} | {x['net']} | {x['n_eval']} | {setting} |")

    for task in ["logical_deduction_five_objects", "formal_fallacies"]:
        base_fp = Path(f"outputs/metrics/model_ablation/bbh_{task}_{tag}_base_fixed.json")
        if not base_fp.exists():
            continue
        b = json.load(open(base_fp, encoding="utf-8"))
        xs = rows(f"outputs/metrics/model_ablation/bbh_{task}_{tag}_total*.json")
        x = best(xs)
        if not x:
            continue
        setting = f"total{x['min_total_support']}_seed{x['min_seed_support']}_margin{x['min_margin']}"
        print(f"| {tag} | bbh_{task} | {b['majority_acc_fixed']:.4f} | {x['estimated_global_acc']:.4f} | {x['estimated_global_acc']-b['majority_acc_fixed']:+.4f} | {x['fixed']} | {x['broken']} | {x['net']} | {x['n_eval']} | {setting} |")
PY

echo "========== DONE open models all datasets =========="
