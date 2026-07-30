#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

MODE=${MODE:-smoke}
MODELS_TO_RUN=${MODELS_TO_RUN:-"deepseek7b qwen3b llama3b"}
NO_SHUTDOWN_FILE=/tmp/NO_AUTODL_OPEN_MODEL_SHUTDOWN

mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/pce_backups
mkdir -p configs/model
mkdir -p outputs/logs/model_ablation_parallel_qwen3b
mkdir -p outputs/metrics/model_ablation_parallel_qwen3b
mkdir -p outputs/predictions/model_ablation_parallel_qwen3b
mkdir -p outputs/targets/model_ablation
mkdir -p outputs/logs/final_summaries
mkdir -p data/processed/unified/model_ablation_parallel_qwen3b
mkdir -p data/processed/trajectories/model_ablation_parallel_qwen3b

on_exit() {
  status=$?
  echo "========== EXIT status=$status =========="

  tar --ignore-failed-read -czf /root/autodl-tmp/pce_backups/open_model_all_datasets_${MODE}_$(date +%Y%m%d_%H%M%S).tar.gz \
    outputs/logs/model_ablation_parallel_qwen3b \
    outputs/metrics/model_ablation_parallel_qwen3b \
    outputs/predictions/model_ablation_parallel_qwen3b \
    outputs/targets/model_ablation \
    outputs/logs/final_summaries/open_model_all_datasets_${MODE}_summary.md \
    configs/model/*ablation*.yaml \
    data/processed/trajectories/model_ablation_parallel_qwen3b || true

  if [ "$status" = "0" ] && [ ! -f "$NO_SHUTDOWN_FILE" ]; then
    echo "[SHUTDOWN] success, shutting down."
    sync
    echo "[skip shutdown qwen3b parallel resume]" || echo "[skip poweroff qwen3b parallel resume]" || echo "[skip halt qwen3b parallel resume]" || true
  else
    echo "[NO SHUTDOWN] status=$status or found $NO_SHUTDOWN_FILE"
  fi
}
trap on_exit EXIT

echo "========== Step 0: environment =========="
date
python --version
pip show huggingface_hub >/dev/null 2>&1 || pip install -U huggingface_hub hf_transfer

echo "========== Step 1: optional git pull =========="
if [ "${GIT_PULL:-0}" = "1" ]; then
  git pull --ff-only || true
fi

echo "========== Step 2: check required code =========="
for FP in \
  scripts/generate_numeric_trajectories_local.py \
  scripts/generate_bbh_logic_trajectories_vllm.py \
  configs/model/generator_llama_local_rewrite.yaml
do
  if [ ! -f "$FP" ]; then
    echo "[MISSING CODE] $FP"
    exit 2
  fi
done

echo "========== Step 3: download models =========="

download_model() {
  local tag="$1"
  local repo="$2"
  local out="$3"

  echo "---------- download $tag ----------"
  echo "repo=$repo"
  echo "out=$out"

  if [ -f "$out/config.json" ] && find "$out" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" \) | grep -q .; then
    echo "[SKIP] model already complete enough: $out"
    return 0
  fi

  mkdir -p "$out"

  rm -f "$out/.cache/huggingface/download/"*.lock 2>/dev/null || true

  hf download "$repo" --local-dir "$out"

  test -f "$out/config.json"

  if ! find "$out" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" \) | grep -q .; then
    echo "[ERROR] no final weight file found after download: $out"
    exit 3
  fi
}

for TAG in $MODELS_TO_RUN
do
  case "$TAG" in
    deepseek7b)
      download_model deepseek7b deepseek-ai/DeepSeek-R1-Distill-Qwen-7B /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-7B
      ;;
    qwen3b)
      download_model qwen3b Qwen/Qwen2.5-3B-Instruct /root/autodl-tmp/models/Qwen2.5-3B-Instruct
      ;;
    llama3b)
      download_model llama3b meta-llama/Llama-3.2-3B-Instruct /root/autodl-tmp/models/Llama-3.2-3B-Instruct
      ;;
    *)
      echo "[WARN] unknown model tag: $TAG"
      ;;
  esac
done

echo "========== Step 4: create generator yamls =========="

cat > scripts/make_ablation_generator_yaml.py <<'PY'
import argparse
from pathlib import Path
import yaml

def patch_model_path(obj, model_path):
    hit = False
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in {"model", "model_path", "model_name", "model_name_or_path", "pretrained_model_name_or_path"}:
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

cfg["max_model_len"] = 2048
cfg["gpu_memory_utilization"] = 0.60
cfg["enforce_eager"] = True
cfg["trust_remote_code"] = True
cfg["tensor_parallel_size"] = 1

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

print("saved:", args.out)
PY

for TAG in $MODELS_TO_RUN
do
  case "$TAG" in
    deepseek7b)
      python scripts/make_ablation_generator_yaml.py \
        --out configs/model/generator_deepseek_r1_distill_qwen7b_ablation.yaml \
        --model_path /root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-7B
      ;;
    qwen3b)
      python scripts/make_ablation_generator_yaml.py \
        --out configs/model/generator_qwen25_3b_ablation.yaml \
        --model_path /root/autodl-tmp/models/Qwen2.5-3B-Instruct
      ;;
    llama3b)
      python scripts/make_ablation_generator_yaml.py \
        --out configs/model/generator_llama32_3b_ablation.yaml \
        --model_path /root/autodl-tmp/models/Llama-3.2-3B-Instruct
      ;;
  esac
done

grep -R "model_name_or_path" -n configs/model/*ablation*.yaml || true


echo "========== Step 4.5: patch qwen3b parallel035 config =========="
cp configs/model/generator_qwen25_3b_parallel035.yaml configs/model/generator_qwen25_3b_ablation.yaml
cat configs/model/generator_qwen25_3b_ablation.yaml

echo "========== Step 5: check datasets =========="

python - <<'PY'
from pathlib import Path

required_any = {
    "gsm8k": ["data/processed/unified/gsm8k/test.jsonl"],
    "svamp": ["data/processed/unified/svamp/test.jsonl"],
    "asdiv": ["data/processed/unified/asdiv/test_numeric_full.jsonl", "data/processed/unified/asdiv/test.jsonl"],
    "math500": ["data/processed/unified/math500/test.jsonl"],
    "mathqa": ["data/processed/unified/mathqa/test.jsonl"],
    "bbh_logical5": ["data/processed/unified/bbh_logic/logical_deduction_five_objects.jsonl"],
    "bbh_formal": ["data/processed/unified/bbh_logic/formal_fallacies.jsonl"],
}

missing = []
for name, cands in required_any.items():
    if not any(Path(p).exists() for p in cands):
        missing.append((name, cands))

if missing:
    print("========== MISSING DATA ==========")
    for name, cands in missing:
        print(name, "need one of:")
        for p in cands:
            print("  ", p)
    print()
    print("说明：git 通常不会带 data/processed/unified。请从旧机器拷贝 data/processed/unified/，或先运行你的数据准备脚本。")
    raise SystemExit(4)

print("all required dataset files exist.")
PY

echo "========== Step 6: create scopes =========="

python - <<'PY'
import json, re
from pathlib import Path

mode = "${MODE}"

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

def pick(rows, smoke_n):
    if mode == "full":
        return rows
    return rows[:smoke_n]

gsm = read_jsonl("data/processed/unified/gsm8k/test.jsonl")
write_jsonl("data/processed/unified/model_ablation_parallel_qwen3b/gsm8k_scope.jsonl", pick(gsm, 300))

svamp = read_jsonl("data/processed/unified/svamp/test.jsonl")
write_jsonl("data/processed/unified/model_ablation_parallel_qwen3b/svamp_scope.jsonl", pick(svamp, 300))

asdiv = read_jsonl("data/processed/unified/asdiv/test_numeric_full.jsonl")
if not asdiv:
    asdiv = [r for r in read_jsonl("data/processed/unified/asdiv/test.jsonl") if is_numeric(r)]
write_jsonl("data/processed/unified/model_ablation_parallel_qwen3b/asdiv_scope.jsonl", pick(asdiv, 300))

math500 = read_jsonl("data/processed/unified/math500/test.jsonl")
write_jsonl("data/processed/unified/model_ablation_parallel_qwen3b/math500_scope.jsonl", pick(math500, 100))

mathqa = read_jsonl("data/processed/unified/mathqa/test.jsonl")
write_jsonl("data/processed/unified/model_ablation_parallel_qwen3b/mathqa_scope.jsonl", mathqa[:500] if mode == "full" else mathqa[:100])

for task in ["logical_deduction_five_objects", "formal_fallacies"]:
    rows = read_jsonl(f"data/processed/unified/bbh_logic/{task}.jsonl")
    write_jsonl(f"data/processed/unified/model_ablation_parallel_qwen3b/bbh_{task}_scope.jsonl", rows[:100])
PY

echo "========== Step 6.5: force REALFULL scopes =========="
python scripts/force_realfull_model_ablation_scopes.py


echo "========== Step 6.6: force REALFULL scopes for parallel qwen3b =========="
mkdir -p data/processed/unified/model_ablation_parallel_qwen3b

cp data/processed/unified/model_ablation/gsm8k_scope.jsonl data/processed/unified/model_ablation_parallel_qwen3b/gsm8k_scope.jsonl
cp data/processed/unified/model_ablation/svamp_scope.jsonl data/processed/unified/model_ablation_parallel_qwen3b/svamp_scope.jsonl
cp data/processed/unified/model_ablation/asdiv_scope.jsonl data/processed/unified/model_ablation_parallel_qwen3b/asdiv_scope.jsonl
cp data/processed/unified/model_ablation/math500_scope.jsonl data/processed/unified/model_ablation_parallel_qwen3b/math500_scope.jsonl
cp data/processed/unified/model_ablation/mathqa_scope.jsonl data/processed/unified/model_ablation_parallel_qwen3b/mathqa_scope.jsonl
cp data/processed/unified/model_ablation/bbh_logical_deduction_five_objects_scope.jsonl data/processed/unified/model_ablation_parallel_qwen3b/bbh_logical_deduction_five_objects_scope.jsonl
cp data/processed/unified/model_ablation/bbh_formal_fallacies_scope.jsonl data/processed/unified/model_ablation_parallel_qwen3b/bbh_formal_fallacies_scope.jsonl

echo "===== forced parallel qwen3b scope counts ====="
wc -l data/processed/unified/model_ablation_parallel_qwen3b/gsm8k_scope.jsonl
wc -l data/processed/unified/model_ablation_parallel_qwen3b/svamp_scope.jsonl
wc -l data/processed/unified/model_ablation_parallel_qwen3b/asdiv_scope.jsonl
wc -l data/processed/unified/model_ablation_parallel_qwen3b/math500_scope.jsonl
wc -l data/processed/unified/model_ablation_parallel_qwen3b/mathqa_scope.jsonl
wc -l data/processed/unified/model_ablation_parallel_qwen3b/bbh_logical_deduction_five_objects_scope.jsonl
wc -l data/processed/unified/model_ablation_parallel_qwen3b/bbh_formal_fallacies_scope.jsonl

echo "========== Step 7: write evaluators =========="

cat > experiments/eval_base_model_ablation.py <<'PY'
import argparse, json, re, string
from pathlib import Path
from collections import defaultdict, Counter

def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]

def norm_num(x):
    s = str(x or "").replace(",", "").replace("$", "").strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return " ".join(s.lower().split())
    y = nums[-1]
    return y.rstrip("0").rstrip(".") if "." in y else y

def norm_choice(x):
    raw = str(x or "")
    candidates = re.findall(r"final answer\s*[:：]\s*([^\n\|]+)", raw, flags=re.I) + [raw]
    for c in reversed(candidates):
        m = re.search(r"\(([A-Ea-e])\)", c)
        if m: return m.group(1).lower()
        m = re.search(r"\boption\s*([A-Ea-e])\b", c, flags=re.I)
        if m: return m.group(1).lower()
        m = re.search(r"^\s*([A-Ea-e])[\)\.\:]\s*", c)
        if m: return m.group(1).lower()
        m = re.search(r"\b([A-Ea-e])\b", c)
        if len(str(c).strip()) <= 5 and m:
            return m.group(1).lower()
    s = raw.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())

def norm(x, task_type):
    return norm_choice(x) if task_type == "choice" else norm_num(x)

ap = argparse.ArgumentParser()
ap.add_argument("--trajectories", required=True)
ap.add_argument("--task_type", choices=["numeric", "choice"], required=True)
ap.add_argument("--out_json", required=True)
ap.add_argument("--out_jsonl", required=True)
args = ap.parse_args()

rows = read_jsonl(args.trajectories)
by = defaultdict(list)
for r in rows:
    by[r["sample_id"]].append(r)

details = []
first = majority = oracle = has_dis = all_dis = 0

for sid, rs in sorted(by.items()):
    rs = sorted(rs, key=lambda x: str(x.get("trajectory_id", "")))
    gold = rs[0].get("gold_answer", rs[0].get("answer", ""))
    gold_n = norm(gold, args.task_type)
    answers = [r.get("final_answer", "") for r in rs]
    ans_n = [norm(a, args.task_type) for a in answers]
    cnt = Counter(a for a in ans_n if a)
    maj = cnt.most_common(1)[0][0] if cnt else ""

    first_ok = int((ans_n[0] if ans_n else "") == gold_n)
    maj_ok = int(maj == gold_n)
    any_ok = int(any(a == gold_n for a in ans_n))

    first += first_ok
    majority += maj_ok
    oracle += any_ok

    uniq = set(a for a in ans_n if a)
    has_dis += int(len(uniq) >= 2)
    all_dis += int(len(uniq) >= 3)

    details.append({
        "sample_id": sid,
        "gold_answer": gold,
        "gold_norm": gold_n,
        "answers": answers,
        "answers_norm": ans_n,
        "majority_answer": maj,
        "majority_ok": maj_ok,
        "first_ok": first_ok,
        "oracle_any_ok": any_ok,
    })

n = len(by)
summary = {
    "n_samples": n,
    "n_trajectories": len(rows),
    "first_acc": first / max(1, n),
    "majority_acc": majority / max(1, n),
    "oracle_any_acc": oracle / max(1, n),
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

cat > experiments/apply_confirm_model_ablation.py <<'PY'
import argparse, json, re, string
from pathlib import Path
from collections import defaultdict, Counter

def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]

def norm_num(x):
    s = str(x or "").replace(",", "").replace("$", "").strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return " ".join(s.lower().split())
    y = nums[-1]
    return y.rstrip("0").rstrip(".") if "." in y else y

def norm_choice(x):
    raw = str(x or "")
    candidates = re.findall(r"final answer\s*[:：]\s*([^\n\|]+)", raw, flags=re.I) + [raw]
    for c in reversed(candidates):
        m = re.search(r"\(([A-Ea-e])\)", c)
        if m: return m.group(1).lower()
        m = re.search(r"\boption\s*([A-Ea-e])\b", c, flags=re.I)
        if m: return m.group(1).lower()
        m = re.search(r"^\s*([A-Ea-e])[\)\.\:]\s*", c)
        if m: return m.group(1).lower()
        m = re.search(r"\b([A-Ea-e])\b", c)
        if len(str(c).strip()) <= 5 and m:
            return m.group(1).lower()
    s = raw.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())

def norm(x, task_type):
    return norm_choice(x) if task_type == "choice" else norm_num(x)

ap = argparse.ArgumentParser()
ap.add_argument("--baseline_details", required=True)
ap.add_argument("--extra_jsonls", nargs="+", required=True)
ap.add_argument("--target_ids", required=True)
ap.add_argument("--task_type", choices=["numeric", "choice"], required=True)
ap.add_argument("--base_acc", type=float, required=True)
ap.add_argument("--n_samples", type=int, required=True)
ap.add_argument("--min_total_support", type=int, default=2)
ap.add_argument("--min_seed_support", type=int, default=2)
ap.add_argument("--min_margin", type=int, default=1)
ap.add_argument("--out_json", required=True)
args = ap.parse_args()

base = {r["sample_id"]: r for r in read_jsonl(args.baseline_details)}
target_ids = [x.strip() for x in open(args.target_ids, encoding="utf-8") if x.strip()]

extras = defaultdict(list)
for seed_idx, fp in enumerate(args.extra_jsonls):
    for r in read_jsonl(fp):
        sid = r["sample_id"]
        ans = norm(r.get("final_answer", ""), args.task_type)
        if ans:
            extras[sid].append((seed_idx, ans))

fixed = broken = changed = cur_correct = final_correct = 0

for sid in target_ids:
    b = base[sid]
    gold = b["gold_norm"]
    cur = b["majority_answer"]
    cur_ok = int(b["majority_ok"])

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

    fin_ok = int(final == gold)

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
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "========== Step 8: run experiments =========="

cfg_for_model() {
  case "$1" in
    deepseek7b) echo "configs/model/generator_deepseek_r1_distill_qwen7b_ablation.yaml" ;;
    qwen3b) echo "configs/model/generator_qwen25_3b_ablation.yaml" ;;
    llama3b) echo "configs/model/generator_llama32_3b_ablation.yaml" ;;
  esac
}

run_numeric_like() {
  local TAG="$1"
  local CFG="$2"
  local DS="$3"
  local INPUT="$4"
  local TASK_TYPE="$5"
  local MAX_NEW="${6:-384}"

  local N
  N=$(grep -cve '^[[:space:]]*$' "$INPUT" || true)

  echo "---------- $TAG $DS base ----------"
  python scripts/generate_numeric_trajectories_resume.py \
    --input "$INPUT" \
    --output data/processed/trajectories/model_ablation_parallel_qwen3b/${DS}_${TAG}_base_3traj.jsonl \
    --generator_config "$CFG" \
    --dataset "$DS" \
    --n_traj 3 \
    --max_samples 0 \
    --max_new_tokens "$MAX_NEW" \
    --temperature 0.7 \
    --top_p 0.95 \
    --seed 42 \
    2>&1 | tee outputs/logs/model_ablation_parallel_qwen3b/generate_${DS}_${TAG}_base.log

  python experiments/eval_base_model_ablation.py \
    --trajectories data/processed/trajectories/model_ablation_parallel_qwen3b/${DS}_${TAG}_base_3traj.jsonl \
    --task_type "$TASK_TYPE" \
    --out_json outputs/metrics/model_ablation_parallel_qwen3b/${DS}_${TAG}_base.json \
    --out_jsonl outputs/predictions/model_ablation_parallel_qwen3b/${DS}_${TAG}_base_details.jsonl

  python - <<PY
import json
from pathlib import Path
ds="$DS"; tag="$TAG"; inp="$INPUT"
rows=[json.loads(x) for x in open(f"outputs/predictions/model_ablation_parallel_qwen3b/{ds}_{tag}_base_details.jsonl", encoding="utf-8") if x.strip()]
ids=[]
for r in rows:
    vals=[str(a).strip() for a in r.get("answers_norm", []) if str(a).strip()]
    if len(set(vals)) >= 2:
        ids.append(r["sample_id"])
Path("outputs/targets/model_ablation").mkdir(parents=True, exist_ok=True)
with open(f"outputs/targets/model_ablation/{ds}_{tag}_has_disagreement_ids.txt", "w", encoding="utf-8") as f:
    for sid in ids:
        f.write(sid+"\\n")
unified=[json.loads(x) for x in open(inp, encoding="utf-8") if x.strip()]
idset=set(ids)
subset=[r for r in unified if (r.get("sample_id") or r.get("id")) in idset]
out=Path(f"data/processed/unified/model_ablation_parallel_qwen3b/{ds}_{tag}_has_disagreement.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for r in subset:
        f.write(json.dumps(r, ensure_ascii=False)+"\\n")
print(ds, tag, "targets", len(ids))
PY

  for SEED in 42 101 202
  do
    python scripts/generate_numeric_trajectories_resume.py \
      --input data/processed/unified/model_ablation_parallel_qwen3b/${DS}_${TAG}_has_disagreement.jsonl \
      --output data/processed/trajectories/model_ablation_parallel_qwen3b/${DS}_${TAG}_extra_seed${SEED}.jsonl \
      --generator_config "$CFG" \
      --dataset "$DS" \
      --n_traj 4 \
      --max_samples 0 \
      --max_new_tokens "$MAX_NEW" \
      --temperature 0.95 \
      --top_p 0.95 \
      --seed "$SEED" \
      2>&1 | tee outputs/logs/model_ablation_parallel_qwen3b/generate_${DS}_${TAG}_extra_seed${SEED}.log
  done

  BASE_ACC=$(python - <<PY
import json
x=json.load(open("outputs/metrics/model_ablation_parallel_qwen3b/${DS}_${TAG}_base.json", encoding="utf-8"))
print(x["majority_acc"])
PY
)

  EXTRA_FILES="data/processed/trajectories/model_ablation_parallel_qwen3b/${DS}_${TAG}_extra_seed42.jsonl data/processed/trajectories/model_ablation_parallel_qwen3b/${DS}_${TAG}_extra_seed101.jsonl data/processed/trajectories/model_ablation_parallel_qwen3b/${DS}_${TAG}_extra_seed202.jsonl"

  for TOTAL in 2 3 4
  do
    for SEEDSUP in 1 2 3
    do
      for MARGIN in 0 1 2
      do
        python experiments/apply_confirm_model_ablation.py \
          --baseline_details outputs/predictions/model_ablation_parallel_qwen3b/${DS}_${TAG}_base_details.jsonl \
          --extra_jsonls $EXTRA_FILES \
          --target_ids outputs/targets/model_ablation/${DS}_${TAG}_has_disagreement_ids.txt \
          --task_type "$TASK_TYPE" \
          --base_acc "$BASE_ACC" \
          --n_samples "$N" \
          --min_total_support "$TOTAL" \
          --min_seed_support "$SEEDSUP" \
          --min_margin "$MARGIN" \
          --out_json outputs/metrics/model_ablation_parallel_qwen3b/${DS}_${TAG}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.json
      done
    done
  done
}

run_bbh() {
  local TAG="$1"
  local CFG="$2"
  local TASK="$3"
  local INPUT="data/processed/unified/model_ablation_parallel_qwen3b/bbh_${TASK}_scope.jsonl"
  local DS="bbh_${TASK}"
  local N
  N=$(grep -cve '^[[:space:]]*$' "$INPUT" || true)

  echo "---------- $TAG $DS base ----------"
  python scripts/generate_bbh_logic_trajectories_vllm.py \
    --input "$INPUT" \
    --output data/processed/trajectories/model_ablation_parallel_qwen3b/${DS}_${TAG}_base_3traj.jsonl \
    --generator_config "$CFG" \
    --n_traj 3 \
    --max_samples 0 \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.95 \
    --seed 42 \
    --batch_size 4 \
    2>&1 | tee outputs/logs/model_ablation_parallel_qwen3b/generate_${DS}_${TAG}_base.log

  python experiments/eval_base_model_ablation.py \
    --trajectories data/processed/trajectories/model_ablation_parallel_qwen3b/${DS}_${TAG}_base_3traj.jsonl \
    --task_type choice \
    --out_json outputs/metrics/model_ablation_parallel_qwen3b/${DS}_${TAG}_base.json \
    --out_jsonl outputs/predictions/model_ablation_parallel_qwen3b/${DS}_${TAG}_base_details.jsonl

  python - <<PY
import json
from pathlib import Path
ds="$DS"; tag="$TAG"; inp="$INPUT"
rows=[json.loads(x) for x in open(f"outputs/predictions/model_ablation_parallel_qwen3b/{ds}_{tag}_base_details.jsonl", encoding="utf-8") if x.strip()]
ids=[]
for r in rows:
    vals=[str(a).strip() for a in r.get("answers_norm", []) if str(a).strip()]
    if len(set(vals)) >= 2:
        ids.append(r["sample_id"])
Path("outputs/targets/model_ablation").mkdir(parents=True, exist_ok=True)
with open(f"outputs/targets/model_ablation/{ds}_{tag}_has_disagreement_ids.txt", "w", encoding="utf-8") as f:
    for sid in ids:
        f.write(sid+"\\n")
unified=[json.loads(x) for x in open(inp, encoding="utf-8") if x.strip()]
idset=set(ids)
subset=[r for r in unified if (r.get("sample_id") or r.get("id")) in idset]
out=Path(f"data/processed/unified/model_ablation_parallel_qwen3b/{ds}_{tag}_has_disagreement.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for r in subset:
        f.write(json.dumps(r, ensure_ascii=False)+"\\n")
print(ds, tag, "targets", len(ids))
PY

  for SEED in 303 404 505 606 707 808
  do
    python scripts/generate_bbh_logic_trajectories_vllm.py \
      --input data/processed/unified/model_ablation_parallel_qwen3b/${DS}_${TAG}_has_disagreement.jsonl \
      --output data/processed/trajectories/model_ablation_parallel_qwen3b/${DS}_${TAG}_extra_seed${SEED}.jsonl \
      --generator_config "$CFG" \
      --n_traj 1 \
      --max_samples 0 \
      --max_new_tokens 512 \
      --temperature 0.9 \
      --top_p 0.95 \
      --seed "$SEED" \
      --batch_size 4 \
      2>&1 | tee outputs/logs/model_ablation_parallel_qwen3b/generate_${DS}_${TAG}_extra_seed${SEED}.log
  done

  BASE_ACC=$(python - <<PY
import json
x=json.load(open("outputs/metrics/model_ablation_parallel_qwen3b/${DS}_${TAG}_base.json", encoding="utf-8"))
print(x["majority_acc"])
PY
)

  EXTRA_FILES=""
  for SEED in 303 404 505 606 707 808
  do
    EXTRA_FILES="$EXTRA_FILES data/processed/trajectories/model_ablation_parallel_qwen3b/${DS}_${TAG}_extra_seed${SEED}.jsonl"
  done

  for TOTAL in 2 3 4
  do
    for SEEDSUP in 1 2 3
    do
      for MARGIN in 0 1 2
      do
        python experiments/apply_confirm_model_ablation.py \
          --baseline_details outputs/predictions/model_ablation_parallel_qwen3b/${DS}_${TAG}_base_details.jsonl \
          --extra_jsonls $EXTRA_FILES \
          --target_ids outputs/targets/model_ablation/${DS}_${TAG}_has_disagreement_ids.txt \
          --task_type choice \
          --base_acc "$BASE_ACC" \
          --n_samples "$N" \
          --min_total_support "$TOTAL" \
          --min_seed_support "$SEEDSUP" \
          --min_margin "$MARGIN" \
          --out_json outputs/metrics/model_ablation_parallel_qwen3b/${DS}_${TAG}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.json
      done
    done
  done
}

for TAG in $MODELS_TO_RUN
do
  CFG=$(cfg_for_model "$TAG")
  echo "==================== MODEL $TAG CFG=$CFG ===================="

  run_numeric_like "$TAG" "$CFG" gsm8k data/processed/unified/model_ablation_parallel_qwen3b/gsm8k_scope.jsonl numeric 384
  run_numeric_like "$TAG" "$CFG" svamp data/processed/unified/model_ablation_parallel_qwen3b/svamp_scope.jsonl numeric 384
  run_numeric_like "$TAG" "$CFG" asdiv data/processed/unified/model_ablation_parallel_qwen3b/asdiv_scope.jsonl numeric 384
  run_numeric_like "$TAG" "$CFG" math500 data/processed/unified/model_ablation_parallel_qwen3b/math500_scope.jsonl numeric 512
  run_numeric_like "$TAG" "$CFG" mathqa data/processed/unified/model_ablation_parallel_qwen3b/mathqa_scope.jsonl choice 512

  run_bbh "$TAG" "$CFG" logical_deduction_five_objects
  run_bbh "$TAG" "$CFG" formal_fallacies
done

echo "========== Step 9: summarize =========="

python - <<'PY' | tee outputs/logs/final_summaries/open_model_all_datasets_${MODE}_summary.md
import json
from pathlib import Path

mode = "${MODE}"

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

def best(rows):
    if not rows:
        return None
    return sorted(rows, key=lambda x: (-x.get("estimated_global_acc", -1), x.get("broken", 999), -x.get("fixed", -1), x.get("changed", 999)))[0]

print(f"# Open model all-dataset summary ({mode})\n")
print("| Model | Dataset | Base | Best | Gain | n_eval | fixed | broken | net | changed | setting |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

models = ["deepseek7b", "qwen3b", "llama3b"]
datasets = [
    "gsm8k",
    "svamp",
    "asdiv",
    "math500",
    "mathqa",
    "bbh_logical_deduction_five_objects",
    "bbh_formal_fallacies",
]

for tag in models:
    for ds in datasets:
        base_fp = Path(f"outputs/metrics/model_ablation_parallel_qwen3b/{ds}_{tag}_base.json")
        if not base_fp.exists():
            continue
        b = json.load(open(base_fp, encoding="utf-8"))
        rows = load_rows(f"outputs/metrics/model_ablation_parallel_qwen3b/{ds}_{tag}_total*.json")
        x = best(rows)
        if not x:
            continue
        setting = f"total{x['min_total_support']}_seed{x['min_seed_support']}_margin{x['min_margin']}"
        base = b["majority_acc"]
        final = x["estimated_global_acc"]
        print(f"| {tag} | {ds} | {base:.4f} | {final:.4f} | {final-base:+.4f} | {x['n_eval']} | {x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} | {setting} |")
PY

echo "========== DONE =========="
