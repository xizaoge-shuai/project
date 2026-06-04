#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}

unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

mkdir -p outputs/logs/model_ablation_sweeps

TOTALS=(2 3 4)
SEEDSUPS=(1 2 3)
MARGINS=(0 1 2 3)

count_lines() {
  local f="$1"
  if [ -f "$f" ]; then
    wc -l < "$f"
  else
    echo 0
  fi
}

ensure_target_ids() {
  local target_jsonl="$1"
  local target_ids="$2"

  if [ -f "$target_ids" ]; then
    echo "[OK] target ids exists: $target_ids ($(wc -l < "$target_ids"))"
    return 0
  fi

  if [ ! -f "$target_jsonl" ]; then
    echo "[WARN] missing target_jsonl and target_ids:"
    echo "       target_jsonl=$target_jsonl"
    echo "       target_ids=$target_ids"
    return 1
  fi

  mkdir -p "$(dirname "$target_ids")"

  python - "$target_jsonl" "$target_ids" <<'PY'
import json, sys
from pathlib import Path

inp = Path(sys.argv[1])
out = Path(sys.argv[2])

ids = []
for line in inp.open(encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    sid = r.get("sample_id") or r.get("id")
    if sid is not None:
        ids.append(str(sid))

out.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
print(f"[SAVE] {out} n_ids={len(ids)}")
PY
}

run_confirm_sweep() {
  local tag="$1"
  local dataset="$2"
  local model="$3"
  local task_type="$4"
  local n_samples="$5"
  local base_acc="$6"
  local traj_dir="$7"
  local pred_dir="$8"
  local target_dir="$9"
  local target_jsonl="${10}"
  local out_dir="${11}"

  local base_details="${pred_dir}/${dataset}_${model}_base_details.jsonl"
  local target_ids="${target_dir}/${dataset}_${model}_has_disagreement_ids.txt"

  echo
  echo "================================================================================"
  echo "[SWEEP] tag=${tag} dataset=${dataset} model=${model}"
  echo "================================================================================"

  if [ ! -f "$base_details" ]; then
    echo "[SKIP] missing base_details: $base_details"
    return 0
  fi

  ensure_target_ids "$target_jsonl" "$target_ids" || {
    echo "[SKIP] cannot build target ids for ${dataset}_${model}"
    return 0
  }

  mapfile -t extras < <(ls "${traj_dir}/${dataset}_${model}_extra_seed"*.jsonl 2>/dev/null | sort || true)

  if [ "${#extras[@]}" -eq 0 ]; then
    echo "[SKIP] no extra jsonls for ${dataset}_${model} in ${traj_dir}"
    return 0
  fi

  echo "[INFO] base_details=$base_details"
  echo "[INFO] target_ids=$target_ids ($(wc -l < "$target_ids"))"
  echo "[INFO] extras:"
  for f in "${extras[@]}"; do
    echo "  $(wc -l < "$f")  $f"
  done

  mkdir -p "$out_dir"

  local max_seed_sup="${#extras[@]}"
  if [ "$max_seed_sup" -gt 3 ]; then
    max_seed_sup=3
  fi

  for TOTAL in "${TOTALS[@]}"; do
    for SEEDSUP in "${SEEDSUPS[@]}"; do
      if [ "$SEEDSUP" -gt "$max_seed_sup" ]; then
        continue
      fi
      for MARGIN in "${MARGINS[@]}"; do
        local out_json="${out_dir}/${dataset}_${model}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.json"

        echo "[RUN] ${out_json}"

        python experiments/apply_confirm_model_ablation.py \
          --baseline_details "$base_details" \
          --extra_jsonls "${extras[@]}" \
          --target_ids "$target_ids" \
          --task_type "$task_type" \
          --base_acc "$base_acc" \
          --n_samples "$n_samples" \
          --min_total_support "$TOTAL" \
          --min_seed_support "$SEEDSUP" \
          --min_margin "$MARGIN" \
          --out_json "$out_json"
      done
    done
  done
}

run_mathqa_optionmap() {
  local tag="$1"
  local prefix="$2"
  local scope="$3"
  local base="$4"
  local target_jsonl="$5"
  local target_ids="$6"
  local out_dir="$7"
  shift 7
  local extras=("$@")

  echo
  echo "================================================================================"
  echo "[MATHQA OPTIONMAP] ${tag}"
  echo "================================================================================"

  if [ ! -f "$scope" ]; then
    echo "[WARN] missing scope=$scope"
    if [ -f "data/processed/unified/model_ablation/mathqa_scope.jsonl" ]; then
      scope="data/processed/unified/model_ablation/mathqa_scope.jsonl"
      echo "[FALLBACK] scope=$scope"
    else
      echo "[SKIP] no available scope"
      return 0
    fi
  fi

  if [ ! -f "$base" ]; then
    echo "[SKIP] missing base=$base"
    return 0
  fi

  ensure_target_ids "$target_jsonl" "$target_ids" || {
    echo "[SKIP] cannot build target ids for mathqa optionmap"
    return 0
  }

  local ok_extras=()
  for f in "${extras[@]}"; do
    if [ -f "$f" ]; then
      ok_extras+=("$f")
    fi
  done

  if [ "${#ok_extras[@]}" -eq 0 ]; then
    echo "[SKIP] no mathqa extras"
    return 0
  fi

  mkdir -p "$out_dir"

  echo "[INFO] scope=$scope"
  echo "[INFO] base=$base ($(wc -l < "$base"))"
  echo "[INFO] target_ids=$target_ids ($(wc -l < "$target_ids"))"
  echo "[INFO] extras:"
  for f in "${ok_extras[@]}"; do
    echo "  $(wc -l < "$f")  $f"
  done

  python scripts/reeval_mathqa_option_mapping_confirm.py \
    --scope "$scope" \
    --base "$base" \
    --extras "${ok_extras[@]}" \
    --targets "$target_ids" \
    --target_jsonl "$target_jsonl" \
    --out_dir "$out_dir" \
    --prefix "$prefix"
}

check_long1024_complete() {
  local target_jsonl="$1"
  shift
  local extras=("$@")

  if [ ! -f "$target_jsonl" ]; then
    echo "[CHECK] missing target_jsonl=$target_jsonl"
    return 1
  fi

  local n_targets
  n_targets=$(wc -l < "$target_jsonl")
  local expected=$((n_targets * 4))

  if [ "${#extras[@]}" -eq 0 ]; then
    echo "[CHECK] no long1024 extras"
    return 1
  fi

  local all_ok=1
  for f in "${extras[@]}"; do
    if [ ! -f "$f" ]; then
      echo "[CHECK] missing $f"
      all_ok=0
      continue
    fi
    local rows
    rows=$(wc -l < "$f")
    echo "[CHECK] $rows / $expected  $f"
    if [ "$rows" -ne "$expected" ]; then
      all_ok=0
    fi
  done

  if [ "$all_ok" -eq 1 ]; then
    return 0
  else
    return 1
  fi
}

run_long1024_sweep() {
  local tag="$1"
  local dataset="$2"
  local model="$3"
  local task_type="$4"
  local n_samples="$5"
  local base_acc="$6"
  local base_details="$7"
  local target_jsonl="$8"
  local target_ids="$9"
  local out_dir="${10}"
  shift 10
  local extras=("$@")

  echo
  echo "================================================================================"
  echo "[LONG1024 SWEEP] tag=${tag} dataset=${dataset} model=${model}"
  echo "================================================================================"

  if [ ! -f "$base_details" ]; then
    echo "[SKIP] missing base_details=$base_details"
    return 0
  fi

  ensure_target_ids "$target_jsonl" "$target_ids" || {
    echo "[SKIP] cannot build target ids"
    return 0
  }

  if ! check_long1024_complete "$target_jsonl" "${extras[@]}"; then
    echo "[SKIP] long1024 extras are not complete; skip eval now"
    return 0
  fi

  mkdir -p "$out_dir"

  for TOTAL in "${TOTALS[@]}"; do
    for SEEDSUP in "${SEEDSUPS[@]}"; do
      for MARGIN in "${MARGINS[@]}"; do
        local out_json="${out_dir}/${dataset}_${model}_long1024_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}.json"

        echo "[RUN] $out_json"

        python experiments/apply_confirm_model_ablation.py \
          --baseline_details "$base_details" \
          --extra_jsonls "${extras[@]}" \
          --target_ids "$target_ids" \
          --task_type "$task_type" \
          --base_acc "$base_acc" \
          --n_samples "$n_samples" \
          --min_total_support "$TOTAL" \
          --min_seed_support "$SEEDSUP" \
          --min_margin "$MARGIN" \
          --out_json "$out_json"
      done
    done
  done
}

summarize_json_dir() {
  local root="$1"
  local out_md="$2"

  mkdir -p "$(dirname "$out_md")"

  python - "$root" "$out_md" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])

rows = []
for fp in sorted(root.glob("*.json")):
    try:
        d = json.load(open(fp, encoding="utf-8"))
    except Exception:
        continue

    final = d.get("estimated_global_acc", d.get("final_acc", d.get("acc", d.get("accuracy"))))
    if final is None:
        continue

    base = d.get("base_acc", d.get("majority_acc"))
    gain = d.get("gain")
    if gain is None and base is not None:
        try:
            gain = float(final) - float(base)
        except Exception:
            gain = None

    fixed = d.get("fixed")
    broken = d.get("broken")
    net = d.get("net")
    if net is None and fixed is not None and broken is not None:
        net = fixed - broken

    rows.append((float(final), fp, d, base, gain, fixed, broken, net))

rows.sort(key=lambda t: (-t[0], int(t[2].get("broken", 999999)), -int(t[2].get("fixed", -1))))

def fmt(x):
    if x is None:
        return "NA"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

lines = []
lines.append(f"# Sweep summary: `{root}`")
lines.append("")
lines.append("| file | base | final | gain | n_eval | changed | fixed | broken | net |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

for final, fp, d, base, gain, fixed, broken, net in rows[:50]:
    lines.append(
        f"| `{fp.name}` | {fmt(base)} | {fmt(final)} | {fmt(gain)} | "
        f"{d.get('n_eval', d.get('n_samples', 'NA'))} | {d.get('changed', 'NA')} | "
        f"{fixed if fixed is not None else 'NA'} | {broken if broken is not None else 'NA'} | "
        f"{net if net is not None else 'NA'} |"
    )

out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[SAVE]", out)
PY
}

echo
echo "################################################################################"
echo "# 1. DS7B ordinary sweeps"
echo "################################################################################"

run_confirm_sweep ds7b asdiv deepseek7b numeric 2249 0.7514 \
  data/processed/trajectories/model_ablation \
  outputs/predictions/model_ablation \
  outputs/targets/model_ablation \
  data/processed/unified/model_ablation/asdiv_deepseek7b_has_disagreement.jsonl \
  outputs/metrics/model_ablation_ablation_sweep

run_confirm_sweep ds7b gsm8k deepseek7b numeric 1319 0.4936 \
  data/processed/trajectories/model_ablation \
  outputs/predictions/model_ablation \
  outputs/targets/model_ablation \
  data/processed/unified/model_ablation/gsm8k_deepseek7b_has_disagreement.jsonl \
  outputs/metrics/model_ablation_ablation_sweep

run_confirm_sweep ds7b svamp deepseek7b numeric 300 0.6933 \
  data/processed/trajectories/model_ablation \
  outputs/predictions/model_ablation \
  outputs/targets/model_ablation \
  data/processed/unified/model_ablation/svamp_deepseek7b_has_disagreement.jsonl \
  outputs/metrics/model_ablation_ablation_sweep

run_confirm_sweep ds7b math500 deepseek7b numeric 500 0.3100 \
  data/processed/trajectories/model_ablation \
  outputs/predictions/model_ablation \
  outputs/targets/model_ablation \
  data/processed/unified/model_ablation/math500_deepseek7b_has_disagreement.jsonl \
  outputs/metrics/model_ablation_ablation_sweep

run_confirm_sweep ds7b bbh_formal_fallacies deepseek7b choice 100 0.1500 \
  data/processed/trajectories/model_ablation \
  outputs/predictions/model_ablation \
  outputs/targets/model_ablation \
  data/processed/unified/model_ablation/bbh_formal_fallacies_deepseek7b_has_disagreement.jsonl \
  outputs/metrics/model_ablation_ablation_sweep

run_confirm_sweep ds7b bbh_logical_deduction_five_objects deepseek7b choice 100 0.1000 \
  data/processed/trajectories/model_ablation \
  outputs/predictions/model_ablation \
  outputs/targets/model_ablation \
  data/processed/unified/model_ablation/bbh_logical_deduction_five_objects_deepseek7b_has_disagreement.jsonl \
  outputs/metrics/model_ablation_ablation_sweep

echo
echo "################################################################################"
echo "# 2. Qwen3B ordinary sweeps"
echo "################################################################################"

run_confirm_sweep qwen3b asdiv qwen3b numeric 2249 0.7835 \
  data/processed/trajectories/model_ablation_parallel_qwen3b \
  outputs/predictions/model_ablation_parallel_qwen3b \
  outputs/targets/model_ablation_parallel_qwen3b \
  data/processed/unified/model_ablation_parallel_qwen3b/asdiv_qwen3b_has_disagreement.jsonl \
  outputs/metrics/model_ablation_parallel_qwen3b_ablation_sweep

run_confirm_sweep qwen3b gsm8k qwen3b numeric 1319 0.5004 \
  data/processed/trajectories/model_ablation_parallel_qwen3b \
  outputs/predictions/model_ablation_parallel_qwen3b \
  outputs/targets/model_ablation_parallel_qwen3b \
  data/processed/unified/model_ablation_parallel_qwen3b/gsm8k_qwen3b_has_disagreement.jsonl \
  outputs/metrics/model_ablation_parallel_qwen3b_ablation_sweep

run_confirm_sweep qwen3b svamp qwen3b numeric 300 0.8000 \
  data/processed/trajectories/model_ablation_parallel_qwen3b \
  outputs/predictions/model_ablation_parallel_qwen3b \
  outputs/targets/model_ablation_parallel_qwen3b \
  data/processed/unified/model_ablation_parallel_qwen3b/svamp_qwen3b_has_disagreement.jsonl \
  outputs/metrics/model_ablation_parallel_qwen3b_ablation_sweep

run_confirm_sweep qwen3b math500 qwen3b numeric 500 0.2980 \
  data/processed/trajectories/model_ablation_parallel_qwen3b \
  outputs/predictions/model_ablation_parallel_qwen3b \
  outputs/targets/model_ablation_parallel_qwen3b \
  data/processed/unified/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement.jsonl \
  outputs/metrics/model_ablation_parallel_qwen3b_ablation_sweep

run_confirm_sweep qwen3b bbh_formal_fallacies qwen3b choice 100 0.4500 \
  data/processed/trajectories/model_ablation_parallel_qwen3b \
  outputs/predictions/model_ablation_parallel_qwen3b \
  outputs/targets/model_ablation_parallel_qwen3b \
  data/processed/unified/model_ablation_parallel_qwen3b/bbh_formal_fallacies_qwen3b_has_disagreement.jsonl \
  outputs/metrics/model_ablation_parallel_qwen3b_ablation_sweep

run_confirm_sweep qwen3b bbh_logical_deduction_five_objects qwen3b choice 100 0.3400 \
  data/processed/trajectories/model_ablation_parallel_qwen3b \
  outputs/predictions/model_ablation_parallel_qwen3b \
  outputs/targets/model_ablation_parallel_qwen3b \
  data/processed/unified/model_ablation_parallel_qwen3b/bbh_logical_deduction_five_objects_qwen3b_has_disagreement.jsonl \
  outputs/metrics/model_ablation_parallel_qwen3b_ablation_sweep

echo
echo "################################################################################"
echo "# 3. MathQA optionmap sweeps"
echo "################################################################################"

run_mathqa_optionmap ds7b mathqa_deepseek7b_optionmap_ablation \
  data/processed/unified/model_ablation/mathqa_scope.jsonl \
  data/processed/trajectories/model_ablation/mathqa_deepseek7b_base_3traj.jsonl \
  data/processed/unified/model_ablation/mathqa_deepseek7b_has_disagreement.jsonl \
  outputs/targets/model_ablation/mathqa_deepseek7b_has_disagreement_ids.txt \
  outputs/metrics/model_ablation_mathqa_optionmap_ablation \
  data/processed/trajectories/model_ablation/mathqa_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/mathqa_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/mathqa_deepseek7b_extra_seed202.jsonl

run_mathqa_optionmap qwen3b mathqa_qwen3b_optionmap_ablation \
  data/processed/unified/model_ablation_parallel_qwen3b/mathqa_scope.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/mathqa_qwen3b_base_3traj.jsonl \
  data/processed/unified/model_ablation_parallel_qwen3b/mathqa_qwen3b_has_disagreement.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/mathqa_qwen3b_has_disagreement_ids.txt \
  outputs/metrics/model_ablation_mathqa_optionmap_qwen3b_ablation \
  data/processed/trajectories/model_ablation_parallel_qwen3b/mathqa_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/mathqa_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/mathqa_qwen3b_extra_seed202.jsonl

echo
echo "################################################################################"
echo "# 4. MATH500 long1024 sweeps, eval only if files complete"
echo "################################################################################"

run_long1024_sweep ds7b math500 deepseek7b numeric 500 0.3100 \
  outputs/predictions/model_ablation/math500_deepseek7b_base_details.jsonl \
  data/processed/unified/model_ablation/math500_deepseek7b_has_disagreement.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  outputs/metrics/model_ablation_boost_ablation_sweep \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed303.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed404.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed505.jsonl

run_long1024_sweep qwen3b math500 qwen3b numeric 500 0.2980 \
  outputs/predictions/model_ablation_parallel_qwen3b/math500_qwen3b_base_details.jsonl \
  data/processed/unified/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  outputs/metrics/model_ablation_boost_qwen3b_long1024_ablation_sweep \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed303.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed404.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed505.jsonl

echo
echo "################################################################################"
echo "# 5. Summarize sweep dirs"
echo "################################################################################"

summarize_json_dir outputs/metrics/model_ablation_ablation_sweep \
  outputs/metrics/model_ablation_ablation_sweep/summary_top50.md

summarize_json_dir outputs/metrics/model_ablation_parallel_qwen3b_ablation_sweep \
  outputs/metrics/model_ablation_parallel_qwen3b_ablation_sweep/summary_top50.md

summarize_json_dir outputs/metrics/model_ablation_mathqa_optionmap_ablation \
  outputs/metrics/model_ablation_mathqa_optionmap_ablation/summary_top50.md

summarize_json_dir outputs/metrics/model_ablation_mathqa_optionmap_qwen3b_ablation \
  outputs/metrics/model_ablation_mathqa_optionmap_qwen3b_ablation/summary_top50.md

summarize_json_dir outputs/metrics/model_ablation_boost_ablation_sweep \
  outputs/metrics/model_ablation_boost_ablation_sweep/summary_top50.md

summarize_json_dir outputs/metrics/model_ablation_boost_qwen3b_long1024_ablation_sweep \
  outputs/metrics/model_ablation_boost_qwen3b_long1024_ablation_sweep/summary_top50.md

echo
echo "################################################################################"
echo "# 6. Rebuild final report tables if scripts exist"
echo "################################################################################"

if [ -f scripts/build_ds7b_report_tables.py ]; then
  python scripts/build_ds7b_report_tables.py
fi

if [ -f scripts/build_qwen3b_report_tables.py ]; then
  python scripts/build_qwen3b_report_tables.py
fi

echo
echo "################################################################################"
echo "# DONE"
echo "################################################################################"

echo "[SUMMARY FILES]"
find outputs/metrics -path "*ablation*summary_top50.md" -type f -print

