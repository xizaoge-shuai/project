#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

mkdir -p outputs/logs/model_swap_chain /root/autodl-tmp/pce_backups

ORIG_CFG=configs/model/generator_llama_local_rewrite.yaml
DS_CFG=configs/model/generator_deepseek_r1_distill_qwen7b_original.yaml
Q3_CFG=configs/model/generator_qwen25_3b_original.yaml

log(){ echo "[$(date '+%F %T')] $*"; }

on_exit() {
  status=$?
  log "EXIT status=$status"
  tar --ignore-failed-read -czf /root/autodl-tmp/pce_backups/ds7b_qwen3b_fixed_route_$(date +%Y%m%d_%H%M%S).tar.gz \
    outputs/logs/model_swap_chain \
    outputs/metrics \
    outputs/predictions \
    outputs/final_selected_results \
    data/processed/trajectories/model_swap_base \
    data/processed/trajectories/gsm8k_deepseek \
    configs/model/generator_deepseek_r1_distill_qwen7b_original.yaml \
    configs/model/generator_qwen25_3b_original.yaml \
    2>/dev/null || true

  if [ "$status" = "0" ]; then
    log "ALL DONE, shutdown now"
    sync
    shutdown -h now || poweroff || halt || true
  else
    log "FAILED, no shutdown"
  fi
}
trap on_exit EXIT

write_cfgs() {
  log "write model configs"

  cp "$ORIG_CFG" "$DS_CFG"
  python - <<'PY'
from pathlib import Path
fp = Path("configs/model/generator_deepseek_r1_distill_qwen7b_original.yaml")
txt = fp.read_text()
txt = txt.replace("/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
                  "/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-7B")
fp.write_text(txt)
PY

  cp "$ORIG_CFG" "$Q3_CFG"
  python - <<'PY'
from pathlib import Path
fp = Path("configs/model/generator_qwen25_3b_original.yaml")
txt = fp.read_text()
txt = txt.replace("/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
                  "/root/autodl-tmp/models/Qwen2.5-3B-Instruct")
fp.write_text(txt)
PY
}

ensure_qwen3b() {
  Q3_MODEL=/root/autodl-tmp/models/Qwen2.5-3B-Instruct

  if [ -f "$Q3_MODEL/config.json" ] && find "$Q3_MODEL" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" \) | grep -q .; then
    log "Qwen3B already exists"
    du -sh "$Q3_MODEL" || true
    return 0
  fi

  log "Qwen3B incomplete, download by ModelScope"
  pip install -U modelscope

  rm -rf "$Q3_MODEL"
  mkdir -p "$Q3_MODEL"

  python - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    "Qwen/Qwen2.5-3B-Instruct",
    local_dir="/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
)
PY

  du -sh "$Q3_MODEL"
}

generate_numeric_if_missing() {
  TAG="$1"
  CFG="$2"
  DATASET="$3"
  INPUT="$4"
  OUTPUT="$5"
  NTRAJ="$6"
  MAX_NEW="$7"
  TEMP="$8"
  TOPP="$9"
  SEED="${10}"

  mkdir -p "$(dirname "$OUTPUT")"

  if [ -f "$OUTPUT" ]; then
    rows=$(grep -cve '^[[:space:]]*$' "$OUTPUT" || echo 0)
    log "[SKIP] exists $OUTPUT rows=$rows"
    return 0
  fi

  log "[GEN] $TAG $DATASET -> $OUTPUT"
  python scripts/generate_numeric_trajectories_local.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --generator_config "$CFG" \
    --dataset "$DATASET" \
    --n_traj "$NTRAJ" \
    --max_samples 0 \
    --max_new_tokens "$MAX_NEW" \
    --temperature "$TEMP" \
    --top_p "$TOPP" \
    --seed "$SEED" \
    2>&1 | tee "outputs/logs/model_swap_chain/gen_${TAG}_${DATASET}_$(basename "$OUTPUT").log"
}

generate_bbh_if_missing() {
  TAG="$1"
  CFG="$2"
  TASK="$3"
  INPUT="$4"
  OUTPUT="$5"
  NTRAJ="$6"
  SEED="$7"

  mkdir -p "$(dirname "$OUTPUT")"

  if [ -f "$OUTPUT" ]; then
    rows=$(grep -cve '^[[:space:]]*$' "$OUTPUT" || echo 0)
    log "[SKIP] exists $OUTPUT rows=$rows"
    return 0
  fi

  log "[GEN] $TAG BBH $TASK -> $OUTPUT"
  python scripts/generate_bbh_logic_trajectories_vllm.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --generator_config "$CFG" \
    --task "$TASK" \
    --n_traj "$NTRAJ" \
    --max_samples 0 \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.95 \
    --seed "$SEED" \
    2>&1 | tee "outputs/logs/model_swap_chain/gen_${TAG}_bbh_${TASK}.log"
}

prepare_base_for_model() {
  TAG="$1"
  CFG="$2"

  log "========== PREPARE BASE TRAJECTORIES FOR $TAG =========="

  BASE="data/processed/trajectories/model_swap_base/${TAG}"

  # GSM8K：DeepSeek 当前已经手动生成过，复制进统一 base 目录；Qwen3B 则重新生成
  if [ "$TAG" = "deepseek7b" ] && [ -f data/processed/trajectories/gsm8k_deepseek/test_local_3traj_full1319.jsonl ]; then
    mkdir -p "$BASE/gsm8k"
    cp data/processed/trajectories/gsm8k_deepseek/test_local_3traj_full1319.jsonl \
       "$BASE/gsm8k/test_local_3traj_full1319.jsonl"
    log "[COPY] existing DS GSM8K to $BASE/gsm8k/test_local_3traj_full1319.jsonl"
  else
    generate_numeric_if_missing "$TAG" "$CFG" gsm8k \
      data/processed/unified/gsm8k/test.jsonl \
      "$BASE/gsm8k/test_local_3traj_full1319.jsonl" \
      3 384 0.95 0.95 42
  fi

  # SVAMP full300，run_expanded_cross_resample 需要这个
  generate_numeric_if_missing "$TAG" "$CFG" svamp \
    data/processed/unified/svamp/test_local_3traj_full300.jsonl \
    "$BASE/svamp/test_local_3traj_full300.jsonl" \
    3 384 0.95 0.95 42

  # 如果上面的 input 不存在，尝试 common fallback
  if [ ! -f "$BASE/svamp/test_local_3traj_full300.jsonl" ]; then
    generate_numeric_if_missing "$TAG" "$CFG" svamp \
      data/processed/unified/svamp/test.jsonl \
      "$BASE/svamp/test_local_3traj_full300.jsonl" \
      3 384 0.95 0.95 42
  fi

  # ASDiv 500 / numeric full 由后续 asdiv extra 脚本生成 extra，这里只准备 cross 里可能需要的 base
  if [ -f data/processed/unified/asdiv/test_500.jsonl ]; then
    generate_numeric_if_missing "$TAG" "$CFG" asdiv \
      data/processed/unified/asdiv/test_500.jsonl \
      "$BASE/asdiv/test_local_3traj_500.jsonl" \
      3 384 0.95 0.95 42
  fi

  # BBH logical5 / formal 的 base，如果脚本后面需要
  if [ -f data/processed/unified/bbh_logic/logical_deduction_five_objects.jsonl ]; then
    generate_bbh_if_missing "$TAG" "$CFG" logical_deduction_five_objects \
      data/processed/unified/bbh_logic/logical_deduction_five_objects.jsonl \
      "$BASE/bbh_logic/logical_deduction_five_objects_3traj.jsonl" \
      3 42
  fi

  if [ -f data/processed/unified/bbh_logic/formal_fallacies.jsonl ]; then
    generate_bbh_if_missing "$TAG" "$CFG" formal_fallacies \
      data/processed/unified/bbh_logic/formal_fallacies.jsonl \
      "$BASE/bbh_logic/formal_fallacies_3traj.jsonl" \
      3 42
  fi
}

make_and_run_old_route() {
  TAG="$1"
  CFG="$2"

  log "========== RUN OLD QWEN7B ROUTE FOR $TAG =========="

  WORK="scripts/fixed_route_${TAG}"
  rm -rf "$WORK"
  mkdir -p "$WORK"

  OLD_ROUTE=(
    experiments/run_expanded_cross_resample_autoshutdown.sh
    experiments/run_asdiv_numeric_extra_confirm_v2.sh
    experiments/run_mathqa_scale_extra_confirm.sh
    experiments/run_reasoning_qa_extra_confirm_autoshutdown.sh
    experiments/run_true_generation_ablation_qwen_autoshutdown.sh
  )

  for SRC in "${OLD_ROUTE[@]}"; do
    BASE_NAME=$(basename "$SRC" .sh)
    DST="$WORK/${BASE_NAME}_${TAG}.sh"
    cp "$SRC" "$DST"

    sed -i "s#configs/model/generator_llama_local_rewrite.yaml#${CFG}#g" "$DST"

    # 只精确替换这些旧脚本会读的 base trajectories
    sed -i "s#data/processed/trajectories/gsm8k/test_local_3traj_full1319.jsonl#data/processed/trajectories/model_swap_base/${TAG}/gsm8k/test_local_3traj_full1319.jsonl#g" "$DST"
    sed -i "s#data/processed/trajectories/svamp/test_local_3traj_full300.jsonl#data/processed/trajectories/model_swap_base/${TAG}/svamp/test_local_3traj_full300.jsonl#g" "$DST"
    sed -i "s#data/processed/trajectories/asdiv/test_local_3traj_500.jsonl#data/processed/trajectories/model_swap_base/${TAG}/asdiv/test_local_3traj_500.jsonl#g" "$DST"
    sed -i "s#data/processed/trajectories/bbh_logic/logical_deduction_five_objects_3traj.jsonl#data/processed/trajectories/model_swap_base/${TAG}/bbh_logic/logical_deduction_five_objects_3traj.jsonl#g" "$DST"
    sed -i "s#data/processed/trajectories/bbh_logic/formal_fallacies_3traj.jsonl#data/processed/trajectories/model_swap_base/${TAG}/bbh_logic/formal_fallacies_3traj.jsonl#g" "$DST"

    sed -i 's#shutdown -h now#echo "[skip inner shutdown]"#g' "$DST"
    sed -i 's#poweroff#echo "[skip inner poweroff]"#g' "$DST"
    sed -i 's#halt#echo "[skip inner halt]"#g' "$DST"

    chmod +x "$DST"

    log "[RUN] $TAG :: $BASE_NAME"
    bash "$DST" 2>&1 | tee "outputs/logs/model_swap_chain/${BASE_NAME}_${TAG}.log"
    log "[DONE] $TAG :: $BASE_NAME"
  done
}

write_cfgs

prepare_base_for_model deepseek7b "$DS_CFG"
make_and_run_old_route deepseek7b "$DS_CFG"

ensure_qwen3b
prepare_base_for_model qwen3b "$Q3_CFG"
make_and_run_old_route qwen3b "$Q3_CFG"

log "========== ALL DONE =========="
