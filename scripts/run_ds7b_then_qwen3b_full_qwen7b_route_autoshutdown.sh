#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/pce_reasoning_project/project

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate pce
else
  eval "$(conda shell.bash hook)"
  conda activate pce
fi

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

MASTER_LOG_DIR=outputs/logs/model_swap_chain
mkdir -p "$MASTER_LOG_DIR" /root/autodl-tmp/pce_backups

ORIG_CFG=configs/model/generator_llama_local_rewrite.yaml
DS_CFG=configs/model/generator_deepseek_r1_distill_qwen7b_original.yaml
Q3_CFG=configs/model/generator_qwen25_3b_original.yaml

DS_MODEL=/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-7B
Q3_MODEL=/root/autodl-tmp/models/Qwen2.5-3B-Instruct

# 这几条就是你前面 grep 出来的 Qwen7B 老路线脚本。
# 不再使用 run_autodl_open_models_all_datasets_autoshutdown.sh 那种 smoke wrapper。
OLD_ROUTE_SCRIPTS=(
  experiments/run_expanded_cross_resample_autoshutdown.sh
  experiments/run_asdiv_numeric_extra_confirm_v2.sh
  experiments/run_mathqa_scale_extra_confirm.sh
  experiments/run_reasoning_qa_extra_confirm_autoshutdown.sh
  experiments/run_true_generation_ablation_qwen_autoshutdown.sh
)

log() {
  echo "[$(date '+%F %T')] $*"
}

on_exit() {
  status=$?
  log "EXIT status=$status"

  tar --ignore-failed-read -czf /root/autodl-tmp/pce_backups/ds7b_qwen3b_full_qwen7b_route_$(date +%Y%m%d_%H%M%S).tar.gz \
    outputs/logs/model_swap_chain \
    outputs/logs/model_swap_fullroute \
    outputs/metrics/model_swap_fullroute \
    outputs/predictions/model_swap_fullroute \
    outputs/final_selected_results/model_swap_fullroute \
    outputs/targets/model_swap_fullroute \
    data/processed/trajectories/model_swap_fullroute \
    data/processed/trajectories/gsm8k_deepseek \
    configs/model/generator_deepseek_r1_distill_qwen7b_original.yaml \
    configs/model/generator_qwen25_3b_original.yaml \
    2>/dev/null || true

  if [ "$status" = "0" ]; then
    log "ALL DONE. shutdown now."
    sync
    shutdown -h now || poweroff || halt || true
  else
    log "FAILED. no shutdown."
  fi
}
trap on_exit EXIT

write_cfgs() {
  log "write DS7B config from Qwen7B config"
  cp "$ORIG_CFG" "$DS_CFG"
  python - <<'PY'
from pathlib import Path
fp = Path("configs/model/generator_deepseek_r1_distill_qwen7b_original.yaml")
txt = fp.read_text(encoding="utf-8")
txt = txt.replace(
    "/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
    "/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-7B"
)
fp.write_text(txt, encoding="utf-8")
PY

  log "write Qwen3B config from Qwen7B config"
  cp "$ORIG_CFG" "$Q3_CFG"
  python - <<'PY'
from pathlib import Path
fp = Path("configs/model/generator_qwen25_3b_original.yaml")
txt = fp.read_text(encoding="utf-8")
txt = txt.replace(
    "/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
    "/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
)
fp.write_text(txt, encoding="utf-8")
PY

  log "DS config diff:"
  diff -u "$ORIG_CFG" "$DS_CFG" || true
  log "Qwen3B config diff:"
  diff -u "$ORIG_CFG" "$Q3_CFG" || true
}

wait_current_ds_gsm8k() {
  log "wait current DeepSeek GSM8K if still running"

  PID="${CURRENT_GSM_PID:-}"
  if [ -z "$PID" ]; then
    PID=$(pgrep -f "generate_numeric_trajectories_local.py.*gsm8k_deepseek.*generator_deepseek" | head -1 || true)
  fi

  if [ -n "$PID" ]; then
    log "found current DS-GSM8K PID=$PID"
    while kill -0 "$PID" 2>/dev/null; do
      log "current DS-GSM8K still running..."
      sleep 120
    done
    log "current DS-GSM8K finished."
  else
    log "no current DS-GSM8K process found."
  fi
}

ensure_qwen3b() {
  log "check Qwen3B model"

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
model_dir = snapshot_download(
    "Qwen/Qwen2.5-3B-Instruct",
    local_dir="/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
)
print("downloaded to:", model_dir)
PY

  du -sh "$Q3_MODEL"
  find "$Q3_MODEL" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.bin" \) -printf "%f %s\n" | sort
}

make_route_scripts_for_model() {
  TAG="$1"
  CFG="$2"

  OUTDIR="scripts/model_swap_fullroute_${TAG}"
  rm -rf "$OUTDIR"
  mkdir -p "$OUTDIR"

  mkdir -p \
    "outputs/logs/model_swap_fullroute/${TAG}" \
    "outputs/metrics/model_swap_fullroute/${TAG}" \
    "outputs/predictions/model_swap_fullroute/${TAG}" \
    "outputs/final_selected_results/model_swap_fullroute/${TAG}" \
    "outputs/targets/model_swap_fullroute/${TAG}" \
    "data/processed/trajectories/model_swap_fullroute/${TAG}"

  for SRC in "${OLD_ROUTE_SCRIPTS[@]}"; do
    BASE=$(basename "$SRC" .sh)
    DST="$OUTDIR/${BASE}_${TAG}.sh"

    cp "$SRC" "$DST"

    # 只换模型配置；采样参数、数据集、原脚本逻辑不动
    sed -i "s#configs/model/generator_llama_local_rewrite.yaml#${CFG}#g" "$DST"
    sed -i "s#/root/autodl-tmp/models/Qwen2.5-7B-Instruct#$(grep -m1 '^model_name_or_path:' "$CFG" | awk '{print $2}')#g" "$DST"

    # 防止子脚本中途关机，最后由本总脚本统一关机
    sed -i 's#shutdown -h now#echo "[skip inner shutdown]"#g' "$DST"
    sed -i 's#poweroff#echo "[skip inner poweroff]"#g' "$DST"
    sed -i 's#halt#echo "[skip inner halt]"#g' "$DST"

    # 隔离不同模型输出，避免覆盖原 Qwen7B 结果
    sed -i "s#outputs/logs/#outputs/logs/model_swap_fullroute/${TAG}/#g" "$DST"
    sed -i "s#outputs/metrics/#outputs/metrics/model_swap_fullroute/${TAG}/#g" "$DST"
    sed -i "s#outputs/predictions/#outputs/predictions/model_swap_fullroute/${TAG}/#g" "$DST"
    sed -i "s#outputs/final_selected_results/#outputs/final_selected_results/model_swap_fullroute/${TAG}/#g" "$DST"
    sed -i "s#outputs/targets/#outputs/targets/model_swap_fullroute/${TAG}/#g" "$DST"

    # 只替换 trajectory 的输出/引用目录，让不同模型结果隔离
    sed -i "s#data/processed/trajectories/#data/processed/trajectories/model_swap_fullroute/${TAG}/#g" "$DST"

    chmod +x "$DST"
  done

  log "created model route scripts for $TAG:"
  find "$OUTDIR" -type f -name "*.sh" | sort
}

run_full_qwen7b_route_for_model() {
  TAG="$1"
  CFG="$2"

  log "========== START FULL QWEN7B ROUTE FOR $TAG =========="
  make_route_scripts_for_model "$TAG" "$CFG"

  OUTDIR="scripts/model_swap_fullroute_${TAG}"

  for SRC in "${OLD_ROUTE_SCRIPTS[@]}"; do
    BASE=$(basename "$SRC" .sh)
    SCRIPT="$OUTDIR/${BASE}_${TAG}.sh"
    LOG="outputs/logs/model_swap_fullroute/${TAG}/${BASE}_${TAG}.log"

    log "run $TAG :: $BASE"
    bash "$SCRIPT" 2>&1 | tee "$LOG"
    log "done $TAG :: $BASE"
  done

  log "========== DONE FULL QWEN7B ROUTE FOR $TAG =========="
}

log "========== DS7B -> Qwen3B FULL QWEN7B ROUTE START =========="
write_cfgs
wait_current_ds_gsm8k

run_full_qwen7b_route_for_model "deepseek7b" "$DS_CFG"

ensure_qwen3b
run_full_qwen7b_route_for_model "qwen3b" "$Q3_CFG"

log "========== ALL MODEL ROUTES DONE =========="
