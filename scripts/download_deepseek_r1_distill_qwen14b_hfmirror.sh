#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

# 不走本地 SSH 反向隧道/代理，直接让服务器访问镜像站
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export HF_HUB_DISABLE_TELEMETRY=1

# 镜像站场景下，先禁用 Xet，走普通 HTTP 下载更稳
export HF_HUB_DISABLE_XET=1

MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
MODEL_DIR=/root/autodl-tmp/models/DeepSeek-R1-Distill-Qwen-14B

mkdir -p "$MODEL_DIR"

echo "========== START DOWNLOAD =========="
date
echo "MODEL_ID=$MODEL_ID"
echo "MODEL_DIR=$MODEL_DIR"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
echo
df -h /root/autodl-tmp || true
echo

# 自动重试；中断后重新执行也会跳过已完成文件/继续未完成文件
while true
do
  echo
  echo "========== hf download try =========="
  date

  hf download "$MODEL_ID" \
    --local-dir "$MODEL_DIR"

  status=$?
  echo "hf download exit status=$status"
  date

  if [ "$status" -eq 0 ]; then
    echo "========== DOWNLOAD DONE =========="
    date
    du -sh "$MODEL_DIR" || true
    ls -lh "$MODEL_DIR" | head -50 || true
    exit 0
  fi

  echo "[WARN] download failed, sleep 60s then retry..."
  sleep 60
done
