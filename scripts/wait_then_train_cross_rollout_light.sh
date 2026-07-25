#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

LOG="outputs/logs/cross_table/wait_then_rollout_light.log"

while true
do
  DONE=$(
    find \
      data/processed/labels_cross_table_rollout_p09/meta/atom_level \
      data/processed/labels_cross_table_rollout_p08/meta/atom_level \
      -type f -name "*.json" 2>/dev/null | wc -l
  )

  echo "[$(date '+%F %T')] completed=$DONE/24" | tee -a "$LOG"

  if [ "$DONE" -ge 24 ]; then
    break
  fi

  sleep 300
done

echo "[$(date '+%F %T')] rollout labels complete; start Light PCE" \
  | tee -a "$LOG"

bash scripts/run_cross_rollout_light_table1.sh \
  >> outputs/logs/cross_table/rollout_light_table1.log 2>&1

echo "[$(date '+%F %T')] Light PCE complete" | tee -a "$LOG"
