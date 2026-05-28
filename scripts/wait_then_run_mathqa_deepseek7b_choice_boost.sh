#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

mkdir -p data/processed/trajectories/model_ablation_boost
mkdir -p outputs/logs/model_ablation_boost
mkdir -p outputs/metrics/model_ablation_mathqa_choiceboost
mkdir -p outputs/metrics/model_ablation_mathqa_mixedboost

echo "========== WAIT FOR DS7B MATH500 long1024 =========="
date

while ps -ef | grep -E "run_math500_deepseek7b_long1024_boost_parallel_now|math500_deepseek7b_long1024|generator_deepseek_r1_distill_qwen7b_math500_long1024" | grep -v grep >/dev/null
do
  echo "[WAIT] current DS7B MATH500 long1024 still running"
  date
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  sleep 300
done

echo "========== START DS7B MathQA choice-aware extra =========="
date

for SEED in 606 707 808
do
  echo
  echo "========== MathQA choice-aware seed=${SEED} =========="
  python scripts/generate_numeric_trajectories_resume.py \
    --input data/processed/unified/model_ablation_boost/mathqa_deepseek7b_has_disagreement_choiceprompt.jsonl \
    --output data/processed/trajectories/model_ablation_boost/mathqa_deepseek7b_choice_extra_seed${SEED}.jsonl \
    --generator_config configs/model/generator_deepseek_r1_distill_qwen7b_mathqa_choice.yaml \
    --dataset mathqa \
    --n_traj 4 \
    --max_samples 0 \
    --max_new_tokens 384 \
    --temperature 0.95 \
    --top_p 0.95 \
    --seed ${SEED} \
    2>&1 | tee outputs/logs/model_ablation_boost/generate_mathqa_deepseek7b_choice_extra_seed${SEED}.log

  wc -l data/processed/trajectories/model_ablation_boost/mathqa_deepseek7b_choice_extra_seed${SEED}.jsonl || true
done

echo "========== REEVAL: choice-aware extras only =========="
python scripts/reeval_mathqa_option_mapping_confirm.py \
  --scope data/processed/unified/model_ablation/mathqa_scope.jsonl \
  --base data/processed/trajectories/model_ablation/mathqa_deepseek7b_base_3traj.jsonl \
  --extras \
    data/processed/trajectories/model_ablation_boost/mathqa_deepseek7b_choice_extra_seed606.jsonl \
    data/processed/trajectories/model_ablation_boost/mathqa_deepseek7b_choice_extra_seed707.jsonl \
    data/processed/trajectories/model_ablation_boost/mathqa_deepseek7b_choice_extra_seed808.jsonl \
  --targets outputs/targets/model_ablation/mathqa_deepseek7b_has_disagreement_ids.txt \
  --target_jsonl data/processed/unified/model_ablation/mathqa_deepseek7b_has_disagreement.jsonl \
  --out_dir outputs/metrics/model_ablation_mathqa_choiceboost \
  --prefix mathqa_deepseek7b_choiceboost \
  2>&1 | tee outputs/logs/model_ablation_boost/reeval_mathqa_deepseek7b_choiceboost.log

echo "========== REEVAL: original extras + choice-aware extras =========="
python scripts/reeval_mathqa_option_mapping_confirm.py \
  --scope data/processed/unified/model_ablation/mathqa_scope.jsonl \
  --base data/processed/trajectories/model_ablation/mathqa_deepseek7b_base_3traj.jsonl \
  --extras \
    data/processed/trajectories/model_ablation/mathqa_deepseek7b_extra_seed42.jsonl \
    data/processed/trajectories/model_ablation/mathqa_deepseek7b_extra_seed101.jsonl \
    data/processed/trajectories/model_ablation/mathqa_deepseek7b_extra_seed202.jsonl \
    data/processed/trajectories/model_ablation_boost/mathqa_deepseek7b_choice_extra_seed606.jsonl \
    data/processed/trajectories/model_ablation_boost/mathqa_deepseek7b_choice_extra_seed707.jsonl \
    data/processed/trajectories/model_ablation_boost/mathqa_deepseek7b_choice_extra_seed808.jsonl \
  --targets outputs/targets/model_ablation/mathqa_deepseek7b_has_disagreement_ids.txt \
  --target_jsonl data/processed/unified/model_ablation/mathqa_deepseek7b_has_disagreement.jsonl \
  --out_dir outputs/metrics/model_ablation_mathqa_mixedboost \
  --prefix mathqa_deepseek7b_mixedboost \
  2>&1 | tee outputs/logs/model_ablation_boost/reeval_mathqa_deepseek7b_mixedboost.log

echo "========== DONE MathQA choice boost =========="
date
