#!/usr/bin/env bash
set -e

cd /root/pce_reasoning_project/project

export PYTHONPATH=/root/pce_reasoning_project/project:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

GSM_LABEL_DIR=data/processed/labels_gsm8k_official_full_local/success/atom_level/gsm8k
GSM_TRAIN_FILES=()

if [ -s "$GSM_LABEL_DIR/train.jsonl" ]; then
  GSM_TRAIN_FILES+=("$GSM_LABEL_DIR/train.jsonl")
fi

if [ -s "$GSM_LABEL_DIR/val.jsonl" ]; then
  GSM_TRAIN_FILES+=("$GSM_LABEL_DIR/val.jsonl")
fi

if [ ${#GSM_TRAIN_FILES[@]} -eq 0 ]; then
  echo "[WARN] train/val not found, fallback to test.jsonl."
  GSM_TRAIN_FILES=("$GSM_LABEL_DIR/test.jsonl")
fi

for SEED in 42 101 202
do
  echo "==================== SPLIT SEED = $SEED ===================="

  for DS in svamp asdiv
  do
    NTRAIN=100
    SPLIT_DIR=data/processed/pce_target_splits/${DS}_train${NTRAIN}_seed${SEED}

    echo "========== split $DS seed=$SEED =========="
    python experiments/split_cross_pce_by_sample.py \
      --labels_jsonl data/processed/labels_cross_pce_zero/success/atom_level/${DS}/test.jsonl \
      --trajectories_jsonl data/processed/trajectories/${DS}/test_local_3traj_200.jsonl \
      --dataset "$DS" \
      --n_train "$NTRAIN" \
      --seed "$SEED" \
      --out_dir "$SPLIT_DIR"

    echo "========== target-only PCE $DS seed=$SEED =========="
    python experiments/predict_cross_pce_light_zeroshot.py \
      --train_jsonls ${SPLIT_DIR}/train_labels.jsonl \
      --test_jsonl ${SPLIT_DIR}/test_labels.jsonl \
      --out_jsonl outputs/predictions/${DS}_targettrained_pce_light_train${NTRAIN}_seed${SEED}.jsonl \
      --model_out outputs/checkpoints/${DS}_targettrained_pce_light_train${NTRAIN}_seed${SEED}.pkl \
      --feature_set prefix_plus_question_answer \
      --max_features 100000 \
      --min_df 1 \
      --C 2.0 \
      2>&1 | tee outputs/logs/${DS}_targettrained_pce_light_train${NTRAIN}_seed${SEED}.log

    python experiments/eval_cross_pce_weighted_selection.py \
      --predictions outputs/predictions/${DS}_targettrained_pce_light_train${NTRAIN}_seed${SEED}.jsonl \
      --trajectories ${SPLIT_DIR}/test_trajectories.jsonl \
      --dataset "$DS" \
      --tail_k 5 \
      --out_json outputs/metrics/${DS}_targettrained_pce_weighted_tail5_train${NTRAIN}_seed${SEED}.json \
      --out_jsonl outputs/predictions/${DS}_targettrained_pce_weighted_tail5_train${NTRAIN}_seed${SEED}_details.jsonl

    echo "========== GSM8K+target PCE $DS seed=$SEED =========="
    python experiments/predict_cross_pce_light_zeroshot.py \
      --train_jsonls "${GSM_TRAIN_FILES[@]}" ${SPLIT_DIR}/train_labels.jsonl \
      --test_jsonl ${SPLIT_DIR}/test_labels.jsonl \
      --out_jsonl outputs/predictions/${DS}_gsm8k_plus_target_pce_light_train${NTRAIN}_seed${SEED}.jsonl \
      --model_out outputs/checkpoints/${DS}_gsm8k_plus_target_pce_light_train${NTRAIN}_seed${SEED}.pkl \
      --feature_set prefix_plus_question_answer \
      --max_features 200000 \
      --min_df 2 \
      --C 2.0 \
      2>&1 | tee outputs/logs/${DS}_gsm8k_plus_target_pce_light_train${NTRAIN}_seed${SEED}.log

    python experiments/eval_cross_pce_weighted_selection.py \
      --predictions outputs/predictions/${DS}_gsm8k_plus_target_pce_light_train${NTRAIN}_seed${SEED}.jsonl \
      --trajectories ${SPLIT_DIR}/test_trajectories.jsonl \
      --dataset "$DS" \
      --tail_k 5 \
      --out_json outputs/metrics/${DS}_gsm8k_plus_target_pce_weighted_tail5_train${NTRAIN}_seed${SEED}.json \
      --out_jsonl outputs/predictions/${DS}_gsm8k_plus_target_pce_weighted_tail5_train${NTRAIN}_seed${SEED}_details.jsonl
  done

  DS=multiarith
  NTRAIN=90
  SPLIT_DIR=data/processed/pce_target_splits/${DS}_train${NTRAIN}_seed${SEED}

  echo "========== split $DS seed=$SEED =========="
  python experiments/split_cross_pce_by_sample.py \
    --labels_jsonl data/processed/labels_cross_pce_zero/success/atom_level/${DS}/test.jsonl \
    --trajectories_jsonl data/processed/trajectories/${DS}/test_local_3traj_200.jsonl \
    --dataset "$DS" \
    --n_train "$NTRAIN" \
    --seed "$SEED" \
    --out_dir "$SPLIT_DIR"

  echo "========== target-only PCE $DS seed=$SEED =========="
  python experiments/predict_cross_pce_light_zeroshot.py \
    --train_jsonls ${SPLIT_DIR}/train_labels.jsonl \
    --test_jsonl ${SPLIT_DIR}/test_labels.jsonl \
    --out_jsonl outputs/predictions/${DS}_targettrained_pce_light_train${NTRAIN}_seed${SEED}.jsonl \
    --model_out outputs/checkpoints/${DS}_targettrained_pce_light_train${NTRAIN}_seed${SEED}.pkl \
    --feature_set prefix_plus_question_answer \
    --max_features 100000 \
    --min_df 1 \
    --C 2.0 \
    2>&1 | tee outputs/logs/${DS}_targettrained_pce_light_train${NTRAIN}_seed${SEED}.log

  python experiments/eval_cross_pce_weighted_selection.py \
    --predictions outputs/predictions/${DS}_targettrained_pce_light_train${NTRAIN}_seed${SEED}.jsonl \
    --trajectories ${SPLIT_DIR}/test_trajectories.jsonl \
    --dataset "$DS" \
    --tail_k 5 \
    --out_json outputs/metrics/${DS}_targettrained_pce_weighted_tail5_train${NTRAIN}_seed${SEED}.json \
    --out_jsonl outputs/predictions/${DS}_targettrained_pce_weighted_tail5_train${NTRAIN}_seed${SEED}_details.jsonl

  echo "========== GSM8K+target PCE $DS seed=$SEED =========="
  python experiments/predict_cross_pce_light_zeroshot.py \
    --train_jsonls "${GSM_TRAIN_FILES[@]}" ${SPLIT_DIR}/train_labels.jsonl \
    --test_jsonl ${SPLIT_DIR}/test_labels.jsonl \
    --out_jsonl outputs/predictions/${DS}_gsm8k_plus_target_pce_light_train${NTRAIN}_seed${SEED}.jsonl \
    --model_out outputs/checkpoints/${DS}_gsm8k_plus_target_pce_light_train${NTRAIN}_seed${SEED}.pkl \
    --feature_set prefix_plus_question_answer \
    --max_features 200000 \
    --min_df 2 \
    --C 2.0 \
    2>&1 | tee outputs/logs/${DS}_gsm8k_plus_target_pce_light_train${NTRAIN}_seed${SEED}.log

  python experiments/eval_cross_pce_weighted_selection.py \
    --predictions outputs/predictions/${DS}_gsm8k_plus_target_pce_light_train${NTRAIN}_seed${SEED}.jsonl \
    --trajectories ${SPLIT_DIR}/test_trajectories.jsonl \
    --dataset "$DS" \
    --tail_k 5 \
    --out_json outputs/metrics/${DS}_gsm8k_plus_target_pce_weighted_tail5_train${NTRAIN}_seed${SEED}.json \
    --out_jsonl outputs/predictions/${DS}_gsm8k_plus_target_pce_weighted_tail5_train${NTRAIN}_seed${SEED}_details.jsonl
done
