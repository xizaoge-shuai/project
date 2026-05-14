#!/usr/bin/env bash
set -e

echo "========== Dataset configs =========="
find configs/dataset -maxdepth 1 -type f | sort || true

echo
echo "========== Raw data =========="
for DS in svamp multiarith asdiv strategyqa hotpotqa gsm8k
do
  echo "--- $DS raw ---"
  find data/raw -maxdepth 4 -type f | grep -Ei "$DS" | head -20 || true
done

echo
echo "========== Unified data =========="
for DS in svamp multiarith asdiv strategyqa hotpotqa gsm8k
do
  echo "--- $DS unified ---"
  find data/processed/unified -maxdepth 4 -type f | grep -Ei "$DS" | head -20 || true
done

echo
echo "========== Trajectories =========="
for DS in svamp multiarith asdiv strategyqa hotpotqa gsm8k
do
  echo "--- $DS trajectories ---"
  find data/processed/trajectories -maxdepth 4 -type f | grep -Ei "$DS" | head -20 || true
done

echo
echo "========== Existing predictions =========="
for DS in svamp multiarith asdiv strategyqa hotpotqa gsm8k
do
  echo "--- $DS predictions ---"
  find outputs/predictions -maxdepth 2 -type f | grep -Ei "$DS" | head -20 || true
done

echo
echo "========== Script help =========="
for S in \
  data/preprocess.py \
  data/build_trajectories.py \
  data/build_prefixes.py \
  data/build_labels.py \
  experiments/run_sample_level_selection.py \
  experiments/run_selective_resampling.py
do
  echo
  echo "----- $S -----"
  if [ -f "$S" ]; then
    python "$S" --help | head -80 || true
  else
    echo "MISSING"
  fi
done
