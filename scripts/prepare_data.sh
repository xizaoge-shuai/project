#!/usr/bin/env bash
set -e
python data/download.py --dataset all
python data/preprocess.py --dataset all
