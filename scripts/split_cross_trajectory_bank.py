#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def sample_id(row, index):
    return str(
        row.get("sample_id")
        or row.get("question_id")
        or row.get("problem_id")
        or row.get("qid")
        or row.get("id")
        or index
    ).split("_traj_")[0]


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--seed", type=int, default=2027)
args = parser.parse_args()

rows_by_sample = defaultdict(list)

with open(args.input, encoding="utf-8") as f:
    for index, line in enumerate(f):
        if not line.strip():
            continue
        row = json.loads(line)
        sid = sample_id(row, index)
        row["sample_id"] = sid
        row["dataset"] = args.dataset
        rows_by_sample[sid].append(row)

ids = list(rows_by_sample)
random.Random(args.seed).shuffle(ids)

n = len(ids)
n_train = int(n * 0.60)
n_val = int(n * 0.20)

splits = {
    "train": set(ids[:n_train]),
    "val": set(ids[n_train:n_train + n_val]),
    "test": set(ids[n_train + n_val:]),
}

root = Path(args.out_dir) / args.dataset
root.mkdir(parents=True, exist_ok=True)

for split, split_ids in splits.items():
    out = root / f"{split}.jsonl"
    n_rows = 0

    with out.open("w", encoding="utf-8") as fw:
        for sid in ids:
            if sid not in split_ids:
                continue

            for row in rows_by_sample[sid]:
                row = dict(row)
                row["split"] = split
                fw.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(
        f"[WRITE] {out}: "
        f"samples={len(split_ids)}, trajectories={n_rows}"
    )
