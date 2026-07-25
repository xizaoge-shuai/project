#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--min_progress", type=float, required=True)
args = parser.parse_args()

src = Path(args.input)
dst = Path(args.output)
dst.parent.mkdir(parents=True, exist_ok=True)

n_input = 0
n_output = 0

with src.open(encoding="utf-8") as fin, dst.open(
    "w", encoding="utf-8"
) as fout:
    for line in fin:
        if not line.strip():
            continue

        n_input += 1
        row = json.loads(line)

        progress = float(
            row.get(
                "prefix_progress",
                row.get("progress", 0.0),
            )
        )

        if progress < args.min_progress:
            continue

        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_output += 1

print(
    f"[DONE] {src} -> {dst}; "
    f"input={n_input}, output={n_output}, "
    f"min_progress={args.min_progress}"
)
