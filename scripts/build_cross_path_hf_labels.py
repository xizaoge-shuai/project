#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def row_rank(row: dict) -> tuple:
    progress = float(
        row.get("prefix_progress", row.get("progress", 0.0))
    )
    units = int(
        row.get(
            "prefix_num_units",
            row.get("trajectory_total_units", 0),
        )
    )
    text_len = len(str(row.get("prefix_text", "")))
    return progress, units, text_len


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--split", required=True)
args = parser.parse_args()

input_path = Path(args.input)
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)

last_prefix_by_trajectory: dict[str, dict] = {}
n_input = 0

with input_path.open(encoding="utf-8") as fin:
    for line_no, line in enumerate(fin, 1):
        if not line.strip():
            continue

        n_input += 1
        row = json.loads(line)

        trajectory_id = row.get("trajectory_id")
        if trajectory_id is None:
            raise KeyError(
                f"{input_path}:{line_no} missing trajectory_id"
            )

        trajectory_id = str(trajectory_id)
        previous = last_prefix_by_trajectory.get(trajectory_id)

        if previous is None or row_rank(row) > row_rank(previous):
            last_prefix_by_trajectory[trajectory_id] = row

n_progress_not_one = 0

with output_path.open("w", encoding="utf-8") as fout:
    for trajectory_id in sorted(last_prefix_by_trajectory):
        source = last_prefix_by_trajectory[trajectory_id]
        progress = row_rank(source)[0]

        if progress < 0.999:
            n_progress_not_one += 1

        row = dict(source)
        row.update({
            "dataset": args.dataset,
            "task": args.dataset,
            "split": args.split,
            "trajectory_id": trajectory_id,
            "prefix_id": f"{trajectory_id}::path",
            "level": "path_level",
            "prefix_progress": 1.0,
            "prefix_num_units": int(
                source.get(
                    "trajectory_total_units",
                    source.get("prefix_num_units", 1),
                )
            ),
            "pce_source": "cross_path_from_final_atom",
        })

        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

print(
    json.dumps(
        {
            "input": str(input_path),
            "output": str(output_path),
            "dataset": args.dataset,
            "split": args.split,
            "n_atom_prefix_rows": n_input,
            "n_path_rows": len(last_prefix_by_trajectory),
            "n_selected_progress_below_1": n_progress_not_one,
        },
        ensure_ascii=False,
        indent=2,
    )
)
