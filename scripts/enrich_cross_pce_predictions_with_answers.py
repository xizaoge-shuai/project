#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


parser = argparse.ArgumentParser()
parser.add_argument("--predictions", required=True)
parser.add_argument("--trajectories", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

pred_path = Path(args.predictions)
traj_path = Path(args.trajectories)
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

trajectories = read_jsonl(traj_path)
traj_map = {}

for row in trajectories:
    trajectory_id = first_nonempty(
        row.get("trajectory_id"),
        row.get("traj_id"),
        row.get("id"),
    )
    if not trajectory_id:
        raise KeyError(f"trajectory row missing id: {row}")
    traj_map[trajectory_id] = row

n_rows = 0
n_missing_trajectory = 0
n_blank_before = 0
n_blank_after = 0

with pred_path.open(encoding="utf-8") as fin, \
     out_path.open("w", encoding="utf-8") as fout:

    for line_no, line in enumerate(fin, 1):
        if not line.strip():
            continue

        n_rows += 1
        pred = json.loads(line)

        trajectory_id = first_nonempty(
            pred.get("trajectory_id"),
            pred.get("traj_id"),
        )

        traj = traj_map.get(trajectory_id)
        if traj is None:
            n_missing_trajectory += 1
            continue

        old_answer = first_nonempty(
            pred.get("final_answer"),
            pred.get("answer"),
        )
        if not old_answer:
            n_blank_before += 1

        final_answer = first_nonempty(
            pred.get("final_answer"),
            pred.get("answer"),
            traj.get("final_answer"),
            traj.get("answer"),
        )

        gold_answer = first_nonempty(
            pred.get("gold_answer"),
            traj.get("gold_answer"),
            traj.get("answer"),
        )

        question = first_nonempty(
            pred.get("question"),
            traj.get("question"),
        )

        sample_id = first_nonempty(
            pred.get("sample_id"),
            traj.get("sample_id"),
            traj.get("id"),
        )

        pred["trajectory_id"] = trajectory_id
        pred["sample_id"] = sample_id
        pred["question"] = question
        pred["gold_answer"] = gold_answer
        pred["final_answer"] = final_answer
        pred["answer"] = final_answer

        if not final_answer:
            n_blank_after += 1

        fout.write(json.dumps(pred, ensure_ascii=False) + "\n")

print(json.dumps({
    "predictions": str(pred_path),
    "trajectories": str(traj_path),
    "output": str(out_path),
    "n_rows": n_rows,
    "n_missing_trajectory": n_missing_trajectory,
    "n_blank_answer_before": n_blank_before,
    "n_blank_answer_after": n_blank_after,
}, ensure_ascii=False, indent=2))

if n_missing_trajectory:
    raise SystemExit(
        f"{n_missing_trajectory} prediction rows lack matching trajectories"
    )
