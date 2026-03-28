from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--name", default="setting")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = load_rows(args.predictions)
    y = [1 - int(r["label_success"]) for r in rows]  # bad = 1
    p = [float(r["success_prob"]) for r in rows]

    pred_bad = [1 if x < args.tau else 0 for x in p]

    tp = sum(1 for yy, pp in zip(y, pred_bad) if yy == 1 and pp == 1)
    fp = sum(1 for yy, pp in zip(y, pred_bad) if yy == 0 and pp == 1)
    tn = sum(1 for yy, pp in zip(y, pred_bad) if yy == 0 and pp == 0)
    fn = sum(1 for yy, pp in zip(y, pred_bad) if yy == 1 and pp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    flagged = sum(pred_bad) / len(pred_bad)

    result = {
        "setting": args.name,
        "tau": args.tau,
        "precision_bad": precision,
        "recall_bad": recall,
        "false_alarm_rate": false_alarm,
        "flagged_rate": flagged,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

    # 如果有 trajectory_id 和 prefix_progress，再算 avg_first_trigger_progress
    has_traj = all("trajectory_id" in r for r in rows)
    has_prog = all("prefix_progress" in r for r in rows)

    if has_traj and has_prog:
        by_traj = defaultdict(list)
        for r in rows:
            by_traj[r["trajectory_id"]].append(r)

        first_positions = []
        for tid, traj_rows in by_traj.items():
            traj_rows = sorted(
                traj_rows, key=lambda x: float(x.get("prefix_progress", 0.0))
            )
            hit = None
            for r in traj_rows:
                if float(r["success_prob"]) < args.tau:
                    hit = float(r.get("prefix_progress", 0.0))
                    break
            if hit is not None:
                first_positions.append(hit)

        result["num_trajectories_with_trigger"] = len(first_positions)
        result["avg_first_trigger_progress"] = (
            sum(first_positions) / len(first_positions) if first_positions else None
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
