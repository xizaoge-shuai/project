from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def parse_csv_floats(x: str) -> List[float]:
    return [float(v.strip()) for v in x.split(",") if v.strip()]


def load_rows(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--taus", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rows = load_rows(args.predictions)
    y = [1 - int(r["label_success"]) for r in rows]  # bad = 1
    p = [float(r["success_prob"]) for r in rows]
    taus = parse_csv_floats(args.taus)

    results = []
    print("tau\tprecision_bad\trecall_bad\tfalse_alarm_rate\tflagged_rate")
    for tau in taus:
        pred_bad = [1 if x < tau else 0 for x in p]

        tp = sum(1 for yy, pp in zip(y, pred_bad) if yy == 1 and pp == 1)
        fp = sum(1 for yy, pp in zip(y, pred_bad) if yy == 0 and pp == 1)
        tn = sum(1 for yy, pp in zip(y, pred_bad) if yy == 0 and pp == 0)
        fn = sum(1 for yy, pp in zip(y, pred_bad) if yy == 1 and pp == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        flagged = sum(pred_bad) / len(pred_bad)

        row = {
            "tau": tau,
            "precision_bad": precision,
            "recall_bad": recall,
            "false_alarm_rate": false_alarm,
            "flagged_rate": flagged,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        results.append(row)

        print(
            f"{tau:.2f}\t{precision:.4f}\t{recall:.4f}\t{false_alarm:.4f}\t{flagged:.4f}"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
