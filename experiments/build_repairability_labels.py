from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--repair_threshold",
        type=float,
        default=0.34,
        help="local_success_rate >= repair_threshold 视为可修复",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    out_rows = []

    for r in rows:
        # 优先直接读取已有字段
        if "repairable" in r:
            repairable = int(r["repairable"])
            local_success_rate = float(r.get("local_success_rate", repairable))
        else:
            # 否则尝试从 rollout 局部成功率生成
            if "local_success_rate" in r:
                local_success_rate = float(r["local_success_rate"])
            elif "rollout_success_rate" in r:
                local_success_rate = float(r["rollout_success_rate"])
            else:
                raise ValueError(
                    "Input rows must contain either 'repairable' or "
                    "'local_success_rate' / 'rollout_success_rate'."
                )

            repairable = 1 if local_success_rate >= args.repair_threshold else 0

        out_rows.append(
            {
                "prefix_id": r.get("prefix_id", ""),
                "trajectory_id": r.get("trajectory_id", ""),
                "sample_id": r.get("sample_id", ""),
                "dataset": r.get("dataset", ""),
                "split": r.get("split", ""),
                "level": r.get("level", "atom"),
                "prefix_progress": float(r.get("prefix_progress", 0.0) or 0.0),
                "prefix_num_units": r.get("prefix_num_units", 0),
                "local_success_rate": local_success_rate,
                "repairable": repairable,
            }
        )

    write_jsonl(args.output, out_rows)
    print(f"Saved {len(out_rows)} repairability labels to {args.output}")


if __name__ == "__main__":
    main()
