import argparse
import json
from pathlib import Path
from collections import Counter


def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        raise FileNotFoundError(fp)
    return [json.loads(x) for x in p.open("r", encoding="utf-8") if x.strip()]


def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_success", required=True)
    ap.add_argument("--repair_jsonl", required=True)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--level_dir", default="atom_level")
    ap.add_argument("--split", default="test")
    ap.add_argument("--window_radius", type=int, default=0)
    args = ap.parse_args()

    base_rows = read_jsonl(args.base_success)
    repair_rows = read_jsonl(args.repair_jsonl)

    # 先复制 success label，保证 train_hf 的主成功标签还能正常读。
    success_out = Path(args.out_base) / "success" / args.level_dir / args.dataset / f"{args.split}.jsonl"
    write_jsonl(success_out, base_rows)

    repair_map = {}

    for r in repair_rows:
        if not r.get("triggered"):
            continue

        tid = r.get("trajectory_id")
        trigger_idx = r.get("trigger_idx", None)
        if tid is None or trigger_idx is None:
            continue

        original_ok = bool(r.get("original_is_correct", False))
        recovered = bool(r.get("recovered", False))
        harmed = bool(r.get("harmed_good", False))
        decision = str(r.get("repair_decision", ""))

        # repairability 表示：这个 prefix 是否值得交给 repair。
        if (not original_ok) and recovered:
            repairability = 1.0
            error_type = "repairable_error"
        elif (not original_ok) and decision == "REWRITE" and (not recovered):
            repairability = 0.0
            error_type = "unrecovered_error"
        elif original_ok and harmed:
            repairability = 0.0
            error_type = "unsafe_good"
        elif original_ok:
            repairability = 0.0
            error_type = "false_alarm_good"
        else:
            repairability = 0.0
            error_type = "unknown_error"

        for d in range(-args.window_radius, args.window_radius + 1):
            k = int(trigger_idx) + d
            if k >= 0:
                repair_map[(tid, k)] = {
                    "repairability": repairability,
                    "error_type": error_type,
                }

    error_rows = []
    repair_rows_out = []
    cnt = Counter()

    for r in base_rows:
        tid = r.get("trajectory_id")
        k = r.get("prefix_num_units", None)
        prefix_id = r.get("prefix_id")

        label = {
            "prefix_id": prefix_id,
            "sample_id": r.get("sample_id"),
            "trajectory_id": tid,
            "prefix_num_units": k,
            "split": r.get("split", args.split),
            "dataset": r.get("dataset", args.dataset),
        }

        aux = {"repairability": 0.0, "error_type": "none"}
        if k is not None and (tid, int(k)) in repair_map:
            aux = repair_map[(tid, int(k))]

        er = dict(label)
        er["error_type"] = aux["error_type"]

        rr = dict(label)
        rr["repairability"] = aux["repairability"]

        error_rows.append(er)
        repair_rows_out.append(rr)
        cnt[(aux["error_type"], aux["repairability"])] += 1

    error_out = Path(args.out_base) / "error_type" / args.level_dir / args.dataset / f"{args.split}.jsonl"
    repair_out = Path(args.out_base) / "repairability" / args.level_dir / args.dataset / f"{args.split}.jsonl"

    write_jsonl(error_out, error_rows)
    write_jsonl(repair_out, repair_rows_out)

    print("saved success:", success_out)
    print("saved error_type:", error_out)
    print("saved repairability:", repair_out)
    print("rows:", len(base_rows))
    print("non-none error rows:", sum(1 for r in error_rows if r["error_type"] != "none"))
    print("positive repairability rows:", sum(1 for r in repair_rows_out if float(r["repairability"]) > 0))
    print("counter:")
    for k, v in cnt.most_common():
        print(k, v)


if __name__ == "__main__":
    main()
