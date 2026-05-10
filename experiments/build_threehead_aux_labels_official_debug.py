import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter


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
    args = ap.parse_args()

    base_rows = read_jsonl(args.base_success)
    repair_rows = read_jsonl(args.repair_jsonl)

    by_tid = defaultdict(list)
    for r in base_rows:
        by_tid[r.get("trajectory_id")].append(r)

    for tid in by_tid:
        by_tid[tid].sort(key=lambda x: float(x.get("prefix_progress", 0.0)))

    aux_by_prefix_id = {}

    matched = 0
    unmatched = 0

    for rr in repair_rows:
        if not rr.get("triggered"):
            continue

        tid = rr.get("trajectory_id")
        candidates = by_tid.get(tid, [])
        if not candidates:
            unmatched += 1
            continue

        target_prog = float(rr.get("trigger_progress", -1.0))
        target_idx = rr.get("trigger_idx", None)

        if target_prog >= 0:
            best = min(
                candidates,
                key=lambda x: abs(float(x.get("prefix_progress", 0.0)) - target_prog)
            )
        elif target_idx is not None:
            best = min(
                candidates,
                key=lambda x: abs(int(x.get("prefix_num_units", 0)) - int(target_idx))
            )
        else:
            unmatched += 1
            continue

        original_ok = bool(rr.get("original_is_correct", False))
        recovered = bool(rr.get("recovered", False))
        harmed = bool(rr.get("harmed_good", False))
        decision = str(rr.get("repair_decision", ""))

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

        aux_by_prefix_id[best["prefix_id"]] = {
            "error_type": error_type,
            "repairability": repairability,
            "matched_trigger_progress": rr.get("trigger_progress"),
            "matched_prefix_progress": best.get("prefix_progress"),
        }
        matched += 1

    success_out = Path(args.out_base) / "success" / args.level_dir / args.dataset / f"{args.split}.jsonl"
    error_out = Path(args.out_base) / "error_type" / args.level_dir / args.dataset / f"{args.split}.jsonl"
    repair_out = Path(args.out_base) / "repairability" / args.level_dir / args.dataset / f"{args.split}.jsonl"

    write_jsonl(success_out, base_rows)

    error_rows = []
    repair_rows_out = []

    cnt = Counter()

    for r in base_rows:
        prefix_id = r.get("prefix_id")
        aux = aux_by_prefix_id.get(prefix_id, {
            "error_type": "none",
            "repairability": 0.0,
            "matched_trigger_progress": None,
            "matched_prefix_progress": r.get("prefix_progress"),
        })

        common = {
            "prefix_id": prefix_id,
            "sample_id": r.get("sample_id"),
            "trajectory_id": r.get("trajectory_id"),
            "prefix_num_units": r.get("prefix_num_units"),
            "prefix_progress": r.get("prefix_progress"),
            "split": r.get("split", args.split),
            "dataset": r.get("dataset", args.dataset),
        }

        er = dict(common)
        er["error_type"] = aux["error_type"]

        rep = dict(common)
        rep["repairability"] = aux["repairability"]

        error_rows.append(er)
        repair_rows_out.append(rep)

        cnt[(aux["error_type"], aux["repairability"])] += 1

    write_jsonl(error_out, error_rows)
    write_jsonl(repair_out, repair_rows_out)

    print("saved success:", success_out)
    print("saved error_type:", error_out)
    print("saved repairability:", repair_out)
    print("base rows:", len(base_rows))
    print("triggered matched:", matched)
    print("triggered unmatched:", unmatched)
    print("non-none error rows:", sum(1 for r in error_rows if r["error_type"] != "none"))
    print("positive repairability rows:", sum(1 for r in repair_rows_out if float(r["repairability"]) > 0))
    print("counter:")
    for k, v in cnt.most_common():
        print(k, v)


if __name__ == "__main__":
    main()
