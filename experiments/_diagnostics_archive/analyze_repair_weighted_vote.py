import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_utils import is_correct_prediction


def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        raise FileNotFoundError(fp)
    return [json.loads(x) for x in p.open("r", encoding="utf-8") if x.strip()]


def norm_num(x):
    x = str(x or "").replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", x)
    if not nums:
        return str(x).strip()
    v = nums[-1]
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v


def extract_answer_from_steps(steps):
    for s in reversed(steps or []):
        s = str(s)
        if "Final Answer:" in s:
            return s.split("Final Answer:", 1)[-1].strip()
        if "Answer:" in s:
            return s.split("Answer:", 1)[-1].strip()
        if "####" in s:
            return s.split("####", 1)[-1].strip()
    joined = "\n".join(str(x) for x in steps or [])
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", joined.replace(",", ""))
    return nums[-1] if nums else joined.strip()


def traj_order(tid):
    m = re.search(r"_traj_(\d+)$", str(tid))
    return int(m.group(1)) if m else 999999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--repair_jsonl", required=True)
    args = ap.parse_args()

    pred_rows = read_jsonl(args.predictions)
    traj_rows = read_jsonl(args.trajectories)
    repair_rows = read_jsonl(args.repair_jsonl)

    repair_by_tid = {r["trajectory_id"]: r for r in repair_rows}

    scores_by_tid = defaultdict(list)
    for r in pred_rows:
        if "success_prob" not in r:
            continue
        tid = r["trajectory_id"]
        k = int(r.get("prefix_num_units", 0))
        p = float(r["success_prob"])
        scores_by_tid[tid].append((k, p))

    def score(tid, mode):
        arr = sorted(scores_by_tid.get(tid, []), key=lambda x: x[0])
        vals = [x[1] for x in arr]
        if not vals:
            return 1.0
        if mode == "last":
            return vals[-1]
        if mode == "tail3":
            return sum(vals[-3:]) / min(3, len(vals))
        if mode == "tail5":
            return sum(vals[-5:]) / min(5, len(vals))
        if mode == "min_tail5":
            return min(vals[-5:])
        if mode == "mean":
            return sum(vals) / len(vals)
        return vals[-1]

    modes = ["last", "tail3", "tail5", "min_tail5", "mean"]

    by_sample = defaultdict(list)

    for tr in traj_rows:
        tid = tr["trajectory_id"]
        sid = tr["sample_id"]
        gold = str(tr["gold_answer"])

        before_raw = extract_answer_from_steps(tr.get("steps", []))
        before = norm_num(before_raw)

        after = before
        rr = repair_by_tid.get(tid)
        if rr and rr.get("repair_decision") == "REWRITE" and str(rr.get("repaired_final_answer", "")).strip():
            after = norm_num(rr.get("repaired_final_answer"))

        by_sample[sid].append({
            "tid": tid,
            "gold": gold,
            "before": before,
            "after": after,
            "before_ok": int(is_correct_prediction(before_raw, gold, answer_mode="numeric")),
            "after_ok": int(is_correct_prediction(after, gold, answer_mode="numeric")),
        })

    n = len(by_sample)

    before_majority = after_majority = 0
    before_weighted = {m: 0 for m in modes}
    after_weighted = {m: 0 for m in modes}

    for sid, rs in by_sample.items():
        rs = sorted(rs, key=lambda x: traj_order(x["tid"]))
        gold = rs[0]["gold"]

        vb = Counter(x["before"] for x in rs)
        va = Counter(x["after"] for x in rs)

        before_majority += int(is_correct_prediction(vb.most_common(1)[0][0], gold, answer_mode="numeric"))
        after_majority += int(is_correct_prediction(va.most_common(1)[0][0], gold, answer_mode="numeric"))

        for mode in modes:
            wb = defaultdict(float)
            wa = defaultdict(float)

            for x in rs:
                w = score(x["tid"], mode)
                wb[x["before"]] += w
                wa[x["after"]] += w

            pred_b = max(wb.items(), key=lambda z: z[1])[0]
            pred_a = max(wa.items(), key=lambda z: z[1])[0]

            before_weighted[mode] += int(is_correct_prediction(pred_b, gold, answer_mode="numeric"))
            after_weighted[mode] += int(is_correct_prediction(pred_a, gold, answer_mode="numeric"))

    print("| Method | Before | After Repair | Gain |")
    print("|---|---:|---:|---:|")
    print(f"| majority | {before_majority/n:.4f} | {after_majority/n:.4f} | {(after_majority-before_majority)/n:.4f} |")
    for mode in modes:
        b = before_weighted[mode] / n
        a = after_weighted[mode] / n
        print(f"| weighted_{mode} | {b:.4f} | {a:.4f} | {a-b:.4f} |")


if __name__ == "__main__":
    main()
