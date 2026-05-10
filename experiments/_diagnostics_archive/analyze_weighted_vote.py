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
    args = ap.parse_args()

    pred_rows = read_jsonl(args.predictions)
    traj_rows = read_jsonl(args.trajectories)

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
        if mode == "median":
            vals2 = sorted(vals)
            return vals2[len(vals2) // 2]
        return vals[-1]

    by_sample = defaultdict(list)

    for tr in traj_rows:
        tid = tr["trajectory_id"]
        sid = tr["sample_id"]
        gold = str(tr["gold_answer"])

        ans_raw = extract_answer_from_steps(tr.get("steps", []))
        ans_norm = norm_num(ans_raw)

        by_sample[sid].append({
            "tid": tid,
            "gold": gold,
            "answer_raw": ans_raw,
            "answer_norm": ans_norm,
            "correct": int(is_correct_prediction(ans_raw, gold, answer_mode="numeric")),
        })

    modes = ["last", "tail3", "tail5", "min_tail5", "mean", "median"]
    n = len(by_sample)

    first = 0
    majority = 0
    oracle_any = 0
    pce_top1 = {m: 0 for m in modes}
    weighted_correct = {m: 0 for m in modes}

    for sid, rs in by_sample.items():
        rs = sorted(rs, key=lambda x: traj_order(x["tid"]))
        gold = rs[0]["gold"]

        first += rs[0]["correct"]
        oracle_any += int(any(x["correct"] for x in rs))

        votes = Counter(x["answer_norm"] for x in rs)
        majority_ans = votes.most_common(1)[0][0]
        majority += int(is_correct_prediction(majority_ans, gold, answer_mode="numeric"))

        for mode in modes:
            best = max(rs, key=lambda x: score(x["tid"], mode))
            pce_top1[mode] += int(is_correct_prediction(best["answer_raw"], gold, answer_mode="numeric"))

            weighted_votes = defaultdict(float)
            for x in rs:
                weighted_votes[x["answer_norm"]] += score(x["tid"], mode)
            pred = max(weighted_votes.items(), key=lambda z: z[1])[0]
            weighted_correct[mode] += int(is_correct_prediction(pred, gold, answer_mode="numeric"))

    print("| Method | Accuracy |")
    print("|---|---:|")
    print(f"| first | {first/n:.4f} |")
    print(f"| majority | {majority/n:.4f} |")
    for mode in modes:
        print(f"| pce_top1_{mode} | {pce_top1[mode]/n:.4f} |")
    for mode in modes:
        print(f"| weighted_{mode} | {weighted_correct[mode]/n:.4f} |")
    print(f"| oracle_any | {oracle_any/n:.4f} |")


if __name__ == "__main__":
    main()
