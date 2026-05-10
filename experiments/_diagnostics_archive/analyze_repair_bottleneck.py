import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import Counter
from utils.eval_utils import is_correct_prediction

def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.open("r", encoding="utf-8") if x.strip()]

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

def safe_bool(x):
    return bool(x) if x is not None else False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair_jsonl", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--dataset", default="gsm8k")
    args = ap.parse_args()

    rows = read_jsonl(args.repair_jsonl)
    trajs = {r.get("trajectory_id"): r for r in read_jsonl(args.trajectories)}

    total = len(rows)
    n_bad = n_good = 0
    triggered_bad = triggered_good = 0
    bad_not_triggered = good_not_triggered = 0
    rewritten_bad = rewritten_good = 0
    keep_bad = keep_good = 0
    malformed = 0
    recovered = harmed = preserved_good = 0
    rewrite_but_wrong_bad = 0
    action_counter = Counter()
    safe_reason_counter = Counter()

    for r in rows:
        tid = r.get("trajectory_id")
        tr = trajs.get(tid, {})
        gold = str(r.get("gold_answer") or tr.get("gold_answer") or "")
        before_ans = extract_answer_from_steps(tr.get("steps", []))

        before_ok = int(r.get("original_is_correct")) if "original_is_correct" in r else int(
            is_correct_prediction(before_ans, gold, answer_mode="numeric")
        )

        triggered = safe_bool(r.get("triggered"))
        decision = r.get("repair_decision")
        safe_reason = r.get("safe_reason")
        action_counter[decision] += 1
        safe_reason_counter[safe_reason] += 1

        if before_ok:
            n_good += 1
        else:
            n_bad += 1

        if not triggered:
            if before_ok:
                good_not_triggered += 1
            else:
                bad_not_triggered += 1
            continue

        if before_ok:
            triggered_good += 1
        else:
            triggered_bad += 1

        if decision == "KEEP":
            if before_ok:
                keep_good += 1
            else:
                keep_bad += 1
            if safe_reason and "invalid" in str(safe_reason):
                malformed += 1

        if decision == "REWRITE":
            repaired_ans = str(r.get("repaired_final_answer", "") or "").strip()
            after_ok = int(is_correct_prediction(repaired_ans, gold, answer_mode="numeric"))

            if before_ok:
                rewritten_good += 1
                if after_ok:
                    preserved_good += 1
                else:
                    harmed += 1
            else:
                rewritten_bad += 1
                if after_ok:
                    recovered += 1
                else:
                    rewrite_but_wrong_bad += 1

    def div(a, b):
        return a / b if b else 0.0

    summary = {
        "file": args.repair_jsonl,
        "total": total,
        "n_bad": n_bad,
        "n_good": n_good,
        "bad_not_triggered": bad_not_triggered,
        "good_not_triggered": good_not_triggered,
        "triggered_bad": triggered_bad,
        "triggered_good": triggered_good,
        "trigger_precision_bad": div(triggered_bad, triggered_bad + triggered_good),
        "bad_trigger_recall": div(triggered_bad, n_bad),
        "rewritten_bad": rewritten_bad,
        "rewritten_good": rewritten_good,
        "keep_bad": keep_bad,
        "keep_good": keep_good,
        "malformed_or_invalid_keep": malformed,
        "recovered": recovered,
        "harmed": harmed,
        "preserved_good_by_rewrite": preserved_good,
        "rewrite_but_wrong_bad": rewrite_but_wrong_bad,
        "bad_rewrite_recovery_rate": div(recovered, rewritten_bad),
        "good_rewrite_harm_rate": div(harmed, rewritten_good),
        "action_counter": dict(action_counter),
        "safe_reason_counter": dict(safe_reason_counter),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n| Metric | Value |")
    print("|---|---:|")
    for k, v in summary.items():
        if isinstance(v, dict):
            continue
        if isinstance(v, float):
            print(f"| {k} | {v:.4f} |")
        else:
            print(f"| {k} | {v} |")

if __name__ == "__main__":
    main()
