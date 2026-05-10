import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_utils import is_correct_prediction


def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.open("r", encoding="utf-8") if x.strip()]


def extract_loose_number(text):
    text = str(text or "").replace(",", "")

    m = re.findall(r"Final Answer\s*:\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.I)
    if m:
        return m[-1]

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair_jsonl", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.repair_jsonl)

    triggered = 0
    triggered_bad = 0
    triggered_good = 0

    strict_recovered_bad = 0
    strict_harmed_good = 0

    loose_candidate = 0
    no_loose_answer = 0

    loose_bad_candidate = 0
    loose_good_candidate = 0

    loose_recovered_bad = 0
    loose_wrong_bad = 0

    loose_preserved_good = 0
    loose_harmed_good = 0

    for r in rows:
        if not r.get("triggered"):
            continue

        triggered += 1
        gold = str(r.get("gold_answer", ""))
        original_ok = bool(r.get("original_is_correct", False))

        if original_ok:
            triggered_good += 1
        else:
            triggered_bad += 1

        if r.get("recovered"):
            strict_recovered_bad += 1
        if r.get("harmed_good"):
            strict_harmed_good += 1

        loose_ans = extract_loose_number(r.get("repair_output", ""))
        if not loose_ans:
            no_loose_answer += 1
            continue

        loose_candidate += 1
        loose_ok = bool(is_correct_prediction(loose_ans, gold, answer_mode="numeric"))

        if original_ok:
            loose_good_candidate += 1
            if loose_ok:
                loose_preserved_good += 1
            else:
                loose_harmed_good += 1
        else:
            loose_bad_candidate += 1
            if loose_ok:
                loose_recovered_bad += 1
            else:
                loose_wrong_bad += 1

    def div(a, b):
        return a / b if b else 0.0

    summary = {
        "file": args.repair_jsonl,
        "triggered": triggered,
        "triggered_bad": triggered_bad,
        "triggered_good": triggered_good,
        "strict_recovered_bad": strict_recovered_bad,
        "strict_harmed_good": strict_harmed_good,
        "loose_candidate": loose_candidate,
        "no_loose_answer": no_loose_answer,
        "loose_bad_candidate": loose_bad_candidate,
        "loose_good_candidate": loose_good_candidate,
        "loose_recovered_bad": loose_recovered_bad,
        "loose_wrong_bad": loose_wrong_bad,
        "loose_preserved_good": loose_preserved_good,
        "loose_harmed_good": loose_harmed_good,
        "loose_bad_recovery_rate": div(loose_recovered_bad, loose_bad_candidate),
        "loose_good_harm_rate": div(loose_harmed_good, loose_good_candidate),
        "strict_net_gain": strict_recovered_bad - strict_harmed_good,
        "loose_net_gain": loose_recovered_bad - loose_harmed_good,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n| Metric | Value |")
    print("|---|---:|")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"| {k} | {v:.4f} |")
        else:
            print(f"| {k} | {v} |")


if __name__ == "__main__":
    main()
