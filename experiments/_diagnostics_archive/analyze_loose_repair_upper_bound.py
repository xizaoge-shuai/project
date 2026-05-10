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

    # 优先抽 Final Answer 后面的数字
    m = re.findall(r"Final Answer\s*:\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.I)
    if m:
        return m[-1]

    # 其次抽最后一个数字，作为 loose upper bound
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair_jsonl", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.repair_jsonl)

    triggered = 0
    strict_rewrite = 0
    strict_recovered = 0
    strict_harmed = 0

    loose_candidate = 0
    loose_correct = 0
    loose_extra_correct_over_strict = 0
    loose_harm_if_used = 0

    for r in rows:
        if not r.get("triggered"):
            continue

        triggered += 1
        gold = str(r.get("gold_answer", ""))

        original_ok = bool(r.get("original_is_correct", False))

        if r.get("repair_decision") == "REWRITE":
            strict_rewrite += 1
            if r.get("recovered"):
                strict_recovered += 1
            if r.get("harmed_good"):
                strict_harmed += 1

        out = r.get("repair_output", "")
        loose_ans = extract_loose_number(out)

        if not loose_ans:
            continue

        loose_candidate += 1
        loose_ok = bool(is_correct_prediction(loose_ans, gold, answer_mode="numeric"))

        if loose_ok:
            loose_correct += 1
            if r.get("repair_decision") != "REWRITE":
                loose_extra_correct_over_strict += 1

        if original_ok and not loose_ok:
            loose_harm_if_used += 1

    summary = {
        "file": args.repair_jsonl,
        "triggered": triggered,
        "strict_rewrite": strict_rewrite,
        "strict_recovered": strict_recovered,
        "strict_harmed": strict_harmed,
        "loose_candidate": loose_candidate,
        "loose_correct": loose_correct,
        "loose_extra_correct_over_strict": loose_extra_correct_over_strict,
        "loose_harm_if_used": loose_harm_if_used,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n| Metric | Value |")
    print("|---|---:|")
    for k, v in summary.items():
        print(f"| {k} | {v} |")


if __name__ == "__main__":
    main()
