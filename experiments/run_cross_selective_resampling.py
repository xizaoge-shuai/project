import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_local_rewrite_backtrack import build_generator
from experiments.run_selective_resampling import call_generator


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(fp, obj):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clean(x):
    s = str(x or "").strip().replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return s
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y


def ok(a, g):
    return clean(a) == clean(g)


def extract_answer(text):
    s = str(text)
    if "Final Answer:" in s:
        s = s.split("Final Answer:")[-1]
    return clean(s)


def majority_answer(answers):
    vals = [clean(a) for a in answers if clean(a) != ""]
    if not vals:
        return ""
    cnt = Counter(vals)
    return cnt.most_common(1)[0][0]


def build_prompt(question, dataset, sample_id, extra_idx):
    return f"""You are solving a grade-school math word problem from {dataset}.

Resampling context:
- sample_id: {sample_id}
- extra_attempt: {extra_idx}

Requirements:
1. Solve independently.
2. Write a concise step-by-step calculation.
3. Recheck the arithmetic.
4. The final line must be exactly:
Final Answer: <number>

Question:
{question}

Solution:
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--generator_config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--trigger", choices=["has_disagreement", "all_disagree"], default="has_disagreement")
    ap.add_argument("--n_extra", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--sampling_seed", type=int, default=42)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.trajectories)

    by = defaultdict(list)
    for r in rows:
        sid = r.get("sample_id") or r.get("id")
        by[sid].append(r)

    targets = []
    for sid, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda x: x.get("trajectory_id", x.get("id", "")))
        answers = [clean(r.get("final_answer", r.get("answer", ""))) for r in rs]
        nonempty = [a for a in answers if a != ""]
        uniq = set(nonempty)

        if args.trigger == "has_disagreement":
            flag = len(uniq) >= 2
        else:
            flag = len(uniq) >= 3

        if flag:
            targets.append((sid, rs, answers))

    if args.max_samples > 0:
        targets = targets[:args.max_samples]

    gen = build_generator(args.generator_config)

    out_rows = []
    base_correct = 0
    any_correct = 0
    extra_any_correct = 0

    for idx, (sid, rs, answers) in enumerate(targets):
        q = rs[0]["question"]
        gold = clean(rs[0].get("gold_answer", rs[0].get("answer", "")))
        cur = majority_answer(answers)

        extra_texts = []
        extra_answers = []
        extra_ok = []

        for j in range(args.n_extra):
            prompt = build_prompt(q, args.dataset, sid, j)
            text = call_generator(
                gen,
                prompt,
                args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.sampling_seed + idx * 1000 + j,
            )
            ans = extract_answer(text)

            extra_texts.append(text)
            extra_answers.append(ans)
            extra_ok.append(int(ok(ans, gold)))

        row = {
            "sample_id": sid,
            "dataset": args.dataset,
            "gold_answer": gold,
            "question": q,
            "orig_answers": answers,
            "current_best_answer": cur,
            "current_best_ok": int(ok(cur, gold)),
            "any_orig_ok": int(any(ok(a, gold) for a in answers)),
            "extra_answers": extra_answers,
            "extra_ok": extra_ok,
            "extra_texts": extra_texts,
        }

        base_correct += row["current_best_ok"]
        any_correct += row["any_orig_ok"]
        extra_any_correct += int(any(extra_ok))

        out_rows.append(row)

        if (idx + 1) % 20 == 0:
            print(f"[progress] {idx+1}/{len(targets)}")

    write_jsonl(args.out_jsonl, out_rows)

    summary = {
        "dataset": args.dataset,
        "trigger": args.trigger,
        "n_total_samples": len(by),
        "target_samples": len(targets),
        "completed_samples": len(out_rows),
        "n_extra": args.n_extra,
        "current_acc_on_targets": base_correct / max(1, len(out_rows)),
        "orig_any_acc_on_targets": any_correct / max(1, len(out_rows)),
        "extra_any_acc_on_targets": extra_any_correct / max(1, len(out_rows)),
        "out_jsonl": args.out_jsonl,
    }

    write_json(args.out_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
