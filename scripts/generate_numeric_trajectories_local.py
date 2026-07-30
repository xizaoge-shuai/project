import argparse
import json
import re
from pathlib import Path

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


def clean_number(x):
    s = str(x or "").strip().replace(",", "").replace("$", "")
    if "####" in s:
        s = s.split("####")[-1]
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return s
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y


def extract_answer(text):
    s = str(text)
    if "Final Answer:" in s:
        s = s.split("Final Answer:")[-1]
    return clean_number(s)


def build_prompt(question, variant=0):
    styles = [
        "Solve step by step and check the arithmetic.",
        "List the given quantities, compute carefully, and verify the final number.",
        "Use equations when helpful, then recheck the calculation.",
    ]
    style = styles[variant % len(styles)]
    return f"""You are solving a grade-school math word problem.

Instruction:
{style}

The final line must be exactly:
Final Answer: <number>

Question:
{question}

Solution:
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--generator_config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_traj", type=int, default=3)
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    gen = build_generator(args.generator_config)

    out = []
    total = len(rows) * args.n_traj
    done = 0

    for i, r in enumerate(rows):
        sid = r.get("sample_id") or r.get("id") or f"{args.dataset}_test_{i}"
        q = r["question"]
        gold = clean_number(r.get("gold_answer", r.get("answer", "")))
        context = r.get("context", "")

        for j in range(args.n_traj):
            prompt = build_prompt(q, variant=j)
            text = call_generator(
                gen,
                prompt,
                args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + i * 100 + j,
            )
            ans = extract_answer(text)
            tid = f"{sid}_traj_{j}"

            out.append({
                "id": tid,
                "sample_id": sid,
                "trajectory_id": tid,
                "dataset": args.dataset,
                "split": "test",
                "task": args.dataset,
                "question": q,
                "context": context,
                "gold_answer": gold,
                "final_answer": ans,
                "answer": ans,
                "trajectory": text,
                "text": text,
                "reasoning": text,
                "meta": {
                    "source_id": sid,
                    "traj_index": j,
                }
            })

            done += 1
            if done % 20 == 0:
                print(f"[progress] {done}/{total}")

    write_jsonl(args.output, out)
    print("[SAVED]", args.output, "rows=", len(out))


if __name__ == "__main__":
    main()
