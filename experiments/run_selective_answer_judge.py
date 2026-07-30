from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_utils import is_correct_prediction
from experiments.eval_cross_pce_weighted_selection import clean as cross_clean, ok as cross_ok
from experiments.run_local_rewrite_backtrack import build_generator


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return [json.loads(x) for x in p.open("r", encoding="utf-8") if x.strip()]


def write_jsonl_append(path: str, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_num(x: Any) -> str:
    x = str(x or "").replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", x)
    if not nums:
        return str(x).strip()
    v = nums[-1]
    if "." in v:
        v = v.rstrip("0").rstrip(".")
    return v


def extract_answer_from_steps(steps: List[str]) -> str:
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



def cross_steps(row: Dict[str, Any]) -> List[str]:
    """Return reasoning steps for both legacy and cross-dataset schemas."""
    steps = row.get("steps")
    if isinstance(steps, list) and steps:
        return [str(x) for x in steps]

    text = (
        row.get("trajectory")
        or row.get("text")
        or row.get("response")
        or row.get("generated_text")
        or row.get("reasoning")
        or ""
    )
    return [str(text)] if str(text).strip() else []


def cross_final_answer(row: Dict[str, Any]) -> str:
    """Read a final answer without assuming the legacy steps schema."""
    direct = row.get("final_answer")
    if direct is None or not str(direct).strip():
        direct = row.get("answer")

    if direct is not None and str(direct).strip():
        return cross_clean(direct)

    return cross_clean(extract_answer_from_steps(cross_steps(row)))

def traj_order(tid: str) -> int:
    m = re.search(r"_traj_(\d+)(?:_seed\d+)?$", str(tid))
    return int(m.group(1)) if m else 999999


def score_tail5(scores_by_tid: Dict[str, List[Tuple[int, float]]], tid: str) -> float:
    vals = [p for _, p in sorted(scores_by_tid.get(tid, []), key=lambda x: x[0])]
    if not vals:
        return 1.0
    return sum(vals[-5:]) / min(5, len(vals))


def weighted_vote(rs: List[Dict[str, Any]]) -> Tuple[str, float, float, List[Tuple[str, float]]]:
    w = defaultdict(float)
    for x in rs:
        w[x["answer"]] += float(x["score"])
    ranked = sorted(w.items(), key=lambda z: z[1], reverse=True)
    top_ans, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(w.values()) if sum(w.values()) > 0 else 1.0
    margin_abs = top_score - second_score
    margin_norm = margin_abs / total
    return top_ans, margin_abs, margin_norm, ranked


def should_flag(rs: List[Dict[str, Any]], trigger: str, margin_threshold: float) -> bool:
    answers = [x["answer"] for x in rs]
    uniq = len(set(answers))
    weighted_ans, _, margin_norm, _ = weighted_vote(rs)
    majority_ans = Counter(answers).most_common(1)[0][0]

    if trigger == "all_disagree":
        return uniq == len(answers)
    if trigger == "has_disagreement":
        return uniq >= 2
    if trigger == "majority_diff_weighted":
        return majority_ans != weighted_ans
    if trigger == "margin":
        return margin_norm <= margin_threshold
    if trigger == "all_disagree_or_margin":
        return (uniq == len(answers)) or (margin_norm <= margin_threshold)

    raise ValueError(f"Unknown trigger: {trigger}")


def build_judge_prompt(
    question: str,
    candidates: List[Dict[str, Any]],
    ranked: List[Tuple[str, float]],
    use_confidence: bool = False,
) -> str:
    lines = []
    lines.append("You are a strict math answer arbitrator.")
    lines.append("Your task is NOT to solve the problem from scratch.")
    lines.append("Choose the most reliable final answer only from the given candidate answers.")
    lines.append("Do not invent a new answer.")
    lines.append("Do not choose a candidate only because it has higher confidence.")
    lines.append("Check whether each candidate's reasoning is consistent with the question and arithmetic.")
    lines.append("If a candidate answer is empty or malformed, treat it as unreliable.")
    lines.append("")
    lines.append("Question:")
    lines.append(str(question).strip())
    lines.append("")
    lines.append("Candidate answers and reasoning traces:")

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for i, c in enumerate(candidates):
        label = labels[i]
        steps = c.get("steps", []) or []
        short_steps = steps[-8:]
        reasoning = "\n".join(str(x) for x in short_steps)

        lines.append("")
        lines.append(f"Candidate {label}:")
        lines.append(f"Final answer: {c['answer']}")
        if use_confidence:
            lines.append(f"Auxiliary PCE confidence, tie-breaker only: {c['score']:.4f}")
        lines.append("Reasoning trace:")
        lines.append(reasoning.strip() if reasoning.strip() else "(empty)")

    if use_confidence:
        lines.append("")
        lines.append("Auxiliary weighted vote ranking, tie-breaker only:")
        for ans, sc in ranked:
            lines.append(f"- answer={ans}, weighted_score={sc:.4f}")

    lines.append("")
    lines.append("Return exactly in this format with no extra explanation:")
    lines.append("Chosen Candidate: <A/B/C>")
    lines.append("Final Answer: <copy the chosen candidate's final answer>")
    lines.append("Reason Type: <arithmetic_error | missing_condition | unit_conversion | inconsistent_reasoning | majority_supported | confidence_supported | unknown>")

    return "\n".join(lines)


def normalize_generation_output(out: Any) -> str:
    if out is None:
        return ""

    if isinstance(out, str):
        return out

    if isinstance(out, dict):
        for k in ["text", "output", "response", "generated_text"]:
            if k in out:
                return str(out[k])
        return str(out)

    if isinstance(out, list):
        if not out:
            return ""
        return normalize_generation_output(out[0])

    if isinstance(out, tuple):
        if not out:
            return ""
        return normalize_generation_output(out[0])

    if hasattr(out, "outputs"):
        outs = getattr(out, "outputs")
        if outs:
            first = outs[0]
            if hasattr(first, "text"):
                return str(first.text)
            return str(first)

    if hasattr(out, "text"):
        return str(getattr(out, "text"))

    return str(out)


def try_call(fn: Any, prompt: str, max_new_tokens: int) -> str:
    attempts = [
        lambda: fn(prompt=prompt, max_new_tokens=max_new_tokens),
        lambda: fn(prompt=prompt),
        lambda: fn(prompt, max_new_tokens=max_new_tokens),
        lambda: fn(prompt),
    ]

    last_err = None
    for call in attempts:
        try:
            return normalize_generation_output(call())
        except TypeError as e:
            last_err = e
            continue

    raise last_err if last_err is not None else RuntimeError("No valid generator call signature.")


def call_generator(generator: Any, prompt: str, max_new_tokens: int) -> str:
    for name in ["generate_one", "generate_text", "complete", "generate", "chat"]:
        if hasattr(generator, name):
            return try_call(getattr(generator, name), prompt, max_new_tokens)

    if callable(generator):
        return try_call(generator, prompt, max_new_tokens)

    for name in ["generate_many", "generate_batch", "batch_generate", "generate_many"]:
        if hasattr(generator, name):
            fn = getattr(generator, name)
            attempts = [
                lambda: fn(prompts=[prompt], max_new_tokens=max_new_tokens),
                lambda: fn(prompts=[prompt]),
                lambda: fn([prompt], max_new_tokens=max_new_tokens),
                lambda: fn([prompt]),
            ]
            last_err = None
            for call in attempts:
                try:
                    return normalize_generation_output(call())
                except TypeError as e:
                    last_err = e
                    continue
            raise last_err if last_err is not None else RuntimeError("No valid batch generator signature.")

    raise AttributeError(
        "Unsupported generator object. Available public methods: "
        + str([x for x in dir(generator) if not x.startswith("_")])
    )


def parse_judge_output(text: str) -> Dict[str, str]:
    chosen = ""
    final = ""
    reason = ""

    m = re.search(r"Chosen Candidate\s*:\s*([A-Z])", text, flags=re.I)
    if m:
        chosen = m.group(1).upper()

    m = re.search(r"Final Answer\s*:\s*([^\n\r]+)", text, flags=re.I)
    if m:
        final = m.group(1).strip()

    m = re.search(r"Reason Type\s*:\s*([^\n\r]+)", text, flags=re.I)
    if m:
        reason = m.group(1).strip()

    return {
        "chosen_candidate": chosen,
        "parsed_final_answer": final,
        "reason_type": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--repair_jsonl", required=True)
    parser.add_argument("--generator_config", required=True)
    parser.add_argument("--dataset", default="gsm8k", choices=[
            "gsm8k",
            "strategyqa",
            "hotpotqa",
            "svamp",
            "asdiv",
            "math500",
            "mathqa"
        ])
    parser.add_argument("--trigger", default="all_disagree", choices=[
        "all_disagree",
        "has_disagreement",
        "majority_diff_weighted",
        "margin",
        "all_disagree_or_margin",
    ])
    parser.add_argument("--margin_threshold", type=float, default=0.08)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_cases", type=int, default=-1)
    parser.add_argument("--accept_policy", default="final_in_candidates", choices=["final_in_candidates", "chosen_map", "agree_only", "any_number"])
    parser.add_argument("--use_confidence_in_prompt", type=int, default=0)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    answer_mode = "numeric" if args.dataset == "gsm8k" else ("yesno" if args.dataset == "strategyqa" else "span")

    pred_rows = read_jsonl(args.predictions)
    traj_rows = read_jsonl(args.trajectories)
    repair_rows = read_jsonl(args.repair_jsonl)

    repair_by_tid = {r["trajectory_id"]: r for r in repair_rows}

    scores_by_tid = defaultdict(list)
    for r in pred_rows:
        if "success_prob" not in r:
            continue
        scores_by_tid[r["trajectory_id"]].append(
            (int(r.get("prefix_num_units", 0)), float(r["success_prob"]))
        )

    by_sample = defaultdict(list)

    for tr in traj_rows:
        tid = tr["trajectory_id"]
        sid = tr["sample_id"]
        gold = cross_clean(tr.get("gold_answer", tr.get("answer", "")))

        before = cross_final_answer(tr)
        after = before

        rr = repair_by_tid.get(tid)
        if rr and rr.get("repair_decision") == "REWRITE" and str(rr.get("repaired_final_answer", "")).strip():
            after = cross_clean(rr["repaired_final_answer"])

        by_sample[sid].append({
            "sample_id": sid,
            "tid": tid,
            "question": tr.get("question", ""),
            "gold": gold,
            "answer": after,
            "score": score_tail5(scores_by_tid, tid),
            "steps": cross_steps(tr),
            "ok": int(cross_ok(after, gold)),
        })

    existing = {}
    out_path = Path(args.out_jsonl)
    if out_path.exists():
        for r in read_jsonl(str(out_path)):
            existing[r["sample_id"]] = r

    samples = []
    for sid, rs in by_sample.items():
        rs = sorted(rs, key=lambda x: traj_order(x["tid"]))
        if should_flag(rs, args.trigger, args.margin_threshold):
            samples.append((sid, rs))

    samples = sorted(samples, key=lambda x: x[0])

    if args.max_cases > 0:
        samples = samples[:args.max_cases]

    print(f"flagged samples: {len(samples)}")
    print(f"already judged: {len(existing)}")

    generator = build_generator(args.generator_config)

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    new_rows = []

    for idx, (sid, rs) in enumerate(samples, 1):
        if sid in existing:
            continue

        gold = rs[0]["gold"]
        weighted_ans, margin_abs, margin_norm, ranked = weighted_vote(rs)
        weighted_ok = int(cross_ok(weighted_ans, gold))
        any_ok = int(any(x["ok"] for x in rs))

        prompt = build_judge_prompt(
            rs[0].get("question", ""),
            rs,
            ranked,
            use_confidence=bool(args.use_confidence_in_prompt),
        )

        output = call_generator(generator, prompt, args.max_new_tokens)
        parsed = parse_judge_output(output)

        label_to_answer = {labels[i]: x["answer"] for i, x in enumerate(rs)}
        candidate_answers = {x["answer"] for x in rs}

        chosen_label = parsed["chosen_candidate"]
        parsed_final_ans = cross_clean(parsed["parsed_final_answer"])

        chosen_ans = label_to_answer.get(chosen_label, "")

        if args.accept_policy == "final_in_candidates":
            # Main clean policy: trust Final Answer only if it is one of candidate answers.
            # This treats the judge as an answer arbitrator, not as a label classifier.
            if parsed_final_ans in candidate_answers:
                judge_ans = parsed_final_ans
                accepted = True
            elif chosen_ans in candidate_answers and chosen_ans:
                judge_ans = chosen_ans
                accepted = True
            else:
                judge_ans = ""
                accepted = False

        elif args.accept_policy == "chosen_map":
            # Strict label-mapping policy: ignore free-form Final Answer if label exists.
            if chosen_ans in candidate_answers and chosen_ans:
                judge_ans = chosen_ans
                accepted = True
            elif parsed_final_ans in candidate_answers:
                judge_ans = parsed_final_ans
                accepted = True
            else:
                judge_ans = ""
                accepted = False

        elif args.accept_policy == "agree_only":
            # Conservative policy: accept only if label answer agrees with parsed Final Answer.
            if chosen_ans and parsed_final_ans and chosen_ans == parsed_final_ans and chosen_ans in candidate_answers:
                judge_ans = chosen_ans
                accepted = True
            else:
                judge_ans = ""
                accepted = False

        else:
            judge_ans = parsed_final_ans
            accepted = bool(judge_ans)

        final_ans = judge_ans if accepted else weighted_ans
        final_ok = int(cross_ok(final_ans, gold))

        row = {
            "sample_id": sid,
            "gold_answer": gold,
            "trigger": args.trigger,
            "margin_threshold": args.margin_threshold,
            "answers": [x["answer"] for x in rs],
            "scores": [x["score"] for x in rs],
            "weighted_answer": weighted_ans,
            "weighted_ok": weighted_ok,
            "any_ok": any_ok,
            "margin_abs": margin_abs,
            "margin_norm": margin_norm,
            "judge_output": output,
            "chosen_candidate": chosen_label,
            "parsed_final_answer": parsed_final_ans,
            "judge_answer": judge_ans,
            "reason_type": parsed["reason_type"],
            "accepted": accepted,
            "final_answer": final_ans,
            "final_ok": final_ok,
        }

        new_rows.append(row)

        if len(new_rows) >= 5:
            write_jsonl_append(args.out_jsonl, new_rows)
            new_rows = []

        if idx % 10 == 0:
            print(f"[progress] {idx}/{len(samples)}", flush=True)

    if new_rows:
        write_jsonl_append(args.out_jsonl, new_rows)

    all_judged = {}
    if Path(args.out_jsonl).exists():
        for r in read_jsonl(args.out_jsonl):
            all_judged[r["sample_id"]] = r

    base_correct = 0
    final_correct = 0
    any_correct = 0
    flagged_count = 0
    accepted_count = 0
    fixed = 0
    broken = 0
    reason_counter = Counter()

    for sid, rs in by_sample.items():
        rs = sorted(rs, key=lambda x: traj_order(x["tid"]))
        gold = rs[0]["gold"]

        weighted_ans, _, _, _ = weighted_vote(rs)
        weighted_ok = int(cross_ok(weighted_ans, gold))

        base_correct += weighted_ok
        any_correct += int(any(x["ok"] for x in rs))

        if sid in all_judged:
            jr = all_judged[sid]
            flagged_count += 1
            accepted_count += int(bool(jr.get("accepted")))
            final_ok = int(jr.get("final_ok", 0))
            reason_counter[str(jr.get("reason_type", ""))] += 1
        else:
            final_ok = weighted_ok

        final_correct += final_ok

        if weighted_ok == 0 and final_ok == 1:
            fixed += 1
        if weighted_ok == 1 and final_ok == 0:
            broken += 1

    n = len(by_sample)
    summary = {
        "dataset": args.dataset,
        "trigger": args.trigger,
        "margin_threshold": args.margin_threshold,
        "accept_policy": args.accept_policy,
        "use_confidence_in_prompt": int(args.use_confidence_in_prompt),
        "n_samples": n,
        "flagged_total": len(samples),
        "judged_total": len(all_judged),
        "accepted_count": accepted_count,
        "base_weighted_tail5_acc": base_correct / n,
        "judge_final_acc": final_correct / n,
        "oracle_any_acc": any_correct / n,
        "fixed_count": fixed,
        "broken_count": broken,
        "net_gain_count": fixed - broken,
        "reason_counter": dict(reason_counter),
        "out_jsonl": args.out_jsonl,
    }

    write_json(args.out_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
