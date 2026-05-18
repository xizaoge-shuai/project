import argparse
import json
import re
from pathlib import Path
from collections import Counter

from experiments.eval_math500_baseline import normalize, math_equal


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_noise_candidate(x):
    raw = str(x or "").strip()
    n = normalize(raw)
    low = raw.lower().replace(" ", "")

    if not n:
        return True

    bad_exact = {"\\[", "\\]", "\\boxed", ":", ".", ",", ";", "therefore", "thus"}
    if n in bad_exact or raw in bad_exact:
        return True

    # 太长的候选通常是没有抽干净的推理片段
    if len(n) > 100:
        return True

    bad_substrings = [
        "weneedto",
        "theequation",
        "theinequality",
        "thecondition",
        "summingall",
        "setting",
        "calculate",
        "problemstatement",
        "finalanswer",
        "istheanswer",
        "isthesolution",
        "coordinatesof",
        "therootsof",
        "probability)",
        "therefore,the",
        "thus,the",
    ]
    if any(b in low for b in bad_substrings):
        return True

    return False


def eval_one(rows, base_acc, n_samples, min_total, min_seed, min_margin, filter_bad):
    fixed = broken = changed = 0
    cur_correct = final_correct = 0
    detail_rows = []

    for r in rows:
        gold = r["gold_answer"]
        current = r["current_answer"]
        cur_norm = normalize(current)

        cur_ok = int(math_equal(current, gold))
        cur_correct += cur_ok

        extra_support = r.get("extra_support", {}) or {}
        extra_seed_support = r.get("extra_seed_support", {}) or {}

        candidates = []
        for cand, total in extra_support.items():
            if filter_bad and is_noise_candidate(cand):
                continue
            seed_sup = int(extra_seed_support.get(cand, 0))
            candidates.append((cand, int(total), seed_sup))

        if candidates:
            candidates = sorted(candidates, key=lambda x: (-x[1], -x[2], len(str(x[0]))))
            top_cand, top_total, top_seed = candidates[0]
            runner_total = candidates[1][1] if len(candidates) >= 2 else 0
        else:
            top_cand, top_total, top_seed, runner_total = "", 0, 0, 0

        margin = top_total - runner_total

        final = current
        reason = "keep_current"

        if (
            top_cand
            and normalize(top_cand) != cur_norm
            and top_total >= min_total
            and top_seed >= min_seed
            and margin >= min_margin
        ):
            final = top_cand
            reason = f"top_total{top_total}_seed{top_seed}_margin{margin}_filter{int(filter_bad)}"

        fin_ok = int(math_equal(final, gold))
        final_correct += fin_ok

        is_changed = int(normalize(final) != cur_norm)
        is_fixed = int(cur_ok == 0 and fin_ok == 1)
        is_broken = int(cur_ok == 1 and fin_ok == 0)

        changed += is_changed
        fixed += is_fixed
        broken += is_broken

        rr = dict(r)
        rr.update({
            "variant_final_answer": final,
            "variant_reason": reason,
            "variant_current_ok": cur_ok,
            "variant_final_ok": fin_ok,
            "variant_fixed": is_fixed,
            "variant_broken": is_broken,
            "variant_changed": is_changed,
            "variant_top_answer": top_cand,
            "variant_top_total": top_total,
            "variant_top_seed": top_seed,
            "variant_runner_total": runner_total,
            "variant_margin": margin,
        })
        detail_rows.append(rr)

    net = fixed - broken
    n_resampled = len(rows)

    return {
        "min_total": min_total,
        "min_seed": min_seed,
        "min_margin": min_margin,
        "filter_bad": int(filter_bad),
        "n_resampled": n_resampled,
        "current_acc_on_resampled": cur_correct / max(1, n_resampled),
        "final_acc_on_resampled": final_correct / max(1, n_resampled),
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "changed": changed,
        "estimated_global_acc": base_acc + net / n_samples,
        "estimated_global_gain": net / n_samples,
        "detail_rows": detail_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--base_acc", type=float, required=True)
    ap.add_argument("--n_samples", type=int, required=True)
    ap.add_argument("--out_summary_jsonl", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_best_jsonl", required=True)
    ap.add_argument("--out_safe_jsonl", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.input_jsonl)

    summaries = []
    all_results = []

    for filter_bad in [0, 1]:
        for min_total in range(2, 10):
            for min_seed in range(2, 9):
                for min_margin in range(0, 6):
                    res = eval_one(
                        rows=rows,
                        base_acc=args.base_acc,
                        n_samples=args.n_samples,
                        min_total=min_total,
                        min_seed=min_seed,
                        min_margin=min_margin,
                        filter_bad=bool(filter_bad),
                    )
                    all_results.append(res)
                    s = {k: v for k, v in res.items() if k != "detail_rows"}
                    summaries.append(s)

    # 按准确率优先，其次误伤少，其次 changed 少
    sorted_all = sorted(
        all_results,
        key=lambda x: (
            -x["estimated_global_acc"],
            x["broken"],
            x["changed"],
            -x["fixed"],
            x["min_total"],
            x["min_seed"],
            x["min_margin"],
            x["filter_bad"],
        ),
    )

    # safe: broken 最少优先，然后 accuracy 高
    sorted_safe = sorted(
        all_results,
        key=lambda x: (
            x["broken"],
            -x["estimated_global_acc"],
            x["changed"],
            -x["fixed"],
        ),
    )

    best = sorted_all[0]
    safe = sorted_safe[0]

    write_jsonl(args.out_summary_jsonl, summaries)
    write_jsonl(args.out_best_jsonl, best["detail_rows"])
    write_jsonl(args.out_safe_jsonl, safe["detail_rows"])

    lines = []
    lines.append("# MATH500 guard variant sweep")
    lines.append("")
    lines.append("## Top by estimated global accuracy")
    lines.append("")
    lines.append("| filter | total | seed | margin | acc | target_acc | fixed | broken | net | changed |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for x in sorted_all[:30]:
        lines.append(
            f"| {x['filter_bad']} | {x['min_total']} | {x['min_seed']} | {x['min_margin']} | "
            f"{x['estimated_global_acc']:.4f} | {x['final_acc_on_resampled']:.4f} | "
            f"{x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} |"
        )

    lines.append("")
    lines.append("## Low-harm / safe candidates")
    lines.append("")
    lines.append("| filter | total | seed | margin | acc | target_acc | fixed | broken | net | changed |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for x in sorted_safe[:30]:
        lines.append(
            f"| {x['filter_bad']} | {x['min_total']} | {x['min_seed']} | {x['min_margin']} | "
            f"{x['estimated_global_acc']:.4f} | {x['final_acc_on_resampled']:.4f} | "
            f"{x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} |"
        )

    lines.append("")
    lines.append("## Selected")
    lines.append("")
    lines.append(
        f"Best: filter={best['filter_bad']}, total={best['min_total']}, "
        f"seed={best['min_seed']}, margin={best['min_margin']}, "
        f"acc={best['estimated_global_acc']:.4f}, fixed={best['fixed']}, "
        f"broken={best['broken']}, net={best['net']}, changed={best['changed']}"
    )
    lines.append(
        f"Safe: filter={safe['filter_bad']}, total={safe['min_total']}, "
        f"seed={safe['min_seed']}, margin={safe['min_margin']}, "
        f"acc={safe['estimated_global_acc']:.4f}, fixed={safe['fixed']}, "
        f"broken={safe['broken']}, net={safe['net']}, changed={safe['changed']}"
    )

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
