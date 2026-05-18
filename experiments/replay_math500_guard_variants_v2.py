import argparse
import json
import re
from pathlib import Path

from experiments.eval_math500_baseline import normalize, math_equal


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strip_percent_unit(x):
    s = normalize(x)
    s = s.replace("\\%", "").replace("%", "")
    return s


def is_percent_equiv(a, b):
    sa = strip_percent_unit(a)
    sb = strip_percent_unit(b)
    return sa and sb and sa == sb


def is_noise_candidate(x):
    raw = str(x or "").strip()
    n = normalize(raw)
    low = raw.lower().replace(" ", "")

    if not n:
        return True

    if n in {"\\[", "\\]", "\\boxed", ":", ".", ",", ";"}:
        return True

    # 过长候选基本是没抽干净的推理片段
    if len(n) > 80:
        return True

    bad_substrings = [
        "weneedto",
        "theequation",
        "theinequality",
        "thecondition",
        "theareaof",
        "thevalueof",
        "theroots",
        "thevalidroots",
        "theremainingroots",
        "thefinal",
        "therefore",
        "thus",
        "setting",
        "squaringbothsides",
        "calculate",
        "convertradian",
        "solvefor",
        "probability)",
        "coordinatesof",
        "graphof",
        "possiblevalue",
        "summingall",
        "problemstatement",
        "isacircle",
        "isanellipse",
    ]
    if any(b in low for b in bad_substrings):
        return True

    # 只有句子字母、几乎没有数学符号或数字，也像推理片段
    has_digit = any(ch.isdigit() for ch in n)
    has_math = ("\\" in n) or any(ch in n for ch in "+-*/=(){}^")
    if len(n) > 20 and not has_digit and not has_math:
        return True

    return False


def eval_one(rows, base_acc, n_samples, min_total, min_seed, min_margin, filter_bad, percent_guard, current_support_guard):
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
        current_extra_support = 0
        current_extra_seed = 0

        for cand, total in extra_support.items():
            if normalize(cand) == cur_norm or is_percent_equiv(cand, current):
                current_extra_support = max(current_extra_support, int(total))
                current_extra_seed = max(current_extra_seed, int(extra_seed_support.get(cand, 0)))

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

        should_change = (
            top_cand
            and normalize(top_cand) != cur_norm
            and top_total >= min_total
            and top_seed >= min_seed
            and margin >= min_margin
        )

        # 百分号格式 guard：10 vs 10\% 不改
        if should_change and percent_guard and is_percent_equiv(top_cand, current):
            should_change = False
            reason = "keep_current_percent_equiv"

        # current 也被 extra 支持且支持不弱时，不轻易改
        if should_change and current_support_guard:
            if current_extra_support >= top_total:
                should_change = False
                reason = f"keep_current_support_ge_top{current_extra_support}_{top_total}"

        if should_change:
            final = top_cand
            reason = f"top_total{top_total}_seed{top_seed}_margin{margin}_filter{int(filter_bad)}_pct{int(percent_guard)}_curguard{int(current_support_guard)}"

        fin_ok = int(math_equal(final, gold))
        final_correct += fin_ok

        is_changed = int(normalize(final) != cur_norm and not is_percent_equiv(final, current))
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
            "variant_current_extra_support": current_extra_support,
            "variant_current_extra_seed": current_extra_seed,
        })
        detail_rows.append(rr)

    net = fixed - broken
    n_resampled = len(rows)

    return {
        "filter_bad": int(filter_bad),
        "percent_guard": int(percent_guard),
        "current_support_guard": int(current_support_guard),
        "min_total": min_total,
        "min_seed": min_seed,
        "min_margin": min_margin,
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
    ap.add_argument("--out_balanced_jsonl", required=True)
    ap.add_argument("--out_safe_jsonl", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.input_jsonl)

    all_results = []
    summaries = []

    for filter_bad in [0, 1]:
        for percent_guard in [0, 1]:
            for current_support_guard in [0, 1]:
                for min_total in range(2, 8):
                    for min_seed in range(2, 7):
                        for min_margin in range(0, 6):
                            res = eval_one(
                                rows=rows,
                                base_acc=args.base_acc,
                                n_samples=args.n_samples,
                                min_total=min_total,
                                min_seed=min_seed,
                                min_margin=min_margin,
                                filter_bad=filter_bad,
                                percent_guard=percent_guard,
                                current_support_guard=current_support_guard,
                            )
                            all_results.append(res)
                            summaries.append({k: v for k, v in res.items() if k != "detail_rows"})

    top = sorted(
        all_results,
        key=lambda x: (-x["estimated_global_acc"], x["broken"], x["changed"], -x["fixed"])
    )

    # balanced：允许少量 broken，但希望 acc 高
    balanced = sorted(
        all_results,
        key=lambda x: (x["broken"] > 3, -x["estimated_global_acc"], x["broken"], x["changed"])
    )

    safe = sorted(
        all_results,
        key=lambda x: (x["broken"], -x["estimated_global_acc"], x["changed"])
    )

    best = top[0]
    bal = balanced[0]
    sf = safe[0]

    write_jsonl(args.out_summary_jsonl, summaries)
    write_jsonl(args.out_best_jsonl, best["detail_rows"])
    write_jsonl(args.out_balanced_jsonl, bal["detail_rows"])
    write_jsonl(args.out_safe_jsonl, sf["detail_rows"])

    lines = []
    lines.append("# MATH500 guard variant sweep v2")
    lines.append("")
    lines.append("## Top by estimated global accuracy")
    lines.append("")
    lines.append("| filter | pct | curguard | total | seed | margin | acc | target_acc | fixed | broken | net | changed |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for x in top[:40]:
        lines.append(
            f"| {x['filter_bad']} | {x['percent_guard']} | {x['current_support_guard']} | "
            f"{x['min_total']} | {x['min_seed']} | {x['min_margin']} | "
            f"{x['estimated_global_acc']:.4f} | {x['final_acc_on_resampled']:.4f} | "
            f"{x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} |"
        )

    lines.append("")
    lines.append("## Balanced candidates")
    lines.append("")
    lines.append("| filter | pct | curguard | total | seed | margin | acc | target_acc | fixed | broken | net | changed |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for x in balanced[:30]:
        lines.append(
            f"| {x['filter_bad']} | {x['percent_guard']} | {x['current_support_guard']} | "
            f"{x['min_total']} | {x['min_seed']} | {x['min_margin']} | "
            f"{x['estimated_global_acc']:.4f} | {x['final_acc_on_resampled']:.4f} | "
            f"{x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} |"
        )

    lines.append("")
    lines.append("## Safe candidates")
    lines.append("")
    lines.append("| filter | pct | curguard | total | seed | margin | acc | target_acc | fixed | broken | net | changed |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for x in safe[:30]:
        lines.append(
            f"| {x['filter_bad']} | {x['percent_guard']} | {x['current_support_guard']} | "
            f"{x['min_total']} | {x['min_seed']} | {x['min_margin']} | "
            f"{x['estimated_global_acc']:.4f} | {x['final_acc_on_resampled']:.4f} | "
            f"{x['fixed']} | {x['broken']} | {x['net']} | {x['changed']} |"
        )

    lines.append("")
    lines.append("## Selected")
    lines.append("")
    for label, x in [("Best", best), ("Balanced", bal), ("Safe", sf)]:
        lines.append(
            f"{label}: filter={x['filter_bad']}, pct={x['percent_guard']}, "
            f"curguard={x['current_support_guard']}, total={x['min_total']}, "
            f"seed={x['min_seed']}, margin={x['min_margin']}, "
            f"acc={x['estimated_global_acc']:.4f}, fixed={x['fixed']}, "
            f"broken={x['broken']}, net={x['net']}, changed={x['changed']}"
        )

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
