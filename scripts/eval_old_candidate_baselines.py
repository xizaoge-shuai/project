#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path
from collections import Counter


def read_jsonl(fp):
    rows = []
    fp = Path(fp)
    if not fp.exists():
        print("[MISSING]", fp)
        return rows
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def first_nonempty(*xs):
    for x in xs:
        if x is None:
            continue
        if isinstance(x, str) and not x.strip():
            continue
        return x
    return None


def as_bool01(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
        return 0
    return None


def norm_choice(x):
    if x is None:
        return ""
    s = str(x).strip().lower()
    m = re.search(r"\(([abcde])\)", s)
    if m:
        return m.group(1)
    m = re.search(r"\b([abcde])\b", s)
    if m:
        return m.group(1)
    return s[:1]


def norm_yesno(x):
    if x is None:
        return ""
    s = str(x).strip().lower()
    if re.search(r"\byes\b", s):
        return "yes"
    if re.search(r"\bno\b", s):
        return "no"
    return s


def norm_numeric(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = s.replace(",", "").replace("$", "").replace("\\$", "")
    s = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"boxed\{([^{}]+)\}", r"\1", s)

    nums = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", s)
    if nums:
        s = nums[-1]

    if "/" in s:
        try:
            a, b = s.split("/", 1)
            v = float(a) / float(b)
            return f"{v:.10f}".rstrip("0").rstrip(".")
        except Exception:
            pass

    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.10f}".rstrip("0").rstrip(".")
    except Exception:
        return re.sub(r"\s+", "", str(x).strip().lower())


def norm_answer(x, task_type):
    if task_type == "choice":
        return norm_choice(x)
    if task_type == "yesno":
        return norm_yesno(x)
    return norm_numeric(x)


def norm_list(xs, task_type):
    if xs is None:
        return []
    if not isinstance(xs, list):
        return []
    return [norm_answer(x, task_type) for x in xs if norm_answer(x, task_type)]


def majority(ans):
    ans = [a for a in ans if a]
    if not ans:
        return ""
    cnt = Counter(ans)
    first = {}
    for i, a in enumerate(ans):
        if a not in first:
            first[a] = i
    return sorted(cnt.items(), key=lambda kv: (-kv[1], first[kv[0]]))[0][0]


def get_gold(r, task_type):
    return norm_answer(first_nonempty(
        r.get("gold_norm"),
        r.get("gold_answer"),
        r.get("answer"),
        r.get("label"),
    ), task_type)


def get_orig_answers(r, task_type):
    # 旧 Qwen7B 文件字段不统一，尽量都兼容
    for k in [
        "orig_answers_norm",
        "orig_answers",
        "base_answers_norm",
        "base_answers",
        "answers_norm",
        "answers",
    ]:
        vals = norm_list(r.get(k), task_type)
        if vals:
            return vals
    return []


def get_extra_answers(r, task_type):
    for k in [
        "extra_answers_norm",
        "extra_answers",
        "candidate_answers_norm",
        "candidate_answers",
        "candidates_norm",
        "candidates",
    ]:
        vals = norm_list(r.get(k), task_type)
        if vals:
            return vals
    return []


def get_base_pred(r, orig, task_type):
    # 核心修复：旧结果里的 base/current 应该优先用 current_answer
    val = first_nonempty(
        r.get("current_answer"),
        r.get("current_norm"),
        r.get("current_best_answer"),
        r.get("base_answer"),
        r.get("majority_answer"),
        r.get("pred_answer"),
        r.get("prediction"),
    )
    if val is not None:
        a = norm_answer(val, task_type)
        if a:
            return a
    return majority(orig)


def get_recorded_final_pred(r, base_pred, task_type):
    val = first_nonempty(
        r.get("final_answer"),
        r.get("final_guard_answer"),
        r.get("judge_final_answer"),
        r.get("confirm_answer"),
        r.get("seedaware_answer"),
        r.get("orig_majority_guard_answer"),
        r.get("weighted_answer"),
    )
    if val is not None:
        a = norm_answer(val, task_type)
        if a:
            return a
    return base_pred


def esc_predict(cands, k, w):
    used = cands[:min(k, len(cands))]
    if not used:
        return "", 0
    for i in range(w, len(used) + 1):
        win = used[i-w:i]
        if len(win) == w and len(set(win)) == 1 and win[0]:
            return win[0], i
    return majority(used), len(used)


def recorded_extra_used(r, orig, extra):
    # runner_total 在旧文件里通常表示额外 runner / extra 次数
    for k in ["runner_total", "n_extra", "extra_per_target", "extra_used"]:
        if k in r and r[k] is not None:
            try:
                v = int(float(r[k]))
                return max(0, v)
            except Exception:
                pass
    if extra:
        return len(extra)
    if orig:
        return max(0, len(orig) - 3)
    return 0


def evaluate_method(rows, task_type, n_samples, base_acc, method_name, method_fn):
    changed = fixed = broken = 0
    total_extra = 0
    n_eval = 0
    missing_gold = 0

    for r in rows:
        gold = get_gold(r, task_type)
        if not gold:
            missing_gold += 1
            continue

        orig = get_orig_answers(r, task_type)
        extra = get_extra_answers(r, task_type)
        all_cands = orig + extra

        base_pred = get_base_pred(r, orig, task_type)
        if not base_pred:
            base_pred = majority(orig)

        current_ok_field = as_bool01(r.get("current_ok"))
        base_ok = current_ok_field if current_ok_field is not None else int(base_pred == gold)

        if method_name == "Recorded-Ours":
            # 核心修复：如果旧文件已保存 current_ok/final_ok/fixed/broken/changed，直接用它
            final_ok_field = as_bool01(r.get("final_ok"))
            fixed_field = as_bool01(r.get("fixed"))
            broken_field = as_bool01(r.get("broken"))
            changed_field = as_bool01(r.get("changed"))

            if final_ok_field is not None:
                final_ok = final_ok_field
                pred = get_recorded_final_pred(r, base_pred, task_type)
                if changed_field is not None:
                    is_changed = changed_field
                else:
                    is_changed = int(pred != base_pred)

                if fixed_field is not None:
                    is_fixed = fixed_field
                else:
                    is_fixed = int(base_ok == 0 and final_ok == 1)

                if broken_field is not None:
                    is_broken = broken_field
                else:
                    is_broken = int(base_ok == 1 and final_ok == 0)

                extra_used = recorded_extra_used(r, orig, extra)
            else:
                pred, used = method_fn(r, orig, extra, base_pred, gold)
                pred = pred or base_pred
                final_ok = int(pred == gold)
                is_changed = int(pred != base_pred)
                is_fixed = int(base_ok == 0 and final_ok == 1)
                is_broken = int(base_ok == 1 and final_ok == 0)
                extra_used = max(0, used - 3)

        else:
            pred, used = method_fn(r, orig, extra, base_pred, gold)
            pred = pred or base_pred
            final_ok = int(pred == gold)
            is_changed = int(pred != base_pred)
            is_fixed = int(base_ok == 0 and final_ok == 1)
            is_broken = int(base_ok == 1 and final_ok == 0)
            extra_used = max(0, used - 3)

        changed += int(is_changed)
        fixed += int(is_fixed)
        broken += int(is_broken)
        total_extra += int(extra_used)
        n_eval += 1

    net = fixed - broken
    final_acc = float(base_acc) + net / float(n_samples)
    gain = final_acc - float(base_acc)

    if final_acc > 1.000001 or final_acc < -0.000001:
        print(f"[WARN] suspicious final_acc={final_acc:.6f} for method={method_name}; check fields.")

    precision_changed = fixed / max(changed, 1)
    repair_precision = fixed / max(fixed + broken, 1)
    harm = broken / max(changed, 1)

    return {
        "method": method_name,
        "base_acc": base_acc,
        "final_acc": final_acc,
        "gain": gain,
        "n_eval": n_eval,
        "missing_gold": missing_gold,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "extra_calls": total_extra,
        "extra_per_target": total_extra / max(n_eval, 1),
        "extra_per_sample": total_extra / max(n_samples, 1),
        "precision_changed": precision_changed,
        "repair_precision": repair_precision,
        "harm_rate": harm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction_file", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task_type", choices=["numeric", "choice", "yesno"], default="numeric")
    ap.add_argument("--n_samples", type=int, required=True)
    ap.add_argument("--base_acc", type=float, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.prediction_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = []

    def add(name, fn):
        methods.append((name, fn))

    add("Base-current", lambda r, orig, extra, base, gold: (base, 3))
    add("CoT@1", lambda r, orig, extra, base, gold: ((orig[0] if orig else base), 1))
    add("SC@3", lambda r, orig, extra, base, gold: (majority(orig[:3]), min(3, len(orig))))

    for k in [4, 8, 12]:
        add(f"SC@{k}", lambda r, orig, extra, base, gold, k=k: (majority((orig + extra)[:k]), min(k, len(orig + extra))))

    for k in [4, 8, 12]:
        add(f"CISC_support_proxy@{k}", lambda r, orig, extra, base, gold, k=k: (majority((orig + extra)[:k]), min(k, len(orig + extra))))

    for k in [4, 8, 12]:
        for w in [2, 3, 4]:
            add(f"ESC@{k}_w{w}", lambda r, orig, extra, base, gold, k=k, w=w: esc_predict((orig + extra), k, w))

    add("Recorded-Ours", lambda r, orig, extra, base, gold: (get_recorded_final_pred(r, base, args.task_type), 3 + recorded_extra_used(r, orig, extra)))

    results = []
    for name, fn in methods:
        d = evaluate_method(rows, args.task_type, args.n_samples, args.base_acc, name, fn)
        d["dataset"] = args.dataset
        d["prediction_file"] = args.prediction_file
        results.append(d)

    results.sort(key=lambda x: (-x["final_acc"], x["extra_per_sample"], x["harm_rate"]))

    csv_fp = out_dir / f"{args.prefix}_old_candidate_baselines.csv"
    md_fp = out_dir / f"{args.prefix}_old_candidate_baselines.md"

    fields = [
        "dataset", "method", "base_acc", "final_acc", "gain", "n_eval", "missing_gold",
        "changed", "fixed", "broken", "net",
        "extra_calls", "extra_per_target", "extra_per_sample",
        "precision_changed", "repair_precision", "harm_rate", "prediction_file"
    ]

    with csv_fp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})

    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    lines = []
    lines.append(f"# Old-format Candidate Baselines: {args.prefix}")
    lines.append("")
    lines.append("| Method | Base Acc | Final Acc | ΔAcc | n_eval | Changed | Fixed | Broken | Net | Extra/Target | Extra/Sample | Precision | Repair-P | Harm |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['method']} | {fmt(r['base_acc'])} | {fmt(r['final_acc'])} | {fmt(r['gain'])} | "
            f"{r['n_eval']} | {r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | "
            f"{fmt(r['extra_per_target'])} | {fmt(r['extra_per_sample'])} | "
            f"{fmt(r['precision_changed'])} | {fmt(r['repair_precision'])} | {fmt(r['harm_rate'])} |"
        )

    md_fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("saved:", csv_fp)
    print("saved:", md_fp)
    print(md_fp.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
