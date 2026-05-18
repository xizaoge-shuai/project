import json
import re
from pathlib import Path


OUT_DIR = Path("outputs/logs/final_summaries")
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR = Path("outputs/metrics/final_ablation")
METRIC_DIR.mkdir(parents=True, exist_ok=True)


# 这里用你当前最终表里的 GSM8K 三个主要设置
# n_eval 是触发/重采样样本数，不是 full 1319。
KNOWN = [
    {
        "setting": "margin030_currentkeep2",
        "base_acc": 0.8886,
        "final_acc": 0.9196,
        "n_eval": 107,
        "fixed": 13,
        "broken": 1,
        "net": 12,
        "changed": 18,
        "tokens": ["margin030", "currentkeep"],
    },
    {
        "setting": "margin040_currentkeep2",
        "base_acc": 0.8886,
        "final_acc": 0.9204,
        "n_eval": 210,
        "fixed": 17,
        "broken": 4,
        "net": 13,
        "changed": 28,
        "tokens": ["margin040", "currentkeep"],
    },
    {
        "setting": "margin040_selective_origmaj_top7",
        "base_acc": 0.8886,
        "final_acc": 0.9212,
        "n_eval": 210,
        "fixed": 15,
        "broken": 1,
        "net": 14,
        "changed": 22,
        "tokens": ["margin040", "selective", "origmaj", "top7"],
    },
]


def div(a, b):
    return a / b if b else 0.0


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def read_jsonl(fp):
    rows = []
    try:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception:
        return []
    return rows


def get_bool(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            try:
                return int(row[k])
            except Exception:
                pass
    return None


def compute_from_rows(rows):
    """
    从 per-case jsonl 里计算：
    current_wrong, fixed, broken, changed。
    兼容字段：
    cur_ok/current_ok/majority_ok
    pred_ok/final_ok
    fixed/broken/changed
    """
    if not rows:
        return None

    n = len(rows)
    cur_vals = []
    final_vals = []
    fixed_vals = []
    broken_vals = []
    changed_vals = []

    for r in rows:
        cur = get_bool(r, ["cur_ok", "current_ok", "majority_ok", "current_correct"])
        fin = get_bool(r, ["pred_ok", "final_ok", "final_correct"])

        if cur is not None:
            cur_vals.append(cur)
        if fin is not None:
            final_vals.append(fin)

        fixed = get_bool(r, ["fixed"])
        broken = get_bool(r, ["broken"])
        changed = get_bool(r, ["changed"])

        if fixed is None and cur is not None and fin is not None:
            fixed = int(cur == 0 and fin == 1)
        if broken is None and cur is not None and fin is not None:
            broken = int(cur == 1 and fin == 0)

        if changed is None:
            ca = r.get("current", r.get("current_answer", r.get("majority_answer", None)))
            pa = r.get("pred", r.get("pred_answer", r.get("final_answer", None)))
            if ca is not None and pa is not None:
                changed = int(str(ca).strip() != str(pa).strip())

        if fixed is not None:
            fixed_vals.append(int(fixed))
        if broken is not None:
            broken_vals.append(int(broken))
        if changed is not None:
            changed_vals.append(int(changed))

    if not cur_vals:
        return None

    cur_correct = sum(cur_vals)
    current_wrong = len(cur_vals) - cur_correct

    return {
        "n_eval_found": n,
        "current_wrong": current_wrong,
        "current_acc_on_eval": div(cur_correct, len(cur_vals)),
        "fixed_found": sum(fixed_vals) if fixed_vals else None,
        "broken_found": sum(broken_vals) if broken_vals else None,
        "changed_found": sum(changed_vals) if changed_vals else None,
        "source_type": "jsonl",
    }


def parse_debug_txt(fp):
    """
    解析类似：
    ## ALL
    | n | cur_acc | pred_acc | fixed | broken | net | changed |
    | 107 | 0.5981 | 0.6916 | 11 | 1 | 10 | 18 |
    """
    try:
        txt = Path(fp).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    m = re.search(
        r"## ALL.*?\|\s*n\s*\|\s*cur_acc\s*\|\s*pred_acc\s*\|\s*fixed\s*\|\s*broken\s*\|\s*net\s*\|\s*changed\s*\|.*?\n"
        r"\|[-:\s|]+\|\s*\n"
        r"\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(-?\d+)\s*\|\s*(\d+)\s*\|",
        txt,
        flags=re.S,
    )
    if not m:
        return None

    n = int(m.group(1))
    cur_acc = float(m.group(2))
    pred_acc = float(m.group(3))
    fixed = int(m.group(4))
    broken = int(m.group(5))
    net = int(m.group(6))
    changed = int(m.group(7))

    current_wrong = int(round(n * (1.0 - cur_acc)))

    return {
        "n_eval_found": n,
        "current_wrong": current_wrong,
        "current_acc_on_eval": cur_acc,
        "pred_acc_on_eval": pred_acc,
        "fixed_found": fixed,
        "broken_found": broken,
        "net_found": net,
        "changed_found": changed,
        "source_type": "debug_txt",
    }


def all_candidate_files():
    files = []

    for root in ["outputs/predictions", "outputs/metrics", "outputs/logs", "outputs/debug"]:
        p = Path(root)
        if not p.exists():
            continue
        for fp in p.rglob("*"):
            if not fp.is_file():
                continue
            s = str(fp).lower()
            if "gsm8k" not in s and "resample" not in s and "margin" not in s and "origmaj" not in s:
                continue
            if fp.suffix.lower() in [".jsonl", ".json", ".txt", ".log", ".md"]:
                files.append(fp)

    return files


def score_candidate(fp, known, stats):
    path = str(fp).lower()
    score = 0

    for t in known["tokens"]:
        if t.lower() in path:
            score += 5

    # n_eval 匹配很重要
    if stats and stats.get("n_eval_found") == known["n_eval"]:
        score += 10

    # fixed/broken/changed 匹配也加分；但 debug 可能是旧版本，不强制
    if stats:
        if stats.get("fixed_found") == known["fixed"]:
            score += 4
        if stats.get("broken_found") == known["broken"]:
            score += 4
        if stats.get("changed_found") == known["changed"]:
            score += 4

    return score


def find_best_source(known):
    best = None

    for fp in all_candidate_files():
        stats = None

        if fp.suffix.lower() == ".jsonl":
            rows = read_jsonl(fp)
            stats = compute_from_rows(rows)
        elif fp.suffix.lower() in [".txt", ".log", ".md"]:
            stats = parse_debug_txt(fp)

        if not stats:
            continue

        sc = score_candidate(fp, known, stats)
        if sc <= 0:
            continue

        item = {
            "file": str(fp),
            "score": sc,
            **stats,
        }

        if best is None or item["score"] > best["score"]:
            best = item

    return best


def compute_prf(known, source):
    fixed = known["fixed"]
    broken = known["broken"]
    changed = known["changed"]
    n_eval = known["n_eval"]

    precision = div(fixed, changed)
    safe_precision = div(fixed, fixed + broken)
    harm_rate = div(broken, changed)
    change_rate = div(changed, n_eval)

    current_wrong = None
    current_acc_on_eval = None
    recall = None
    cf1 = None
    source_file = None
    source_type = None

    if source:
        current_wrong = source.get("current_wrong")
        current_acc_on_eval = source.get("current_acc_on_eval")
        source_file = source.get("file")
        source_type = source.get("source_type")

    if current_wrong is not None:
        recall = div(fixed, current_wrong)
        cf1 = f1(precision, recall)

    # global fallback：全体 majority 错误样本中修复了多少
    global_n = 1319
    global_wrong = int(round(global_n * (1.0 - known["base_acc"])))
    global_recall = div(fixed, global_wrong)
    global_f1 = f1(precision, global_recall)

    return {
        "setting": known["setting"],
        "base_acc": known["base_acc"],
        "final_acc": known["final_acc"],
        "n_eval": n_eval,
        "current_acc_on_eval": current_acc_on_eval,
        "current_wrong": current_wrong,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": known["net"],
        "precision": precision,
        "recall_on_triggered_wrong": recall,
        "f1_on_triggered_wrong": cf1,
        "safe_precision": safe_precision,
        "harm_rate": harm_rate,
        "change_rate": change_rate,
        "global_wrong": global_wrong,
        "global_recall": global_recall,
        "global_f1": global_f1,
        "source_file": source_file,
        "source_type": source_type,
    }


def make_md(rows):
    lines = []
    lines.append("# GSM8K correction precision / recall / F1 with current_wrong search\n")

    lines.append("## Main table\n")
    lines.append("| Setting | Base | Final | n_eval | current_acc | current_wrong | changed | fixed | broken | net | precision | recall | F1 | safe_precision | harm_rate | source |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        src = r["source_file"] if r["source_file"] else "GLOBAL_FALLBACK_ONLY"
        lines.append(
            f"| {r['setting']} | {fmt(r['base_acc'])} | {fmt(r['final_acc'])} | {r['n_eval']} | "
            f"{fmt(r['current_acc_on_eval'])} | {fmt(r['current_wrong'])} | "
            f"{r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | "
            f"{fmt(r['precision'])} | {fmt(r['recall_on_triggered_wrong'])} | {fmt(r['f1_on_triggered_wrong'])} | "
            f"{fmt(r['safe_precision'])} | {fmt(r['harm_rate'])} | {src} |"
        )

    lines.append("\n## Global fallback table\n")
    lines.append("这里的 global_recall = fixed / full-set majority wrong，仅作为保底参考；论文主表优先使用上面的 triggered current_wrong。")
    lines.append("")
    lines.append("| Setting | global_wrong | fixed | precision | global_recall | global_F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['setting']} | {r['global_wrong']} | {r['fixed']} | "
            f"{fmt(r['precision'])} | {fmt(r['global_recall'])} | {fmt(r['global_f1'])} |"
        )

    lines.append("\n## Source note\n")
    lines.append("- 如果 source 是 jsonl，说明找到了 per-case current_ok/cur_ok 信息。")
    lines.append("- 如果 source 是 debug_txt，说明从 debug log 的 `cur_acc` 反推出 current_wrong。")
    lines.append("- 如果 source 是 GLOBAL_FALLBACK_ONLY，说明没有找到 triggered per-case current_wrong，只能用 full-set majority wrong 做保底。")

    return "\n".join(lines)


def main():
    results = []

    for known in KNOWN:
        src = find_best_source(known)
        row = compute_prf(known, src)
        results.append(row)

    out_json = METRIC_DIR / "gsm8k_correction_prf_with_current_wrong.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    out_jsonl = METRIC_DIR / "gsm8k_correction_prf_with_current_wrong.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    md = make_md(results)
    out_md = OUT_DIR / "gsm8k_correction_prf_with_current_wrong.md"
    out_md.write_text(md, encoding="utf-8")

    print(md)
    print("\nsaved:", out_md)
    print("saved:", out_json)
    print("saved:", out_jsonl)


if __name__ == "__main__":
    main()
