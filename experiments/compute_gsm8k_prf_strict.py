import json
import re
from pathlib import Path

OUT_DIR = Path("outputs/logs/final_summaries")
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR = Path("outputs/metrics/final_ablation")
METRIC_DIR.mkdir(parents=True, exist_ok=True)

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
        "must_tokens": ["margin030"],
        "prefer_tokens": ["currentkeep"],
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
        "must_tokens": ["margin040"],
        "prefer_tokens": ["currentkeep"],
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
        "must_tokens": ["margin040"],
        "prefer_tokens": ["selective", "origmaj", "top7"],
    },
]

BAD_DATASET_TOKENS = [
    "asdiv",
    "bbh",
    "mathqa",
    "math500",
    "svamp",
    "hotpot",
    "strategy",
    "multiarith",
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


def get_int(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            try:
                return int(row[k])
            except Exception:
                pass
    return None


def compute_from_jsonl(fp):
    rows = read_jsonl(fp)
    if not rows:
        return None

    cur_vals = []
    final_vals = []
    fixed_vals = []
    broken_vals = []
    changed_vals = []

    for r in rows:
        cur = get_int(r, ["cur_ok", "current_ok", "majority_ok", "current_correct"])
        fin = get_int(r, ["pred_ok", "final_ok", "final_correct"])

        if cur is not None:
            cur_vals.append(cur)
        if fin is not None:
            final_vals.append(fin)

        fixed = get_int(r, ["fixed"])
        broken = get_int(r, ["broken"])
        changed = get_int(r, ["changed"])

        if fixed is None and cur is not None and fin is not None:
            fixed = int(cur == 0 and fin == 1)
        if broken is None and cur is not None and fin is not None:
            broken = int(cur == 1 and fin == 0)

        if changed is None:
            ca = r.get("current_answer", r.get("current", r.get("majority_answer", None)))
            fa = r.get("final_answer", r.get("pred_answer", r.get("pred", None)))
            if ca is not None and fa is not None:
                changed = int(str(ca).strip() != str(fa).strip())

        if fixed is not None:
            fixed_vals.append(fixed)
        if broken is not None:
            broken_vals.append(broken)
        if changed is not None:
            changed_vals.append(changed)

    if not cur_vals:
        return None

    current_correct = sum(cur_vals)
    current_wrong = len(cur_vals) - current_correct

    return {
        "n_eval_found": len(rows),
        "current_acc_on_eval": div(current_correct, len(cur_vals)),
        "current_wrong": current_wrong,
        "fixed_found": sum(fixed_vals) if fixed_vals else None,
        "broken_found": sum(broken_vals) if broken_vals else None,
        "changed_found": sum(changed_vals) if changed_vals else None,
        "source_type": "jsonl",
    }


def parse_debug_table(fp):
    try:
        txt = Path(fp).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    # 适配 debug_resample_margin030_107.txt 这种表
    m = re.search(
        r"\|\s*n\s*\|\s*cur_acc\s*\|\s*pred_acc\s*\|\s*fixed\s*\|\s*broken\s*\|\s*net\s*\|\s*changed\s*\|"
        r".*?\n\|[-:\s|]+\|\s*\n"
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
        "current_acc_on_eval": cur_acc,
        "pred_acc_on_eval": pred_acc,
        "current_wrong": current_wrong,
        "fixed_found": fixed,
        "broken_found": broken,
        "net_found": net,
        "changed_found": changed,
        "source_type": "debug_txt",
    }


def is_allowed_gsm8k_source(fp):
    s = str(fp).lower()

    # 明确排除其他数据集
    if any(tok in s for tok in BAD_DATASET_TOKENS):
        return False

    # 允许两类：
    # 1. 路径含 gsm8k
    # 2. 明确的 debug_resample_margin 文件
    if "gsm8k" in s:
        return True
    if "debug_resample_margin" in s:
        return True

    return False


def candidate_files():
    roots = ["outputs/predictions", "outputs/metrics", "outputs/logs", "outputs/debug"]
    files = []

    for root in roots:
        p = Path(root)
        if not p.exists():
            continue

        for fp in p.rglob("*"):
            if not fp.is_file():
                continue

            if fp.suffix.lower() not in [".jsonl", ".json", ".txt", ".log", ".md"]:
                continue

            if not is_allowed_gsm8k_source(fp):
                continue

            s = str(fp).lower()
            if not any(tok in s for tok in ["margin030", "margin040", "currentkeep", "origmaj", "resample", "gsm8k"]):
                continue

            files.append(fp)

    return sorted(files)


def file_stats(fp):
    if fp.suffix.lower() == ".jsonl":
        return compute_from_jsonl(fp)
    if fp.suffix.lower() in [".txt", ".log", ".md"]:
        return parse_debug_table(fp)
    return None


def score_source(fp, stats, known):
    if not stats:
        return -1

    s = str(fp).lower()

    # must_tokens 必须满足，比如 margin030 / margin040
    for t in known["must_tokens"]:
        if t not in s:
            return -1

    # n_eval 必须匹配，否则不要用
    if stats.get("n_eval_found") != known["n_eval"]:
        return -1

    score = 100

    for t in known["prefer_tokens"]:
        if t in s:
            score += 10

    # 如果 fixed/broken/changed 能匹配，则更可信
    if stats.get("fixed_found") == known["fixed"]:
        score += 20
    if stats.get("broken_found") == known["broken"]:
        score += 20
    if stats.get("changed_found") == known["changed"]:
        score += 20

    # 如果统计值和 known 完全冲突，降权但不直接杀，因为有些 debug 可能是不同 variant
    if stats.get("fixed_found") is not None and stats.get("fixed_found") != known["fixed"]:
        score -= 30
    if stats.get("broken_found") is not None and stats.get("broken_found") != known["broken"]:
        score -= 30
    if stats.get("changed_found") is not None and stats.get("changed_found") != known["changed"]:
        score -= 30

    return score


def find_source(known):
    best = None
    inspected = []

    for fp in candidate_files():
        stats = file_stats(fp)
        if not stats:
            continue

        sc = score_source(fp, stats, known)
        item = {
            "file": str(fp),
            "score": sc,
            **stats,
        }
        inspected.append(item)

        if sc < 0:
            continue

        if best is None or sc > best["score"]:
            best = item

    return best, sorted(inspected, key=lambda x: -x["score"])


def compute_row(known, source):
    fixed = known["fixed"]
    broken = known["broken"]
    changed = known["changed"]
    n_eval = known["n_eval"]

    precision = div(fixed, changed)
    safe_precision = div(fixed, fixed + broken)
    harm_rate = div(broken, changed)
    change_rate = div(changed, n_eval)

    current_wrong = None
    current_acc = None
    recall = None
    cf1 = None
    source_file = None
    source_type = None

    if source:
        current_wrong = source.get("current_wrong")
        current_acc = source.get("current_acc_on_eval")
        source_file = source.get("file")
        source_type = source.get("source_type")

    if current_wrong is not None:
        recall = div(fixed, current_wrong)
        cf1 = f1(precision, recall)

    full_n = 1319
    global_wrong = int(round(full_n * (1.0 - known["base_acc"])))
    global_recall = div(fixed, global_wrong)
    global_f1 = f1(precision, global_recall)

    return {
        "setting": known["setting"],
        "base_acc": known["base_acc"],
        "final_acc": known["final_acc"],
        "n_eval": n_eval,
        "current_acc_on_eval": current_acc,
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


def main():
    rows = []
    debug = {}

    for known in KNOWN:
        src, inspected = find_source(known)
        rows.append(compute_row(known, src))
        debug[known["setting"]] = {
            "selected_source": src,
            "top_inspected": inspected[:20],
        }

    out_json = METRIC_DIR / "gsm8k_correction_prf_strict.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    out_debug = METRIC_DIR / "gsm8k_correction_prf_strict_debug.json"
    with out_debug.open("w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# GSM8K correction precision / recall / F1 strict\n")
    lines.append("| Setting | Base | Final | n_eval | current_acc | current_wrong | changed | fixed | broken | net | precision | recall | F1 | safe_precision | harm_rate | source |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        source = r["source_file"] if r["source_file"] else "NO_VALID_GSM8K_TRIGGER_SOURCE"
        lines.append(
            f"| {r['setting']} | {fmt(r['base_acc'])} | {fmt(r['final_acc'])} | {r['n_eval']} | "
            f"{fmt(r['current_acc_on_eval'])} | {fmt(r['current_wrong'])} | "
            f"{r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | "
            f"{fmt(r['precision'])} | {fmt(r['recall_on_triggered_wrong'])} | {fmt(r['f1_on_triggered_wrong'])} | "
            f"{fmt(r['safe_precision'])} | {fmt(r['harm_rate'])} | {source} |"
        )

    lines.append("\n## Global fallback\n")
    lines.append("| Setting | global_wrong | fixed | precision | global_recall | global_F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['setting']} | {r['global_wrong']} | {r['fixed']} | "
            f"{fmt(r['precision'])} | {fmt(r['global_recall'])} | {fmt(r['global_f1'])} |"
        )

    lines.append("\n说明：如果 source 是 `NO_VALID_GSM8K_TRIGGER_SOURCE`，说明当前目录没有找到可信的 GSM8K triggered per-case 文件；此时不能报告 triggered recall/F1，只能报告 precision、safe_precision、harm_rate 和 global fallback。")

    md = "\n".join(lines)
    out_md = OUT_DIR / "gsm8k_correction_prf_strict.md"
    out_md.write_text(md, encoding="utf-8")

    print(md)
    print("\nsaved:", out_md)
    print("saved:", out_json)
    print("saved:", out_debug)


if __name__ == "__main__":
    main()
