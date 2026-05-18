import json
import math
from pathlib import Path
from collections import defaultdict, Counter


OUT_DIR = Path("outputs/logs/final_summaries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_DIR = Path("outputs/metrics/final_ablation")
METRIC_DIR.mkdir(parents=True, exist_ok=True)


def read_json(fp):
    p = Path(fp)
    if not p.exists():
        return None
    try:
        return json.load(open(p, "r", encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


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


def get_first_existing(*keys, obj):
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def resolve_out_jsonl(metric_fp, metric):
    # 1) 优先使用 metric 里的 out_jsonl
    p = metric.get("out_jsonl")
    if p and Path(p).exists():
        return Path(p)

    # 2) 常规路径替换：outputs/metrics/xxx/a.json -> outputs/predictions/xxx/a.jsonl
    m = Path(metric_fp)
    parts = list(m.parts)
    if "metrics" in parts:
        idx = parts.index("metrics")
        parts[idx] = "predictions"
        cand = Path(*parts).with_suffix(".jsonl")
        if cand.exists():
            return cand

    # 3) 在 outputs/predictions 下按同名文件搜索
    stem = m.stem
    candidates = list(Path("outputs/predictions").glob(f"**/{stem}.jsonl"))
    if candidates:
        return candidates[0]

    return None


def label_from_metric_path(fp, metric):
    p = Path(fp)
    s = str(p)

    dataset = metric.get("dataset", "")
    subtask = metric.get("subtask", "")
    tag = metric.get("tag", "")
    rule = metric.get("rule", "")

    if "asdiv_numeric_extra_confirm" in s or dataset == "asdiv_numeric":
        ds = "ASDiv-numeric"
        setting = tag or p.stem
    elif "mathqa" in s or dataset == "mathqa":
        ds = "MathQA"
        setting = p.stem
    elif "math500" in s or dataset == "math500":
        ds = "MATH500"
        setting = p.stem
    elif "bbh_logic_fixed_extra_confirm" in s or dataset == "bbh_logic":
        ds = "BBH-" + (subtask or "logic")
        setting = p.stem.replace((subtask or "") + "_", "")
    elif "gsm8k" in s or dataset == "gsm8k":
        ds = "GSM8K"
        setting = p.stem
    elif "svamp" in s or dataset == "svamp":
        ds = "SVAMP"
        setting = p.stem
    else:
        ds = dataset or p.parent.name
        setting = rule or p.stem

    return ds, setting


def compute_from_prediction_rows(rows, metric=None):
    metric = metric or {}

    n_eval = len(rows)
    if n_eval == 0:
        return None

    current_ok_vals = []
    final_ok_vals = []
    changed_vals = []
    fixed_vals = []
    broken_vals = []

    for r in rows:
        cur = get_first_existing("current_ok", "current_correct", "majority_ok", "majority_ok_fixed", obj=r)
        fin = get_first_existing("final_ok", "final_correct", "final_ok_fixed", obj=r)
        chg = get_first_existing("changed", obj=r)
        fix = get_first_existing("fixed", obj=r)
        bro = get_first_existing("broken", obj=r)

        # 如果 fixed/broken 没有，但 current/final 有，可以推出来
        if cur is not None:
            cur = int(cur)
        if fin is not None:
            fin = int(fin)

        if fix is None and cur is not None and fin is not None:
            fix = int(cur == 0 and fin == 1)
        if bro is None and cur is not None and fin is not None:
            bro = int(cur == 1 and fin == 0)
        if chg is None:
            # 如果没有 changed 字段，就只能用 final_answer != current_answer 尝试推
            ca = r.get("current_answer", r.get("current_answer_fixed", None))
            fa = r.get("final_answer", r.get("final_answer_fixed", None))
            if ca is not None and fa is not None:
                chg = int(str(ca).strip() != str(fa).strip())

        if cur is not None:
            current_ok_vals.append(cur)
        if fin is not None:
            final_ok_vals.append(fin)
        if chg is not None:
            changed_vals.append(int(chg))
        if fix is not None:
            fixed_vals.append(int(fix))
        if bro is not None:
            broken_vals.append(int(bro))

    current_correct = sum(current_ok_vals)
    final_correct = sum(final_ok_vals)
    changed = sum(changed_vals)
    fixed = sum(fixed_vals)
    broken = sum(broken_vals)
    net = fixed - broken

    if current_ok_vals:
        current_wrong = len(current_ok_vals) - current_correct
        current_acc = div(current_correct, len(current_ok_vals))
    else:
        current_acc = metric.get("current_acc_on_eval")
        current_wrong = int(round(n_eval * (1 - current_acc))) if current_acc is not None else None

    if final_ok_vals:
        final_acc = div(final_correct, len(final_ok_vals))
    else:
        final_acc = metric.get("final_acc_on_eval")

    precision_all_changed = div(fixed, changed)
    recall_wrong = div(fixed, current_wrong) if current_wrong is not None else 0.0
    correction_f1 = f1(precision_all_changed, recall_wrong)

    safe_precision = div(fixed, fixed + broken)
    harm_rate = div(broken, changed)
    change_rate = div(changed, n_eval)

    return {
        "n_eval": n_eval,
        "current_acc_on_eval": current_acc,
        "final_acc_on_eval": final_acc,
        "current_wrong": current_wrong,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "change_rate": change_rate,
        "correction_precision": precision_all_changed,
        "correction_recall": recall_wrong,
        "correction_f1": correction_f1,
        "safe_precision": safe_precision,
        "harm_rate": harm_rate,
    }


def compute_from_metric_json(metric):
    n_eval = metric.get("n_eval") or metric.get("n_resampled") or metric.get("n_samples")
    if not n_eval:
        return None

    fixed = int(metric.get("fixed", 0))
    broken = int(metric.get("broken", 0))
    changed = int(metric.get("changed", 0))
    net = int(metric.get("net", fixed - broken))

    current_acc = metric.get("current_acc_on_eval")
    final_acc = metric.get("final_acc_on_eval")

    # 有些 ASDiv 叫 estimated_numeric_acc；有些叫 estimated_global_acc
    if final_acc is None:
        final_acc = metric.get("estimated_numeric_acc", metric.get("estimated_global_acc"))

    base_acc = metric.get("numeric_base_acc", metric.get("base_acc"))
    if current_acc is None:
        current_acc = base_acc

    if "current_acc_on_eval" in metric:
        current_wrong = int(round(n_eval * (1 - metric["current_acc_on_eval"])))
    else:
        current_wrong = None

    precision_all_changed = div(fixed, changed)
    recall_wrong = div(fixed, current_wrong) if current_wrong is not None else 0.0
    correction_f1 = f1(precision_all_changed, recall_wrong)
    safe_precision = div(fixed, fixed + broken)
    harm_rate = div(broken, changed)
    change_rate = div(changed, n_eval)

    return {
        "n_eval": n_eval,
        "current_acc_on_eval": current_acc,
        "final_acc_on_eval": final_acc,
        "current_wrong": current_wrong,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "change_rate": change_rate,
        "correction_precision": precision_all_changed,
        "correction_recall": recall_wrong,
        "correction_f1": correction_f1,
        "safe_precision": safe_precision,
        "harm_rate": harm_rate,
    }


def collect_metric_files():
    patterns = [
        "outputs/metrics/asdiv_numeric_extra_confirm/*.json",
        "outputs/metrics/mathqa_scale_extra_confirm/*.json",
        "outputs/metrics/bbh_logic_fixed_extra_confirm_smoke/*.json",
        "outputs/metrics/math500_confirm_clean_v3_all/*.json",
        "outputs/metrics/math500_guard_variant*.json",
        "outputs/metrics/true_ablation/*.json",
    ]

    files = []
    for pat in patterns:
        files.extend(Path(".").glob(pat))

    # 去重
    seen = set()
    out = []
    for fp in files:
        s = str(fp)
        if s not in seen:
            seen.add(s)
            out.append(fp)
    return out


def score_for_best(row):
    # 优先按 final/global acc，再按 broken 少、fixed 多
    acc = row.get("global_or_estimated_acc")
    if acc is None:
        acc = row.get("final_acc_on_eval") or 0.0
    return (-acc, row.get("broken", 999999), -row.get("fixed", -999999), row.get("changed", 999999))


def main():
    all_rows = []

    for fp in collect_metric_files():
        metric = read_json(fp)
        if not isinstance(metric, dict):
            continue

        # 必须是 correction 类结果
        if not any(k in metric for k in ["fixed", "broken", "changed", "net"]):
            continue

        ds, setting = label_from_metric_path(fp, metric)
        out_jsonl = resolve_out_jsonl(fp, metric)

        stats = None
        source = "metric_json"

        if out_jsonl is not None:
            rows = read_jsonl(out_jsonl)
            stats = compute_from_prediction_rows(rows, metric)
            if stats is not None:
                source = str(out_jsonl)

        if stats is None:
            stats = compute_from_metric_json(metric)
            source = str(fp)

        if stats is None:
            continue

        base_acc = metric.get("base_acc", metric.get("numeric_base_acc"))
        est_acc = metric.get("estimated_global_acc", metric.get("estimated_numeric_acc"))
        if est_acc is None:
            est_acc = stats.get("final_acc_on_eval")

        row = {
            "dataset": ds,
            "setting": setting,
            "metric_file": str(fp),
            "source": source,
            "base_acc": base_acc,
            "global_or_estimated_acc": est_acc,
            **stats,
        }
        all_rows.append(row)

    # 手动补 GSM8K 主结果，因为有些 GSM8K 结果只有 summary 表，没有 per-case jsonl
    manual_rows = [
        {
            "dataset": "GSM8K",
            "setting": "margin030_currentkeep2",
            "metric_file": "manual_from_final_summary",
            "source": "final_summary",
            "base_acc": 0.8886,
            "global_or_estimated_acc": 0.9196,
            "n_eval": 107,
            "current_acc_on_eval": None,
            "final_acc_on_eval": None,
            "current_wrong": None,
            "changed": 18,
            "fixed": 13,
            "broken": 1,
            "net": 12,
            "change_rate": div(18, 107),
            "correction_precision": div(13, 18),
            "correction_recall": 0.0,
            "correction_f1": 0.0,
            "safe_precision": div(13, 14),
            "harm_rate": div(1, 18),
        },
        {
            "dataset": "GSM8K",
            "setting": "margin040_selective_origmaj_top7",
            "metric_file": "manual_from_final_summary",
            "source": "final_summary",
            "base_acc": 0.8886,
            "global_or_estimated_acc": 0.9212,
            "n_eval": 210,
            "current_acc_on_eval": None,
            "final_acc_on_eval": None,
            "current_wrong": None,
            "changed": 22,
            "fixed": 15,
            "broken": 1,
            "net": 14,
            "change_rate": div(22, 210),
            "correction_precision": div(15, 22),
            "correction_recall": 0.0,
            "correction_f1": 0.0,
            "safe_precision": div(15, 16),
            "harm_rate": div(1, 22),
        },
    ]

    all_rows.extend(manual_rows)

    # 保存 all jsonl
    jsonl_out = METRIC_DIR / "qwen7b_correction_prf_all.jsonl"
    with jsonl_out.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 每个 dataset 选 best
    by_ds = defaultdict(list)
    for r in all_rows:
        by_ds[r["dataset"]].append(r)

    best_rows = []
    for ds, rows in by_ds.items():
        rows = sorted(rows, key=score_for_best)
        best_rows.append(rows[0])

    best_rows = sorted(best_rows, key=lambda x: x["dataset"])

    # 输出 markdown
    headers = [
        "Dataset", "Setting", "Base", "Best/Final",
        "n_eval", "changed", "fixed", "broken", "net",
        "Prec(fixed/changed)", "Recall(fixed/wrong)", "F1",
        "SafePrec(fixed/(fixed+broken))", "HarmRate"
    ]

    def row_to_md(r):
        return [
            r["dataset"],
            r["setting"],
            fmt(r.get("base_acc")),
            fmt(r.get("global_or_estimated_acc")),
            r.get("n_eval", "-"),
            r.get("changed", "-"),
            r.get("fixed", "-"),
            r.get("broken", "-"),
            r.get("net", "-"),
            fmt(r.get("correction_precision")),
            fmt(r.get("correction_recall")) if r.get("current_wrong") is not None else "-",
            fmt(r.get("correction_f1")) if r.get("current_wrong") is not None else "-",
            fmt(r.get("safe_precision")),
            fmt(r.get("harm_rate")),
        ]

    def make_table(rows):
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(x) for x in row_to_md(r)) + " |")
        return "\n".join(lines)

    best_md = "# Qwen7B correction precision / recall / F1: best per dataset\n\n"
    best_md += make_table(best_rows) + "\n\n"
    best_md += "说明：\n\n"
    best_md += "- Prec(fixed/changed)：所有被修改样本里，真正由错改对的比例。\n"
    best_md += "- Recall(fixed/wrong)：当前错误样本中，被成功修复的比例。若缺少 per-case current_ok，则记为 `-`。\n"
    best_md += "- SafePrec(fixed/(fixed+broken))：只看 fixed 与 broken 的安全精度，忽略 wrong-to-wrong 的中性变化。\n"
    best_md += "- HarmRate：被修改样本中由对改错的比例。\n"

    (OUT_DIR / "qwen7b_correction_prf_best.md").write_text(best_md, encoding="utf-8")

    all_sorted = sorted(
        all_rows,
        key=lambda x: (x["dataset"], -(x.get("global_or_estimated_acc") or 0), x.get("broken", 999), -x.get("fixed", -999))
    )

    all_md = "# Qwen7B correction precision / recall / F1: all discovered settings\n\n"
    all_md += make_table(all_sorted) + "\n"
    (OUT_DIR / "qwen7b_correction_prf_all.md").write_text(all_md, encoding="utf-8")

    print(best_md)
    print("saved:", OUT_DIR / "qwen7b_correction_prf_best.md")
    print("saved:", OUT_DIR / "qwen7b_correction_prf_all.md")
    print("saved:", jsonl_out)


if __name__ == "__main__":
    main()
