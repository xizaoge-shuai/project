import json
from pathlib import Path

OUT = Path("outputs/logs/final_summaries")
OUT.mkdir(parents=True, exist_ok=True)
MET = Path("outputs/metrics/final_ablation")
MET.mkdir(parents=True, exist_ok=True)

ITEMS = [
    (
        "margin030_currentkeep2",
        "outputs/metrics/resample_confirm_margin030_107_extra4_seedaware_total3_seed2_currentkeep2.json",
        0.8886,
        0.9196,
    ),
    (
        "margin040_currentkeep2",
        "outputs/metrics/resample_confirm_margin040_extra4_seedaware_total3_seed2_currentkeep2.json",
        0.8886,
        0.9204,
    ),
    (
        "margin040_selective_origmaj_top7",
        "outputs/metrics/resample_confirm_margin040_extra4_seedaware_total3_seed2_currentkeep2_selective_origmaj_top7.json",
        0.8886,
        0.9212,
    ),
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

rows = []

for setting, fp, base_acc, final_acc in ITEMS:
    p = Path(fp)
    if not p.exists():
        rows.append({
            "setting": setting,
            "metric_exists": False,
            "metric_file": fp,
            "base_acc": base_acc,
            "final_acc": final_acc,
        })
        continue

    x = json.load(open(p, encoding="utf-8"))

    n = int(x.get("n_resampled", x.get("n_eval", 0)))
    fixed = int(x.get("fixed", 0))
    broken = int(x.get("broken", 0))
    changed = int(x.get("changed", 0))
    net = int(x.get("net", fixed - broken))

    acc_on_resampled = x.get("acc_on_resampled", None)

    current_wrong = None
    current_acc_on_resampled = None
    recall = None
    cf1 = None

    if acc_on_resampled is not None and n:
        final_correct = round(float(acc_on_resampled) * n)
        current_correct = final_correct - net
        current_wrong = n - current_correct
        current_acc_on_resampled = current_correct / n
        recall = div(fixed, current_wrong)
        cf1 = f1(div(fixed, changed), recall)

    precision = div(fixed, changed)
    safe_precision = div(fixed, fixed + broken)
    harm_rate = div(broken, changed)

    global_wrong = round(1319 * (1 - base_acc))
    global_recall = div(fixed, global_wrong)
    global_f1 = f1(precision, global_recall)

    rows.append({
        "setting": setting,
        "metric_exists": True,
        "metric_file": fp,
        "base_acc": base_acc,
        "final_acc": final_acc,
        "n_resampled": n,
        "acc_on_resampled": acc_on_resampled,
        "current_acc_on_resampled_derived": current_acc_on_resampled,
        "current_wrong_derived": current_wrong,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "precision": precision,
        "recall_derived": recall,
        "f1_derived": cf1,
        "safe_precision": safe_precision,
        "harm_rate": harm_rate,
        "global_wrong": global_wrong,
        "global_recall": global_recall,
        "global_f1": global_f1,
    })

with open(MET / "gsm8k_prf_from_metrics_acc.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

lines = []
lines.append("# GSM8K PRF derived from metrics acc_on_resampled\n")
lines.append("| Setting | metric_exists | Base | Final | n | acc_on_resampled | current_acc_derived | current_wrong_derived | changed | fixed | broken | net | precision | recall_derived | F1_derived | safe_precision | harm_rate |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in rows:
    lines.append(
        f"| {r['setting']} | {r.get('metric_exists')} | {fmt(r.get('base_acc'))} | {fmt(r.get('final_acc'))} | "
        f"{fmt(r.get('n_resampled'))} | {fmt(r.get('acc_on_resampled'))} | "
        f"{fmt(r.get('current_acc_on_resampled_derived'))} | {fmt(r.get('current_wrong_derived'))} | "
        f"{fmt(r.get('changed'))} | {fmt(r.get('fixed'))} | {fmt(r.get('broken'))} | {fmt(r.get('net'))} | "
        f"{fmt(r.get('precision'))} | {fmt(r.get('recall_derived'))} | {fmt(r.get('f1_derived'))} | "
        f"{fmt(r.get('safe_precision'))} | {fmt(r.get('harm_rate'))} |"
    )

lines.append("\n说明：这里的 current_wrong 是由 metrics 里的 acc_on_resampled 和 net 反推得到的，不是逐样本 current_ok 直接统计。论文里如果使用，应标注为 derived triggered recall/F1。")

md = "\n".join(lines)
(OUT / "gsm8k_prf_from_metrics_acc.md").write_text(md, encoding="utf-8")
print(md)
