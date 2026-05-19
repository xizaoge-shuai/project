import json
from pathlib import Path

rows = [
    {
        "Dataset": "GSM8K",
        "Acc": 0.9212,
        "changed": 22,
        "fixed": 15,
        "broken": 1,
        "net": 14,
        "current_wrong": 76,
        "Type": "derived",
    },
    {
        "Dataset": "SVAMP",
        "Acc": 0.9233,
        "changed": 12,
        "fixed": 8,
        "broken": 1,
        "net": 7,
        "current_wrong": 26,
        "Type": "per-case target",
    },
    {
        "Dataset": "ASDiv-numeric",
        "Acc": 0.9471,
        "changed": 254,
        "fixed": 193,
        "broken": 13,
        "net": 180,
        "current_wrong": 284,
        "Type": "per-case",
    },
    {
        "Dataset": "MathQA",
        "Acc": 0.8580,
        "changed": 84,
        "fixed": 49,
        "broken": 9,
        "net": 40,
        "current_wrong": 101,
        "Type": "per-case",
    },
    {
        "Dataset": "MATH500-best",
        "Acc": 0.7160,
        "changed": 92,
        "fixed": 37,
        "broken": 5,
        "net": 32,
        "current_wrong": 148,
        "Type": "reported+target",
    },
    {
        "Dataset": "MATH500-balanced",
        "Acc": 0.7080,
        "changed": 60,
        "fixed": 31,
        "broken": 3,
        "net": 28,
        "current_wrong": 148,
        "Type": "reported+target",
    },
    {
        "Dataset": "MATH500-old-total2-seed2",
        "Acc": 0.7140,
        "changed": 96,
        "fixed": 38,
        "broken": 7,
        "net": 31,
        "current_wrong": 148,
        "Type": "per-case",
    },
    {
        "Dataset": "BBH-logical5-main",
        "Acc": 0.7500,
        "changed": 28,
        "fixed": 19,
        "broken": 2,
        "net": 17,
        "current_wrong": 39,
        "Type": "per-case",
    },
    {
        "Dataset": "BBH-logical5-safer",
        "Acc": 0.7500,
        "changed": 28,
        "fixed": 18,
        "broken": 1,
        "net": 17,
        "current_wrong": 39,
        "Type": "per-case",
    },
    {
        "Dataset": "BBH-formal",
        "Acc": 0.6300,
        "changed": 31,
        "fixed": 14,
        "broken": 7,
        "net": 7,
        "current_wrong": 43,
        "Type": "per-case",
    },
]

def div(a, b):
    return a / b if b else None

def f1(p, r):
    return 2 * p * r / (p + r) if p is not None and r is not None and (p + r) else None

def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)

for r in rows:
    r["precision"] = div(r["fixed"], r["changed"])
    r["recall"] = div(r["fixed"], r["current_wrong"])
    r["F1"] = f1(r["precision"], r["recall"])
    r["safe_precision"] = div(r["fixed"], r["fixed"] + r["broken"])
    r["harm_rate"] = div(r["broken"], r["changed"])

out_json = Path("outputs/metrics/final_ablation/final_correction_prf_table.json")
out_md = Path("outputs/logs/final_summaries/final_correction_prf_table.md")

out_json.parent.mkdir(parents=True, exist_ok=True)
out_md.parent.mkdir(parents=True, exist_ok=True)

json.dump(rows, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

lines = []
lines.append("# Final correction-level PRF table\n")
lines.append("| Dataset | Acc | changed | fixed | broken | net | precision | recall | F1 | safe_precision | harm_rate | Type |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for r in rows:
    lines.append(
        f"| {r['Dataset']} | {fmt(r['Acc'])} | {r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | "
        f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['F1'])} | "
        f"{fmt(r['safe_precision'])} | {fmt(r['harm_rate'])} | {r['Type']} |"
    )

lines.append("\nNotes:")
lines.append("- per-case: computed directly from prediction-level current/final correctness records.")
lines.append("- derived: reconstructed from aggregate triggered-set metrics.")
lines.append("- reported+target: fixed/broken/changed are taken from the confirmed final summary, while current_wrong is taken from the corresponding target set.")
lines.append("- Correction recall is fixed/current_wrong, so it measures how many originally wrong triggered cases are repaired. Since the method is conservative, recall can be moderate while safe_precision and harm_rate remain strong.")

out_md.write_text("\n".join(lines), encoding="utf-8")

print(out_md.read_text(encoding="utf-8"))
print("saved:", out_md)
print("saved:", out_json)
