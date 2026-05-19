import json
from pathlib import Path

SPECS = [
    {
        "name": "SVAMP-final_guard",
        "file": "outputs/final_selected_results/final_cross_svamp_origmaj2_predictions.jsonl",
        "current_ok": "current_best_ok",
        "final_ok": "final_guard_ok",
        "changed": "final_guard_changed",
        "fixed": "final_guard_fixed",
        "broken": "final_guard_broken",
        "acc": 0.9233,
    },
    {
        "name": "SVAMP-seedaware",
        "file": "outputs/final_selected_results/final_cross_svamp_origmaj2_predictions.jsonl",
        "current_ok": "current_best_ok",
        "final_ok": "seedaware_ok",
        "changed": "seedaware_changed",
        "fixed": "seedaware_fixed",
        "broken": "seedaware_broken",
        "acc": None,
    },
    {
        "name": "SVAMP-orig_majority_guard",
        "file": "outputs/final_selected_results/final_cross_svamp_origmaj2_predictions.jsonl",
        "current_ok": "current_best_ok",
        "final_ok": "orig_majority_guard_ok",
        "changed": "orig_majority_guard_changed",
        "fixed": "orig_majority_guard_fixed",
        "broken": "orig_majority_guard_broken",
        "acc": None,
    },
    {
        "name": "MATH500-old-total2-seed2",
        "file": "outputs/predictions/math500_confirm_clean_v3_all/math500_all244_total2_seed2.jsonl",
        "current_ok": "current_ok",
        "final_ok": "final_ok",
        "changed": "changed",
        "fixed": "fixed",
        "broken": "broken",
        "acc": 0.7140,
    },
    {
        "name": "MATH500-total2-seed4",
        "file": "outputs/predictions/math500_confirm_clean_v3_all/math500_all244_total2_seed4.jsonl",
        "current_ok": "current_ok",
        "final_ok": "final_ok",
        "changed": "changed",
        "fixed": "fixed",
        "broken": "broken",
        "acc": None,
    },
    {
        "name": "MATH500-v2-best",
        "file": "outputs/predictions/math500_confirm_clean_v3_all/math500_guard_variant_v2_best.jsonl",
        "current_ok": "current_ok",
        "final_ok": "final_ok",
        "changed": "changed",
        "fixed": "fixed",
        "broken": "broken",
        "acc": 0.7160,
    },
    {
        "name": "MATH500-v2-balanced",
        "file": "outputs/predictions/math500_confirm_clean_v3_all/math500_guard_variant_v2_balanced.jsonl",
        "current_ok": "current_ok",
        "final_ok": "final_ok",
        "changed": "changed",
        "fixed": "fixed",
        "broken": "broken",
        "acc": 0.7080,
    },
    {
        "name": "BBH-logical5-main",
        "file": "outputs/predictions/bbh_logic_fixed_extra_confirm_smoke/logical_deduction_five_objects_total2_seed2_margin1.jsonl",
        "current_ok": "current_ok",
        "final_ok": "final_ok",
        "changed": "changed",
        "fixed": "fixed",
        "broken": "broken",
        "acc": 0.7500,
    },
    {
        "name": "BBH-logical5-safer",
        "file": "outputs/predictions/true_ablation/bbh_logical_deduction_five_objects_seedbudget6_total2_seed1_margin0.jsonl",
        "current_ok": "current_ok",
        "final_ok": "final_ok",
        "changed": "changed",
        "fixed": "fixed",
        "broken": "broken",
        "acc": 0.7500,
    },
    {
        "name": "BBH-formal",
        "file": "outputs/predictions/bbh_logic_fixed_extra_confirm_smoke/formal_fallacies_total2_seed2_margin0.jsonl",
        "current_ok": "current_ok",
        "final_ok": "final_ok",
        "changed": "changed",
        "fixed": "fixed",
        "broken": "broken",
        "acc": 0.6300,
    },
]

def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]

def get_int(r, k, default=0):
    if k not in r or r[k] is None:
        return default
    try:
        return int(r[k])
    except Exception:
        return int(bool(r[k]))

def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)

rows_out = []

for s in SPECS:
    rows = read_jsonl(s["file"])
    if not rows:
        rows_out.append({
            "name": s["name"], "exists": False, "file": s["file"]
        })
        continue

    current_wrong = sum(1 - get_int(r, s["current_ok"]) for r in rows if s["current_ok"] in r)
    current_ok_n = sum(get_int(r, s["current_ok"]) for r in rows if s["current_ok"] in r)
    final_ok_n = sum(get_int(r, s["final_ok"]) for r in rows if s["final_ok"] in r)

    changed = sum(get_int(r, s["changed"]) for r in rows)
    fixed = sum(get_int(r, s["fixed"]) for r in rows)
    broken = sum(get_int(r, s["broken"]) for r in rows)
    net = fixed - broken

    precision = fixed / changed if changed else None
    recall = fixed / current_wrong if current_wrong else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    safe_precision = fixed / (fixed + broken) if fixed + broken > 0 else None
    harm_rate = broken / changed if changed else None

    acc_from_file = final_ok_n / len(rows) if final_ok_n else None
    current_acc = current_ok_n / len(rows) if current_ok_n else None

    rows_out.append({
        "name": s["name"],
        "exists": True,
        "file": s["file"],
        "n": len(rows),
        "acc": s["acc"] if s["acc"] is not None else acc_from_file,
        "current_acc": current_acc,
        "current_wrong": current_wrong,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "safe_precision": safe_precision,
        "harm_rate": harm_rate,
    })

out_md = Path("outputs/logs/final_summaries/prf_with_real_fields.md")
out_json = Path("outputs/metrics/prf_with_real_fields.json")
out_json.parent.mkdir(parents=True, exist_ok=True)

with out_md.open("w", encoding="utf-8") as f:
    f.write("# PRF with real field mapping\n\n")
    f.write("| Dataset | exists | Acc | n | current_acc | current_wrong | changed | fixed | broken | net | precision | recall | F1 | safe_precision | harm_rate |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows_out:
        if not r.get("exists"):
            f.write(f"| {r['name']} | 0 | - | - | - | - | - | - | - | - | - | - | - | - | - |\n")
            continue
        f.write(
            f"| {r['name']} | 1 | {fmt(r['acc'])} | {r['n']} | {fmt(r['current_acc'])} | {fmt(r['current_wrong'])} | "
            f"{fmt(r['changed'])} | {fmt(r['fixed'])} | {fmt(r['broken'])} | {fmt(r['net'])} | "
            f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['safe_precision'])} | {fmt(r['harm_rate'])} |\n"
        )

json.dump(rows_out, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print(out_md.read_text(encoding="utf-8"))
print("saved:", out_md)
print("saved:", out_json)
