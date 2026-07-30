#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from pathlib import Path

IN = Path("outputs/metrics/baseline_tts_compare_mathqa_14b/mathqa_ds14b_optionmap_fair_fullbase_fixed_with_ggproxy.csv")
OUT = Path("outputs/metrics/baseline_14b_compare/ds14b_family_summary_with_mathqa_gg.md")

def fnum(x):
    try:
        if str(x).strip() in {"NA", ""}:
            return None
        return float(x)
    except Exception:
        return None

def fmt(x):
    return "NA" if x is None else f"{x:.4f}"

def fam(m):
    ml = m.lower()
    if "recorded-ours" in ml or "ours" in ml:
        return "Ours"
    if "gg_lite" in ml or "gg" in ml:
        return "GG-lite-proxy"
    if "cisc" in ml:
        return "CISC"
    if ml.startswith("esc"):
        return "ESC"
    if ml.startswith("sc"):
        return "SC"
    if "base" in ml:
        return "Base"
    return "Other"

rows = list(csv.DictReader(IN.open(encoding="utf-8")))

items = []
for r in rows:
    extra = fnum(r.get("extra_per_sample"))
    gain = fnum(r.get("gain"))
    cost_acc = gain / extra if extra and extra > 0 else None
    items.append({
        "family": fam(r["method"]),
        "method": r["method"],
        "acc": fnum(r["final_acc"]),
        "cost_acc": cost_acc,
        "repair_p": fnum(r.get("repair_precision")),
        "harm": fnum(r.get("harm_rate")),
    })

def cell(family):
    cr = [x for x in items if x["family"] == family]
    if not cr:
        return "—"
    best_acc = max([x for x in cr if x["acc"] is not None], key=lambda x: x["acc"], default=None)
    best_cost = max([x for x in cr if x["cost_acc"] is not None], key=lambda x: x["cost_acc"], default=None)
    best_rp = max([x for x in cr if x["repair_p"] is not None], key=lambda x: x["repair_p"], default=None)
    best_harm = min([x for x in cr if x["harm"] is not None], key=lambda x: x["harm"], default=None)
    return f"{fmt(best_acc['acc'] if best_acc else None)} / {fmt(best_cost['cost_acc'] if best_cost else None)} / {fmt(best_rp['repair_p'] if best_rp else None)} / {fmt(best_harm['harm'] if best_harm else None)}"

old = Path("outputs/metrics/baseline_14b_compare/ds14b_family_summary_with_mathqa.md")
lines = old.read_text(encoding="utf-8").splitlines()
lines = [ln for ln in lines if not ln.startswith("| mathqa |")]
lines.append(f"| mathqa | {cell('SC')} | {cell('ESC')} | {cell('CISC')} | {cell('GG-lite-proxy')} | {cell('Ours')} |")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("saved:", OUT)
print(OUT.read_text(encoding="utf-8"))
