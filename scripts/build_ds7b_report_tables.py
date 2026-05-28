import json
from pathlib import Path
from collections import defaultdict

METRIC_DIR = Path("outputs/metrics/model_ablation")
OPTIONMAP_DIR = Path("outputs/metrics/model_ablation_mathqa_optionmap")
TRAJ_DIR = Path("data/processed/trajectories/model_ablation")

OUT_SUMMARY = METRIC_DIR / "ds7b_summary_with_mathqa_optionmap.md"
OUT_COST = METRIC_DIR / "ds7b_cost_accuracy_table.md"
OUT_DIAG = METRIC_DIR / "ds7b_final_diagnostic_boundary_table.md"

N_TOTAL_FALLBACK = {
    "gsm8k": 1319,
    "svamp": 300,
    "asdiv": 2249,
    "math500": 500,
    "mathqa_optionmap": 500,
    "bbh_formal_fallacies": 100,
    "bbh_logical_deduction_five_objects": 100,
}

def load_json(fp):
    with open(fp, encoding="utf-8") as f:
        return json.load(f)

def pick(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def fmt(x):
    if x is None or x == "NA":
        return "NA"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

def count_lines(fp):
    p = Path(fp)
    if not p.exists():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def get_base_n_samples(ds):
    if ds == "mathqa_optionmap":
        return 500

    fp = METRIC_DIR / f"{ds}_deepseek7b_base.json"
    if fp.exists():
        try:
            d = load_json(fp)
            v = pick(d, ["n_samples", "n_total", "total", "num_samples"])
            if v is not None:
                return int(v)
        except Exception:
            pass

    return N_TOTAL_FALLBACK.get(ds)

def extra_rows_for_dataset(ds):
    if ds == "mathqa_optionmap":
        files = sorted(TRAJ_DIR.glob("mathqa_deepseek7b_extra_seed*.jsonl"))
    else:
        files = sorted(TRAJ_DIR.glob(f"{ds}_deepseek7b_extra_seed*.jsonl"))
    return sum(count_lines(fp) for fp in files), files

def collect_best_rows():
    groups = defaultdict(list)

    for fp in sorted(METRIC_DIR.glob("*deepseek7b*.json")):
        name = fp.name
        if "_deepseek7b_" not in name:
            continue

        ds = name.split("_deepseek7b_")[0]

        # 跳过 MathQA 普通 direct eval；它是错误口径
        if ds == "mathqa":
            continue

        kind = "base" if name.endswith("_base.json") else "confirm"

        try:
            x = load_json(fp)
        except Exception:
            continue

        groups[ds].append((fp, kind, x))

    rows = []

    for ds, items in groups.items():
        base = None
        for fp, kind, x in items:
            if kind == "base":
                base = pick(x, ["base_acc", "majority_acc", "majority_before_acc", "before_acc", "acc", "accuracy"])

        best = None
        for fp, kind, x in items:
            if kind != "confirm":
                continue

            final = pick(x, ["estimated_global_acc", "final_acc", "final_acc_on_eval", "best_acc", "after_acc", "acc", "accuracy"])
            if final is None:
                continue

            if best is None or float(final) > float(best[1]):
                best = (fp, final, x)

        if best is None:
            if base is not None:
                rows.append({
                    "dataset": ds,
                    "base": base,
                    "final": base,
                    "gain": 0.0,
                    "n_eval": "NA",
                    "changed": "NA",
                    "fixed": "NA",
                    "broken": "NA",
                    "net": "NA",
                    "file": "base only",
                })
            continue

        fp, final, x = best

        if base is None:
            base = pick(x, ["base_acc", "majority_acc", "majority_before_acc", "before_acc", "acc", "accuracy"])

        fixed = x.get("fixed")
        broken = x.get("broken")
        net = x.get("net")
        if net is None and fixed is not None and broken is not None:
            net = fixed - broken

        rows.append({
            "dataset": ds,
            "base": base,
            "final": final,
            "gain": None if base is None else float(final) - float(base),
            "n_eval": x.get("n_eval", x.get("n_samples", "NA")),
            "changed": x.get("changed", "NA"),
            "fixed": fixed if fixed is not None else "NA",
            "broken": broken if broken is not None else "NA",
            "net": net if net is not None else "NA",
            "file": str(fp),
        })

    # MathQA optionmap
    option_rows = []
    for fp in sorted(OPTIONMAP_DIR.glob("mathqa_deepseek7b_optionmap*.json")):
        try:
            x = load_json(fp)
        except Exception:
            continue

        final = pick(x, ["estimated_global_acc", "final_acc", "acc"])
        if final is None:
            continue

        option_rows.append((float(final), fp, x))

    if option_rows:
        option_rows.sort(
            key=lambda t: (
                -t[0],
                int(t[2].get("broken", 999999)),
                -int(t[2].get("fixed", -1))
            )
        )
        final, fp, x = option_rows[0]
        base = x.get("base_acc")
        rows.append({
            "dataset": "mathqa_optionmap",
            "base": base,
            "final": final,
            "gain": None if base is None else final - float(base),
            "n_eval": x.get("n_eval", "NA"),
            "changed": x.get("changed", "NA"),
            "fixed": x.get("fixed", "NA"),
            "broken": x.get("broken", "NA"),
            "net": x.get("net", "NA"),
            "file": str(fp),
        })

    rows.sort(key=lambda r: r["dataset"])
    return rows

def write_summary(rows):
    lines = []
    lines.append("| Dataset | Base Acc | Best Final Acc | ΔAcc | n_eval | changed | fixed | broken | net | best_file |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        lines.append(
            f"| {r['dataset']} | {fmt(r['base'])} | {fmt(r['final'])} | {fmt(r['gain'])} | "
            f"{r['n_eval']} | {r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | `{r['file']}` |"
        )

    OUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_cost_accuracy(rows):
    lines = []
    lines.append("| Dataset | Base Acc | Final Acc | ΔAcc | Eval Samples | Triggered Targets | Target Rate | Extra Calls | Extra/Target | Extra/Sample | Fixed | Broken | Net |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        ds = r["dataset"]
        n_total = get_base_n_samples(ds)

        try:
            n_eval = int(r["n_eval"])
        except Exception:
            n_eval = None

        extra_rows, _ = extra_rows_for_dataset(ds)

        target_rate = None
        if n_total and n_eval is not None:
            target_rate = n_eval / n_total

        extra_per_target = None
        if n_eval and n_eval > 0:
            extra_per_target = extra_rows / n_eval

        extra_per_sample = None
        if n_total:
            extra_per_sample = extra_rows / n_total

        lines.append(
            f"| {ds} | {fmt(r['base'])} | {fmt(r['final'])} | {fmt(r['gain'])} | "
            f"{n_total if n_total is not None else 'NA'} | {r['n_eval']} | {fmt(target_rate)} | "
            f"{extra_rows} | {fmt(extra_per_target)} | {fmt(extra_per_sample)} | "
            f"{r['fixed']} | {r['broken']} | {r['net']} |"
        )

    OUT_COST.write_text("\n".join(lines) + "\n", encoding="utf-8")

def verdict_for_row(r):
    ds = r["dataset"]

    try:
        gain = float(r["gain"])
    except Exception:
        gain = 0.0

    try:
        n_eval = int(r["n_eval"])
    except Exception:
        n_eval = None

    try:
        fixed = int(r["fixed"])
    except Exception:
        fixed = 0

    try:
        broken = int(r["broken"])
    except Exception:
        broken = 0

    try:
        net = int(r["net"])
    except Exception:
        net = fixed - broken

    if "mathqa" in ds:
        return "Use option mapping; direct choice/numeric evaluation is invalid."

    if n_eval is not None and n_eval < 80:
        return "Small target set; useful but should be reported as boundary case."

    if gain >= 0.10 and net > 0 and broken <= max(5, 0.15 * max(fixed, 1)):
        return "Strong improvement with low harm."

    if gain > 0 and net > 0:
        return "Positive improvement; acceptable trade-off."

    return "Weak or no improvement; keep as diagnostic boundary."

def write_diag(rows):
    lines = []
    lines.append("| Dataset | Best Setting | ΔAcc | Changed | Fixed | Broken | Net | Fix/(Fix+Broken) | Harm/Changed | Boundary / Diagnostic Note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        try:
            changed = int(r["changed"])
        except Exception:
            changed = 0
        try:
            fixed = int(r["fixed"])
        except Exception:
            fixed = 0
        try:
            broken = int(r["broken"])
        except Exception:
            broken = 0

        precision = None
        if fixed + broken > 0:
            precision = fixed / (fixed + broken)

        harm_rate = None
        if changed > 0:
            harm_rate = broken / changed

        best_setting = Path(r["file"]).name if r["file"] != "base only" else "base only"

        lines.append(
            f"| {r['dataset']} | `{best_setting}` | {fmt(r['gain'])} | "
            f"{r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | "
            f"{fmt(precision)} | {fmt(harm_rate)} | {verdict_for_row(r)} |"
        )

    OUT_DIAG.write_text("\n".join(lines) + "\n", encoding="utf-8")

rows = collect_best_rows()
write_summary(rows)
write_cost_accuracy(rows)
write_diag(rows)

print("saved:", OUT_SUMMARY)
print("saved:", OUT_COST)
print("saved:", OUT_DIAG)
