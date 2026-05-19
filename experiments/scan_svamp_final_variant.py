import json
from pathlib import Path

TARGET = {
    "fixed": 8,
    "broken": 1,
    "changed": 12,
    "net": 7,
}

CURRENT_KEYS = [
    "current_best_ok",
    "current_ok",
    "majority_ok",
    "orig_majority_ok",
    "before_ok",
]

PREFIXES = [
    "seedaware",
    "final_guard",
    "orig_majority_guard",
    "guard",
    "final",
    "",
]

def read_jsonl(fp):
    rows = []
    try:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception:
        return []
    return rows

def read_json(fp):
    try:
        x = json.load(open(fp, encoding="utf-8"))
        return x
    except Exception:
        return None

def get_int(r, k):
    if not k or k not in r or r[k] is None:
        return None
    try:
        return int(r[k])
    except Exception:
        return int(bool(r[k]))

def field(prefix, name):
    return f"{prefix}_{name}" if prefix else name

def score(stats):
    s = 0
    for k, v in TARGET.items():
        if stats.get(k) == v:
            s += 100
        elif stats.get(k) is not None:
            s -= abs(stats[k] - v)
    return s

def calc_jsonl(fp, rows):
    outs = []

    for cur_key in CURRENT_KEYS:
        if rows and cur_key not in rows[0]:
            continue

        for prefix in PREFIXES:
            fixed_k = field(prefix, "fixed")
            broken_k = field(prefix, "broken")
            changed_k = field(prefix, "changed")
            ok_k = field(prefix, "ok")

            keys = set(rows[0].keys()) if rows else set()
            if fixed_k not in keys or broken_k not in keys or changed_k not in keys:
                continue

            n = len(rows)
            current_wrong = sum(1 - get_int(r, cur_key) for r in rows if get_int(r, cur_key) is not None)
            current_correct = sum(get_int(r, cur_key) for r in rows if get_int(r, cur_key) is not None)

            fixed = sum(get_int(r, fixed_k) or 0 for r in rows)
            broken = sum(get_int(r, broken_k) or 0 for r in rows)
            changed = sum(get_int(r, changed_k) or 0 for r in rows)
            net = fixed - broken

            final_correct = None
            final_acc_target = None
            if ok_k in keys:
                vals = [get_int(r, ok_k) for r in rows if get_int(r, ok_k) is not None]
                if vals:
                    final_correct = sum(vals)
                    final_acc_target = final_correct / len(vals)

            precision = fixed / changed if changed else None
            recall = fixed / current_wrong if current_wrong else None
            f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall > 0 else None
            safe_precision = fixed / (fixed + broken) if fixed + broken else None
            harm_rate = broken / changed if changed else None

            stats = {
                "file": str(fp),
                "kind": "jsonl",
                "prefix": prefix or "plain",
                "current_key": cur_key,
                "n": n,
                "current_wrong": current_wrong,
                "current_acc_target": current_correct / n if n else None,
                "final_acc_target": final_acc_target,
                "changed": changed,
                "fixed": fixed,
                "broken": broken,
                "net": net,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "safe_precision": safe_precision,
                "harm_rate": harm_rate,
            }
            stats["score"] = score(stats)
            outs.append(stats)

    return outs

def calc_json(fp, x):
    outs = []

    if not isinstance(x, dict):
        return outs

    # 直接扫 aggregate json
    keys = set(x.keys())
    for prefix in PREFIXES:
        fixed_k = field(prefix, "fixed")
        broken_k = field(prefix, "broken")
        changed_k = field(prefix, "changed")

        if fixed_k in keys and broken_k in keys:
            fixed = int(x[fixed_k])
            broken = int(x[broken_k])
            changed = int(x.get(changed_k, 0)) if changed_k in keys else None
            net = fixed - broken

            stats = {
                "file": str(fp),
                "kind": "json",
                "prefix": prefix or "plain",
                "current_key": "-",
                "n": x.get("n_eval", x.get("n_samples", x.get("n_resampled"))),
                "current_wrong": x.get("current_wrong"),
                "current_acc_target": x.get("current_acc_on_eval", x.get("current_acc")),
                "final_acc_target": x.get("final_acc_on_eval", x.get("acc_on_resampled")),
                "changed": changed,
                "fixed": fixed,
                "broken": broken,
                "net": net,
                "precision": fixed / changed if changed else None,
                "recall": None,
                "f1": None,
                "safe_precision": fixed / (fixed + broken) if fixed + broken else None,
                "harm_rate": broken / changed if changed else None,
            }
            stats["score"] = score(stats)
            outs.append(stats)

    return outs

def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)

all_stats = []

for root in ["outputs", "data/processed"]:
    p = Path(root)
    if not p.exists():
        continue

    for fp in p.rglob("*"):
        if not fp.is_file():
            continue

        s = str(fp).lower()
        if "svamp" not in s:
            continue
        if fp.suffix == ".jsonl":
            rows = read_jsonl(fp)
            if rows:
                all_stats.extend(calc_jsonl(fp, rows))
        elif fp.suffix == ".json":
            x = read_json(fp)
            if x is not None:
                all_stats.extend(calc_json(fp, x))

all_stats = sorted(all_stats, key=lambda r: (-r["score"], r["file"], r["prefix"]))

out = Path("outputs/logs/final_summaries/svamp_final_variant_scan.md")
out.parent.mkdir(parents=True, exist_ok=True)

lines = []
lines.append("# SVAMP final variant scan\n")
lines.append("Target: fixed=8, broken=1, changed=12, net=7\n")
lines.append("| score | file | kind | prefix | current_key | n | current_wrong | current_acc_target | final_acc_target | changed | fixed | broken | net | precision | recall | F1 | safe_precision | harm_rate |")
lines.append("|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in all_stats[:80]:
    lines.append(
        f"| {r['score']} | `{r['file']}` | {r['kind']} | {r['prefix']} | {r['current_key']} | "
        f"{fmt(r['n'])} | {fmt(r['current_wrong'])} | {fmt(r['current_acc_target'])} | {fmt(r['final_acc_target'])} | "
        f"{fmt(r['changed'])} | {fmt(r['fixed'])} | {fmt(r['broken'])} | {fmt(r['net'])} | "
        f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['safe_precision'])} | {fmt(r['harm_rate'])} |"
    )

out.write_text("\n".join(lines), encoding="utf-8")
print(out.read_text(encoding="utf-8"))
print("\nsaved:", out)
