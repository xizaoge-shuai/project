import json
import re
import string
from pathlib import Path
from collections import defaultdict

OUT = Path("outputs/logs/final_summaries")
OUT.mkdir(parents=True, exist_ok=True)

SPECS = [
    {
        "name": "SVAMP",
        "tokens": ["svamp"],
        "expected": {"acc": 0.9233, "fixed": 8, "broken": 1, "net": 7, "changed": 12},
        "task_type": "numeric",
    },
    {
        "name": "MATH500-best",
        "tokens": ["math500"],
        "expected": {"acc": 0.7160, "fixed": 37, "broken": 5, "net": 32, "changed": 92},
        "task_type": "numeric",
    },
    {
        "name": "MATH500-balanced",
        "tokens": ["math500"],
        "expected": {"acc": 0.7080, "fixed": 31, "broken": 3, "net": 28, "changed": 60},
        "task_type": "numeric",
    },
    {
        "name": "MATH500-old-total2-seed2",
        "tokens": ["math500"],
        "expected": {"acc": 0.7140, "fixed": 38, "broken": 7, "net": 31, "changed": 96},
        "task_type": "numeric",
    },
    {
        "name": "BBH-logical5-summary",
        "tokens": ["bbh", "logical_deduction_five"],
        "expected": {"acc": 0.7500, "fixed": 19, "broken": 2, "net": 17, "changed": 28},
        "task_type": "choice",
    },
    {
        "name": "BBH-logical5-percase",
        "tokens": ["bbh", "logical_deduction_five"],
        "expected": {"acc": 0.7500, "fixed": 18, "broken": 1, "net": 17, "changed": 28},
        "task_type": "choice",
    },
]

BAD_DATASET_TOKENS = {
    "SVAMP": ["asdiv", "mathqa", "math500", "bbh", "hotpot", "strategy", "gsm8k"],
    "MATH500-best": ["asdiv", "mathqa", "svamp", "bbh", "hotpot", "strategy", "gsm8k"],
    "MATH500-balanced": ["asdiv", "mathqa", "svamp", "bbh", "hotpot", "strategy", "gsm8k"],
    "MATH500-old-total2-seed2": ["asdiv", "mathqa", "svamp", "bbh", "hotpot", "strategy", "gsm8k"],
    "BBH-logical5-summary": ["asdiv", "mathqa", "math500", "svamp", "hotpot", "strategy", "gsm8k"],
    "BBH-logical5-percase": ["asdiv", "mathqa", "math500", "svamp", "hotpot", "strategy", "gsm8k"],
}

CURRENT_KEYS = ["current_ok", "cur_ok", "majority_ok", "current_correct", "before_ok"]
FINAL_KEYS = ["final_ok", "pred_ok", "after_ok", "final_correct"]
GOLD_KEYS = ["gold", "gold_answer", "answer", "target", "gold_norm"]
CURRENT_ANSWER_KEYS = ["current_answer", "current", "majority_answer", "before_answer", "orig_answer"]
FINAL_ANSWER_KEYS = ["final_answer", "final", "pred", "pred_answer", "chosen_answer", "after_answer"]


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


def get(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def to_int(x):
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    try:
        return int(x)
    except Exception:
        s = str(x).strip().lower()
        if s in {"true", "yes", "correct"}:
            return 1
        if s in {"false", "no", "wrong", "incorrect"}:
            return 0
    return None


def norm_numeric(x):
    s = str(x or "").replace(",", "").replace("$", "").strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return " ".join(s.lower().split())
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y


def norm_choice(x):
    raw = str(x or "")
    cands = re.findall(r"final answer\s*[:：]\s*([^\n\|]+)", raw, flags=re.I) + [raw]
    for c in reversed(cands):
        m = re.search(r"\(([A-Ea-e])\)", c)
        if m:
            return m.group(1).lower()
        m = re.search(r"\boption\s*([A-Ea-e])\b", c, flags=re.I)
        if m:
            return m.group(1).lower()
        m = re.search(r"^\s*([A-Ea-e])[\)\.\:]\s*", c)
        if m:
            return m.group(1).lower()
        if len(c.strip()) <= 5:
            m = re.search(r"\b([A-Ea-e])\b", c)
            if m:
                return m.group(1).lower()
    s = raw.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def norm(x, task_type):
    return norm_choice(x) if task_type == "choice" else norm_numeric(x)


def compute_stats(rows, task_type):
    if not rows:
        return None

    n = len(rows)
    cur_vals, fin_vals, fixed_vals, broken_vals, changed_vals = [], [], [], [], []

    for r in rows:
        cur_ok = to_int(get(r, CURRENT_KEYS))
        fin_ok = to_int(get(r, FINAL_KEYS))

        gold = get(r, GOLD_KEYS)
        cur_ans = get(r, CURRENT_ANSWER_KEYS)
        fin_ans = get(r, FINAL_ANSWER_KEYS)

        if cur_ok is None and gold is not None and cur_ans is not None:
            cur_ok = int(norm(cur_ans, task_type) == norm(gold, task_type))
        if fin_ok is None and gold is not None and fin_ans is not None:
            fin_ok = int(norm(fin_ans, task_type) == norm(gold, task_type))

        fixed = to_int(r.get("fixed"))
        broken = to_int(r.get("broken"))
        changed = to_int(r.get("changed"))

        if fixed is None and cur_ok is not None and fin_ok is not None:
            fixed = int(cur_ok == 0 and fin_ok == 1)
        if broken is None and cur_ok is not None and fin_ok is not None:
            broken = int(cur_ok == 1 and fin_ok == 0)

        if changed is None and cur_ans is not None and fin_ans is not None:
            changed = int(norm(cur_ans, task_type) != norm(fin_ans, task_type))

        if cur_ok is not None:
            cur_vals.append(cur_ok)
        if fin_ok is not None:
            fin_vals.append(fin_ok)
        if fixed is not None:
            fixed_vals.append(fixed)
        if broken is not None:
            broken_vals.append(broken)
        if changed is not None:
            changed_vals.append(changed)

    if not cur_vals and not fixed_vals:
        return None

    current_wrong = None
    current_acc = None
    final_acc = None

    if cur_vals:
        current_acc = sum(cur_vals) / len(cur_vals)
        current_wrong = len(cur_vals) - sum(cur_vals)

    if fin_vals:
        final_acc = sum(fin_vals) / len(fin_vals)

    fixed = sum(fixed_vals) if fixed_vals else None
    broken = sum(broken_vals) if broken_vals else None
    changed = sum(changed_vals) if changed_vals else None
    net = fixed - broken if fixed is not None and broken is not None else None

    precision = fixed / changed if fixed is not None and changed else None
    recall = fixed / current_wrong if fixed is not None and current_wrong else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall > 0 else None
    safe_precision = fixed / (fixed + broken) if fixed is not None and broken is not None and fixed + broken > 0 else None
    harm_rate = broken / changed if broken is not None and changed else None

    return {
        "n_rows": n,
        "current_acc": current_acc,
        "final_acc": final_acc,
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
        "has_current_wrong": current_wrong is not None,
    }


def candidate_files():
    roots = ["outputs/predictions", "outputs/final_selected_results", "outputs/metrics", "outputs/logs"]
    files = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        for fp in p.rglob("*.jsonl"):
            files.append(fp)
    return sorted(files)


def path_ok(fp, spec):
    s = str(fp).lower()
    if any(bad in s for bad in BAD_DATASET_TOKENS.get(spec["name"], [])):
        return False
    return all(tok.lower() in s for tok in spec["tokens"])


def score(stats, expected):
    if stats is None:
        return -999

    sc = 0
    for k in ["fixed", "broken", "net", "changed"]:
        if expected.get(k) is not None and stats.get(k) == expected[k]:
            sc += 20
        elif expected.get(k) is not None and stats.get(k) is not None:
            sc -= abs(stats[k] - expected[k])

    if stats.get("has_current_wrong"):
        sc += 15
    return sc


def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


all_results = []

for spec in SPECS:
    rows = []
    for fp in candidate_files():
        if not path_ok(fp, spec):
            continue
        data = read_jsonl(fp)
        stats = compute_stats(data, spec["task_type"])
        if stats is None:
            continue
        rows.append({
            "spec": spec["name"],
            "file": str(fp),
            "score": score(stats, spec["expected"]),
            **stats,
        })

    rows = sorted(rows, key=lambda x: -x["score"])
    all_results.extend(rows[:10])

md = []
md.append("# Direct per-case PRF source search\n")
md.append("只搜索直接 per-case 文件；本脚本不做 acc_on_resampled/net 反推。\n")
md.append("| spec | score | file | n_rows | current_wrong | changed | fixed | broken | net | precision | recall | F1 | safe_precision | harm_rate |")
md.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in all_results:
    md.append(
        f"| {r['spec']} | {r['score']} | `{r['file']}` | {fmt(r['n_rows'])} | "
        f"{fmt(r['current_wrong'])} | {fmt(r['changed'])} | {fmt(r['fixed'])} | {fmt(r['broken'])} | {fmt(r['net'])} | "
        f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {fmt(r['safe_precision'])} | {fmt(r['harm_rate'])} |"
    )

out = OUT / "direct_percase_prf_source_search.md"
out.write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
print("\nsaved:", out)
