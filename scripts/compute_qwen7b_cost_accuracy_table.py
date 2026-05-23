import json
import re
import glob
import csv
from pathlib import Path

# Qwen7B 已有主结果。数值来自你当前最终表/PRF 表；脚本会尽量用已有 json/jsonl 覆盖 target_n。
SPECS = [
    {
        "dataset": "GSM8K",
        "setting": "margin040_selective_origmaj_top7",
        "n_samples": 1319,
        "base_acc": 0.8886,
        "final_acc": 0.9212,
        "changed": 22,
        "fixed": 15,
        "broken": 1,
        "net": 14,
        "default_target_n": 210,
        "extra_per_target": 4,
        "metric_globs": [
            "outputs/metrics/resample_confirm_margin040_extra4_seedaware_total3_seed2_currentkeep2_selective_origmaj_top7.json",
            "outputs/metrics/*gsm8k*selective*top7*.json",
        ],
        "pred_globs": [
            "outputs/final_selected_results/final_gsm8k_margin040_selective_origmaj_top7_predictions.jsonl",
            "outputs/predictions/*gsm8k*selective*top7*.jsonl",
        ],
        "raw_extra_globs": [
            "data/processed/trajectories/gsm8k/*extra*.jsonl",
            "data/processed/trajectories/*gsm8k*extra*.jsonl",
            "outputs/predictions/*gsm8k*extra*.jsonl",
        ],
    },
    {
        "dataset": "ASDiv-numeric",
        "setting": "numeric_full_total6_seed3_margin2",
        "n_samples": 2249,
        "base_acc": 0.8671,
        "final_acc": 0.9471,
        "changed": 254,
        "fixed": 193,
        "broken": 13,
        "net": 180,
        "default_target_n": 997,
        "extra_per_target": 6,
        "metric_globs": [
            "outputs/metrics/asdiv_numeric_extra_confirm/*total6*seed3*margin2*.json",
            "outputs/metrics/*asdiv*numeric*total6*seed3*margin2*.json",
            "outputs/metrics/*asdiv*confirm*.json",
        ],
        "pred_globs": [
            "outputs/predictions/asdiv_numeric_extra_confirm/numeric_full_total6_seed3_margin2.jsonl",
            "outputs/predictions/asdiv_numeric_extra_confirm/*.jsonl",
            "outputs/final_selected_results/*asdiv*.jsonl",
        ],
        "raw_extra_globs": [
            "data/processed/trajectories/asdiv/*extra*.jsonl",
            "outputs/predictions/asdiv_numeric_extra_confirm/*.jsonl",
        ],
    },
    {
        "dataset": "MathQA",
        "setting": "scale_extra_confirm",
        "n_samples": 500,
        "base_acc": 0.7780,
        "final_acc": 0.8580,
        "changed": 84,
        "fixed": 49,
        "broken": 9,
        "net": 40,
        "default_target_n": None,
        "extra_per_target": None,
        "metric_globs": [
            "outputs/metrics/mathqa_scale_extra_confirm*.json",
            "outputs/metrics/*mathqa*confirm*.json",
            "outputs/metrics/*mathqa*.json",
        ],
        "pred_globs": [
            "outputs/predictions/mathqa_scale_extra_confirm*.jsonl",
            "outputs/predictions/*mathqa*confirm*.jsonl",
            "outputs/final_selected_results/*mathqa*.jsonl",
        ],
        "raw_extra_globs": [
            "data/processed/trajectories/mathqa/*extra*.jsonl",
            "outputs/predictions/*mathqa*extra*.jsonl",
            "outputs/predictions/*mathqa*confirm*.jsonl",
        ],
    },
    {
        "dataset": "MATH500",
        "setting": "old_total2_seed2_or_v2_best",
        "n_samples": 500,
        "base_acc": 0.6520,
        "final_acc": 0.7140,
        "changed": 96,
        "fixed": 38,
        "broken": 7,
        "net": 31,
        "default_target_n": 244,
        "extra_per_target": 2,
        "metric_globs": [
            "outputs/metrics/*math500*total2*seed2*.json",
            "outputs/metrics/*math500*v2*best*.json",
            "outputs/metrics/*math500*confirm*.json",
            "outputs/metrics/*math500*.json",
        ],
        "pred_globs": [
            "outputs/final_selected_results/*math500*.jsonl",
            "outputs/predictions/*math500*total2*seed2*.jsonl",
            "outputs/predictions/*math500*v2*best*.jsonl",
            "outputs/predictions/*math500*confirm*.jsonl",
        ],
        "raw_extra_globs": [
            "data/processed/trajectories/math500/*extra*.jsonl",
            "outputs/predictions/*math500*extra*.jsonl",
            "outputs/predictions/*math500*confirm*.jsonl",
        ],
    },
    {
        "dataset": "SVAMP",
        "setting": "final_guard",
        "n_samples": 300,
        "base_acc": None,
        "final_acc": 0.9233,
        "changed": 10,
        "fixed": 4,
        "broken": 2,
        "net": 2,
        "default_target_n": 81,
        "extra_per_target": None,
        "metric_globs": [
            "outputs/metrics/*svamp*final_guard*.json",
            "outputs/metrics/*svamp*confirm*.json",
            "outputs/metrics/*svamp*.json",
        ],
        "pred_globs": [
            "outputs/final_selected_results/final_cross_svamp_origmaj2_predictions.jsonl",
            "outputs/final_selected_results/*svamp*.jsonl",
            "outputs/predictions/*svamp*.jsonl",
        ],
        "raw_extra_globs": [
            "data/processed/trajectories/svamp/*extra*.jsonl",
            "outputs/predictions/*svamp*extra*.jsonl",
        ],
    },
]

TEXT_KEYS = [
    "trajectory", "text", "reasoning", "completion", "response",
    "output", "generated_text", "final_answer", "answer"
]
PROMPT_KEYS = ["question", "problem", "input", "prompt", "context"]


def nlines(fp):
    p = Path(fp)
    if not p.exists():
        return 0
    return sum(1 for x in open(p, encoding="utf-8", errors="ignore") if x.strip())


def read_json(fp):
    try:
        return json.load(open(fp, encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(fp, limit=None):
    p = Path(fp)
    if not p.exists():
        return []
    rows = []
    for line in open(p, encoding="utf-8", errors="ignore"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
        if limit and len(rows) >= limit:
            break
    return rows


def expand_globs(globs):
    out = []
    for g in globs:
        out.extend(glob.glob(g))
    return sorted(set(out))


def token_proxy(s):
    if s is None:
        return 0
    s = str(s)
    toks = re.findall(r"\d+\.\d+|\d+|[A-Za-z]+|[^\sA-Za-z0-9]", s)
    return len(toks)


def row_text(row, keys):
    parts = []
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        v = str(v)
        if v.strip():
            parts.append(v)
    return "\n".join(parts)


def jsonl_token_proxy(fp, keys):
    total = 0
    for r in read_jsonl(fp):
        total += token_proxy(row_text(r, keys))
    return total


def first_num_field(obj, keys):
    if not obj:
        return None
    for k in keys:
        v = obj.get(k)
        if isinstance(v, (int, float)):
            return v
    return None


def pick_metric(spec):
    files = expand_globs(spec["metric_globs"])
    scored = []
    for fp in files:
        x = read_json(fp)
        if not x:
            continue
        final = first_num_field(x, ["estimated_global_acc", "final_acc", "acc", "accuracy"])
        fixed = x.get("fixed")
        broken = x.get("broken")
        score = 0
        if final is not None:
            score -= abs(final - spec["final_acc"])
        if fixed == spec.get("fixed"):
            score += 1
        if broken == spec.get("broken"):
            score += 1
        scored.append((score, fp, x))
    if scored:
        return sorted(scored, key=lambda z: z[0])[-1][1:]
    return None, None


def pick_pred(spec):
    files = expand_globs(spec["pred_globs"])
    if not files:
        return None
    # 优先 exact setting 命中，其次最近修改
    setting = spec["setting"].lower()
    files2 = [f for f in files if setting.replace("_", "").lower() in f.replace("_", "").lower()]
    if files2:
        return sorted(files2, key=lambda f: Path(f).stat().st_mtime)[-1]
    return sorted(files, key=lambda f: Path(f).stat().st_mtime)[-1]


def infer_target_n(spec, metric, pred_fp):
    keys = [
        "n_eval", "n_resampled", "n_selected", "n_targets", "target_samples",
        "n_has_disagreement", "has_disagreement", "n_reflect", "num_triggered"
    ]
    v = first_num_field(metric, keys)
    if v is not None:
        return int(v), "metric"
    if pred_fp:
        nl = nlines(pred_fp)
        if nl > 0:
            return nl, "pred_jsonl_lines"
    if spec.get("default_target_n") is not None:
        return int(spec["default_target_n"]), "default_from_final_log"
    return None, "missing"


def infer_extra_per_target(spec, metric):
    if spec.get("extra_per_target") is not None:
        return spec["extra_per_target"], "manual_setting"
    for k in ["extra_per_target", "n_extra", "extra_budget", "num_extra", "total", "total_samples"]:
        v = first_num_field(metric, [k])
        if v:
            return int(v), f"metric:{k}"
    return None, "missing"


rows = []
audit = []

for spec in SPECS:
    metric_fp, metric = pick_metric(spec)
    pred_fp = pick_pred(spec)

    target_n, target_source = infer_target_n(spec, metric, pred_fp)
    extra_per_target, extra_source = infer_extra_per_target(spec, metric)

    raw_files = expand_globs(spec["raw_extra_globs"])
    raw_files = [f for f in raw_files if Path(f).exists()]

    observed_raw_rows = sum(nlines(f) for f in raw_files)
    observed_raw_output_tokens = sum(jsonl_token_proxy(f, TEXT_KEYS) for f in raw_files)

    pred_rows = nlines(pred_fp) if pred_fp else 0
    pred_output_tokens = jsonl_token_proxy(pred_fp, TEXT_KEYS) if pred_fp else 0
    pred_prompt_tokens = jsonl_token_proxy(pred_fp, PROMPT_KEYS) if pred_fp else 0

    expected_extra_calls = None
    if target_n is not None and extra_per_target is not None:
        expected_extra_calls = target_n * extra_per_target

    calls_per_sample = None
    if expected_extra_calls is not None:
        calls_per_sample = expected_extra_calls / spec["n_samples"]

    acc_gain = None
    if spec["base_acc"] is not None and spec["final_acc"] is not None:
        acc_gain = spec["final_acc"] - spec["base_acc"]

    precision = spec["fixed"] / spec["changed"] if spec["changed"] else None
    harm_rate = spec["broken"] / spec["changed"] if spec["changed"] else None

    row = {
        "model": "Qwen2.5-7B-Instruct",
        "dataset": spec["dataset"],
        "setting": spec["setting"],
        "n_samples": spec["n_samples"],
        "base_acc": spec["base_acc"],
        "final_acc": spec["final_acc"],
        "acc_gain": acc_gain,
        "changed": spec["changed"],
        "fixed": spec["fixed"],
        "broken": spec["broken"],
        "net": spec["net"],
        "precision": precision,
        "harm_rate": harm_rate,
        "target_n": target_n,
        "target_source": target_source,
        "target_rate": (target_n / spec["n_samples"]) if target_n is not None else None,
        "extra_per_target": extra_per_target,
        "extra_source": extra_source,
        "expected_extra_calls": expected_extra_calls,
        "extra_calls_per_sample": calls_per_sample,
        "observed_raw_extra_rows": observed_raw_rows if observed_raw_rows else None,
        "observed_raw_output_token_proxy": observed_raw_output_tokens if observed_raw_output_tokens else None,
        "selected_pred_rows": pred_rows if pred_rows else None,
        "selected_pred_output_token_proxy": pred_output_tokens if pred_output_tokens else None,
        "selected_pred_prompt_token_proxy": pred_prompt_tokens if pred_prompt_tokens else None,
        "metric_file": metric_fp or "",
        "prediction_file": pred_fp or "",
        "raw_extra_files_found": len(raw_files),
    }
    rows.append(row)

    audit.append({
        "dataset": spec["dataset"],
        "metric_file": metric_fp,
        "prediction_file": pred_fp,
        "raw_extra_files": raw_files[:30],
        "raw_extra_files_count": len(raw_files),
    })


out_dir = Path("outputs/logs/final_summaries")
out_dir.mkdir(parents=True, exist_ok=True)

csv_fp = out_dir / "qwen7b_cost_accuracy_table.csv"
json_fp = out_dir / "qwen7b_cost_accuracy_table.json"
md_fp = out_dir / "qwen7b_cost_accuracy_table.md"
audit_fp = out_dir / "qwen7b_cost_accuracy_audit.json"

fields = list(rows[0].keys())
with open(csv_fp, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

json.dump(rows, open(json_fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(audit, open(audit_fp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)

cols = [
    "dataset", "n_samples", "base_acc", "final_acc", "acc_gain",
    "target_n", "target_rate", "extra_per_target",
    "expected_extra_calls", "extra_calls_per_sample",
    "changed", "fixed", "broken", "net",
    "precision", "harm_rate",
    "metric_file", "prediction_file"
]

with open(md_fp, "w", encoding="utf-8") as f:
    f.write("# Qwen7B cost-accuracy table\n\n")
    f.write("说明：`expected_extra_calls` 是按 target 数和 extra budget 得到的调用量；token proxy 只在能找到 raw/pred jsonl 时统计。若 raw extra 文件未完整保留，表中会保留 metric/prediction 文件以便追溯。\n\n")
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            if c.endswith("_file"):
                vals.append(str(v).replace("|", "\\|"))
            else:
                vals.append(fmt(v))
        f.write("| " + " | ".join(vals) + " |\n")

print("saved:", md_fp)
print("saved:", csv_fp)
print("saved:", json_fp)
print("saved:", audit_fp)
