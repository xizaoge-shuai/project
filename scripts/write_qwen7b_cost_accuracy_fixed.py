import json
import re
import glob
from pathlib import Path

def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        return []
    rows = []
    for line in open(p, encoding="utf-8", errors="ignore"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def nlines(fp):
    p = Path(fp)
    if not p.exists():
        return 0
    return sum(1 for x in open(p, encoding="utf-8", errors="ignore") if x.strip())

def token_proxy(s):
    if s is None:
        return 0
    toks = re.findall(r"\d+\.\d+|\d+|[A-Za-z]+|[^\sA-Za-z0-9]", str(s))
    return len(toks)

def row_text(r):
    keys = ["trajectory", "text", "reasoning", "completion", "response", "output", "generated_text", "final_answer", "answer"]
    parts = []
    for k in keys:
        v = r.get(k)
        if v is None:
            continue
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        v = str(v)
        if v.strip():
            parts.append(v)
    return "\n".join(parts)

def files_tokens(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    total_rows = 0
    total_tok = 0
    for fp in files:
        rows = read_jsonl(fp)
        total_rows += len(rows)
        total_tok += sum(token_proxy(row_text(r)) for r in rows)
    return files, total_rows, total_tok

ROWS = [
    {
        "dataset": "GSM8K",
        "setting": "margin040_selective_origmaj_top7",
        "n_samples": 1319,
        "base_acc": 0.8886,
        "final_acc": 0.9212,
        "target_n": 210,
        "extra_per_target": 12,   # extra4 × 3 seeds
        "token_proxy_override": 486.6,
        "fixed": 15,
        "broken": 1,
        "net": 14,
        "patterns": [
            "outputs/predictions/selective_resample_gsm8k_full_margin040_extra4_seed42_diag.jsonl",
            "outputs/predictions/selective_resample_gsm8k_full_margin040_extra4_seed101_diag.jsonl",
            "outputs/predictions/selective_resample_gsm8k_full_margin040_extra4_seed202_diag.jsonl",
        ],
    },
    {
        "dataset": "SVAMP",
        "setting": "currentkeep2+origmaj2",
        "n_samples": 300,
        "base_acc": 0.9000,
        "final_acc": 0.9233,
        "target_n": 81,
        "extra_per_target": 12,   # extra4 × 3 seeds
        "fixed": 8,
        "broken": 1,
        "net": 7,
        "patterns": [
            "outputs/predictions/cross_svamp_full300_has_disagreement_extra4_seed42.jsonl",
            "outputs/predictions/cross_svamp_full300_has_disagreement_extra4_seed101.jsonl",
            "outputs/predictions/cross_svamp_full300_has_disagreement_extra4_seed202.jsonl",
        ],
    },
    {
        "dataset": "ASDiv-numeric",
        "setting": "numeric_full_total3_seed2_margin0",
        "n_samples": 2249,
        "base_acc": 0.8671,
        "final_acc": 0.9471,
        "target_n": 997,
        "extra_per_target": 12,   # extra4 × 3 seeds
        "fixed": 193,
        "broken": 13,
        "net": 180,
        "patterns": [
            "data/processed/trajectories/asdiv/extra_numeric_full_seed42.jsonl",
            "data/processed/trajectories/asdiv/extra_numeric_full_seed101.jsonl",
            "data/processed/trajectories/asdiv/extra_numeric_full_seed202.jsonl",
        ],
    },
    {
        "dataset": "MathQA",
        "setting": "mathqa_500_total2_seed2_margin0",
        "n_samples": 500,
        "base_acc": 0.7780,
        "final_acc": 0.8580,
        "target_n": 224,
        "extra_per_target": 12,   # 12 extra seeds / target
        "fixed": 49,
        "broken": 9,
        "net": 40,
        "patterns": [
            "data/processed/trajectories/mathqa/mathqa_500_extra_seed303.jsonl",
            "data/processed/trajectories/mathqa/mathqa_500_extra_seed404.jsonl",
            "data/processed/trajectories/mathqa/mathqa_500_extra_seed505.jsonl",
            "data/processed/trajectories/mathqa/mathqa_500_extra_seed606.jsonl",
            "data/processed/trajectories/mathqa/mathqa_500_extra_seed707.jsonl",
            "data/processed/trajectories/mathqa/mathqa_500_extra_seed808.jsonl",
            "data/processed/trajectories/mathqa/mathqa_500_extra_seed909.jsonl",
            "data/processed/trajectories/mathqa/mathqa_500_extra_seed1001.jsonl",
            "data/processed/trajectories/mathqa/mathqa_500_extra_seed1102.jsonl",
            "data/processed/trajectories/mathqa/mathqa_500_extra_seed1203.jsonl",
            "data/processed/trajectories/mathqa/mathqa_500_extra_seed1304.jsonl",
            "data/processed/trajectories/mathqa/mathqa_500_extra_seed1405.jsonl",
        ],
    },
    {
        "dataset": "MATH500",
        "setting": "math500_all244_total2_seed2",
        "n_samples": 500,
        "base_acc": 0.6520,
        "final_acc": 0.7140,
        "target_n": 244,
        "extra_per_target": 2,
        "fixed": 38,
        "broken": 7,
        "net": 31,
        "patterns": [
            "data/processed/trajectories/math500/extra_clean_v3_has_disagreement_all_seed42_clean_v3.jsonl",
            "data/processed/trajectories/math500/extra_clean_v3_has_disagreement_all_seed101_clean_v3.jsonl",
        ],
    },
]

out_rows = []
audit = []

for r in ROWS:
    files, raw_rows, raw_tok = files_tokens(r["patterns"])
    extra_calls = r["target_n"] * r["extra_per_target"]
    gain = r["final_acc"] - r["base_acc"]
    target_rate = r["target_n"] / r["n_samples"]
    calls_per_sample = extra_calls / r["n_samples"]
    precision = r["fixed"] / (r["fixed"] + (r["net"] - r["fixed"] + r["broken"]) if False else r["changed"]) if "changed" in r else None
    # 论文表里更常用 harm rate = broken / changed；这里没有 changed 时只输出 fixed/broken/net。
    tok_per_sample = r.get("token_proxy_override")
    if tok_per_sample is None and raw_tok > 0:
        tok_per_sample = raw_tok / r["n_samples"]

    out = {
        "dataset": r["dataset"],
        "setting": r["setting"],
        "n_samples": r["n_samples"],
        "base_acc": r["base_acc"],
        "final_acc": r["final_acc"],
        "acc_gain": gain,
        "triggered_eval": r["target_n"],
        "target_rate": target_rate,
        "extra_per_target": r["extra_per_target"],
        "extra_calls": extra_calls,
        "extra_calls_per_sample": calls_per_sample,
        "token_proxy_per_sample": tok_per_sample,
        "fixed": r["fixed"],
        "broken": r["broken"],
        "net": r["net"],
        "raw_files_found": len(files),
        "raw_rows_found": raw_rows,
    }
    out_rows.append(out)
    audit.append({
        "dataset": r["dataset"],
        "patterns": r["patterns"],
        "matched_files": files,
        "raw_rows_found": raw_rows,
        "raw_output_token_proxy": raw_tok,
    })

Path("outputs/tables").mkdir(parents=True, exist_ok=True)
Path("outputs/logs/final_summaries").mkdir(parents=True, exist_ok=True)

json.dump(out_rows, open("outputs/tables/qwen7b_cost_accuracy_fixed.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(audit, open("outputs/tables/qwen7b_cost_accuracy_fixed_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)

md = []
md.append("# Qwen7B fixed cost-accuracy table\n")
md.append("| Dataset | Setting | Base | Final | ΔAcc | Triggered/Eval | Target Rate | Extra/Target | Extra Calls | Extra Calls/Sample | Token Proxy/Sample | fixed | broken | net | Raw files |\n")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
for r in out_rows:
    md.append(
        f"| {r['dataset']} | {r['setting']} | {fmt(r['base_acc'])} | {fmt(r['final_acc'])} | {fmt(r['acc_gain'])} | "
        f"{r['triggered_eval']} | {fmt(r['target_rate'])} | {r['extra_per_target']} | {r['extra_calls']} | "
        f"{fmt(r['extra_calls_per_sample'])} | {fmt(r['token_proxy_per_sample'])} | "
        f"{r['fixed']} | {r['broken']} | {r['net']:+d} | {r['raw_files_found']} |\n"
    )

txt = "".join(md)
Path("outputs/tables/qwen7b_cost_accuracy_fixed.md").write_text(txt, encoding="utf-8")
Path("outputs/logs/final_summaries/qwen7b_cost_accuracy_fixed.md").write_text(txt, encoding="utf-8")

print(txt)
print("saved: outputs/tables/qwen7b_cost_accuracy_fixed.md")
print("saved: outputs/tables/qwen7b_cost_accuracy_fixed.json")
print("saved: outputs/tables/qwen7b_cost_accuracy_fixed_audit.json")
