import json, re
from pathlib import Path

OUT = Path("outputs/logs/final_summaries")
OUT.mkdir(parents=True, exist_ok=True)
MET = Path("outputs/metrics/final_ablation")
MET.mkdir(parents=True, exist_ok=True)

ITEMS = [
    ("margin030_currentkeep2",
     "outputs/final_selected_results/final_gsm8k_margin030_extra4_currentkeep2_predictions.jsonl",
     0.8886, 0.9196, 107, 18, 13, 1, 12),
    ("margin040_currentkeep2",
     "outputs/final_selected_results/final_gsm8k_margin040_extra4_currentkeep2_predictions.jsonl",
     0.8886, 0.9204, 210, 28, 17, 4, 13),
    ("margin040_selective_origmaj_top7",
     "outputs/final_selected_results/final_gsm8k_margin040_selective_origmaj_top7_predictions.jsonl",
     0.8886, 0.9212, 210, 22, 15, 1, 14),
]

def norm(x):
    s = str(x or "").replace(",", "").strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return s.lower()
    y = nums[-1]
    return y.rstrip("0").rstrip(".") if "." in y else y

def ok(a, g):
    return norm(a) == norm(g)

def div(a,b):
    return a/b if b else 0.0

def f1(p,r):
    return 2*p*r/(p+r) if p+r else 0.0

def get(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None

def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        return []
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]

rows_out = []

for setting, fp, base, final, n_eval_known, changed_known, fixed_known, broken_known, net_known in ITEMS:
    rows = read_jsonl(fp)

    cur_vals, fin_vals, chg_vals, fix_vals, bro_vals = [], [], [], [], []

    for r in rows:
        gold = get(r, ["gold", "gold_answer", "answer", "target"])
        cur_ans = get(r, ["current", "current_answer", "majority_answer", "orig_answer"])
        fin_ans = get(r, ["final", "final_answer", "pred", "pred_answer", "chosen_answer"])

        cur_ok = get(r, ["cur_ok", "current_ok", "majority_ok", "current_correct"])
        fin_ok = get(r, ["final_ok", "pred_ok", "final_correct"])

        if cur_ok is None and gold is not None and cur_ans is not None:
            cur_ok = int(ok(cur_ans, gold))
        if fin_ok is None and gold is not None and fin_ans is not None:
            fin_ok = int(ok(fin_ans, gold))

        if cur_ok is not None:
            cur_ok = int(cur_ok)
            cur_vals.append(cur_ok)
        if fin_ok is not None:
            fin_ok = int(fin_ok)
            fin_vals.append(fin_ok)

        changed = get(r, ["changed"])
        if changed is None and cur_ans is not None and fin_ans is not None:
            changed = int(norm(cur_ans) != norm(fin_ans))

        fixed = get(r, ["fixed"])
        broken = get(r, ["broken"])

        if fixed is None and cur_ok is not None and fin_ok is not None:
            fixed = int(cur_ok == 0 and fin_ok == 1)
        if broken is None and cur_ok is not None and fin_ok is not None:
            broken = int(cur_ok == 1 and fin_ok == 0)

        if changed is not None: chg_vals.append(int(changed))
        if fixed is not None: fix_vals.append(int(fixed))
        if broken is not None: bro_vals.append(int(broken))

    n_eval = len(rows) if rows else n_eval_known
    current_wrong = None
    current_acc = None
    recall = None
    cf1 = None

    if cur_vals:
        current_wrong = len(cur_vals) - sum(cur_vals)
        current_acc = sum(cur_vals) / len(cur_vals)

    # 如果 per-case 统计不完整，就使用 known 的 changed/fixed/broken
    changed = sum(chg_vals) if chg_vals else changed_known
    fixed = sum(fix_vals) if fix_vals else fixed_known
    broken = sum(bro_vals) if bro_vals else broken_known
    net = fixed - broken

    precision = div(fixed, changed)
    safe_precision = div(fixed, fixed + broken)
    harm_rate = div(broken, changed)

    if current_wrong is not None:
        recall = div(fixed, current_wrong)
        cf1 = f1(precision, recall)

    global_wrong = round(1319 * (1 - base))
    global_recall = div(fixed, global_wrong)
    global_f1 = f1(precision, global_recall)

    rows_out.append({
        "setting": setting,
        "source": fp,
        "source_exists": Path(fp).exists(),
        "base_acc": base,
        "final_acc": final,
        "n_eval": n_eval,
        "current_acc": current_acc,
        "current_wrong": current_wrong,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "precision": precision,
        "recall": recall,
        "f1": cf1,
        "safe_precision": safe_precision,
        "harm_rate": harm_rate,
        "global_wrong": global_wrong,
        "global_recall": global_recall,
        "global_f1": global_f1,
    })

with open(MET/"gsm8k_prf_from_final_selected.json", "w", encoding="utf-8") as f:
    json.dump(rows_out, f, ensure_ascii=False, indent=2)

lines = []
lines.append("# GSM8K PRF from final_selected_results\n")
lines.append("| Setting | source_exists | Base | Final | n_eval | current_acc | current_wrong | changed | fixed | broken | net | precision | recall | F1 | safe_precision | harm_rate |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows_out:
    def fmt(x):
        return "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
    lines.append(
        f"| {r['setting']} | {r['source_exists']} | {fmt(r['base_acc'])} | {fmt(r['final_acc'])} | "
        f"{r['n_eval']} | {fmt(r['current_acc'])} | {fmt(r['current_wrong'])} | "
        f"{r['changed']} | {r['fixed']} | {r['broken']} | {r['net']} | "
        f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | "
        f"{fmt(r['safe_precision'])} | {fmt(r['harm_rate'])} |"
    )

lines.append("\n## Global fallback\n")
lines.append("| Setting | global_wrong | fixed | precision | global_recall | global_F1 |")
lines.append("|---|---:|---:|---:|---:|---:|")
for r in rows_out:
    def fmt(x):
        return "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))
    lines.append(f"| {r['setting']} | {r['global_wrong']} | {r['fixed']} | {fmt(r['precision'])} | {fmt(r['global_recall'])} | {fmt(r['global_f1'])} |")

md = "\n".join(lines)
(OUT/"gsm8k_prf_from_final_selected.md").write_text(md, encoding="utf-8")
print(md)
