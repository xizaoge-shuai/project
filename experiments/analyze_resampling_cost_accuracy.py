import argparse
import json
import re
from pathlib import Path


def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.open("r", encoding="utf-8") if x.strip()]


def read_json(fp):
    p = Path(fp)
    if not p.exists():
        return None
    return json.load(p.open("r", encoding="utf-8"))


def approx_token_count(text):
    # 粗略 token proxy：英文/数字场景下 char/4 通常够做相对成本比较
    s = str(text or "")
    return max(1, int(len(s) / 4))


def row_extra_texts(row):
    xs = row.get("extra_texts")
    if isinstance(xs, list):
        return xs
    # 如果没有 extra_texts，就退回用 extra_answers 估计调用次数，但 token 记为 0
    xs = row.get("extra_answers")
    if isinstance(xs, list):
        return ["" for _ in xs]
    return []


def analyze_seed_files(seed_files):
    n_rows = 0
    n_extra_calls = 0
    total_extra_token_proxy = 0
    per_seed = []

    for fp in seed_files:
        rows = read_jsonl(fp)
        calls = 0
        toks = 0

        for r in rows:
            texts = row_extra_texts(r)
            calls += len(texts)
            toks += sum(approx_token_count(t) for t in texts if t)

        n_rows += len(rows)
        n_extra_calls += calls
        total_extra_token_proxy += toks

        per_seed.append({
            "file": fp,
            "rows": len(rows),
            "extra_calls": calls,
            "extra_token_proxy": toks,
        })

    return {
        "seed_files": per_seed,
        "total_seed_rows": n_rows,
        "extra_calls": n_extra_calls,
        "extra_token_proxy": total_extra_token_proxy,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=1319)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    methods = [
        {
            "name": "majority_voting",
            "acc": 0.8886,
            "summary": "",
            "seed_files": [],
        },
        {
            "name": "selective_judge",
            "acc": 0.9105,
            "summary": "",
            "seed_files": [],
        },
        {
            "name": "margin030_currentkeep2",
            "summary": "outputs/metrics/resample_confirm_margin030_107_extra4_seedaware_total3_seed2_currentkeep2.json",
            "seed_files": [
                "outputs/predictions/selective_resample_gsm8k_full_margin030_107_extra4_seed42_diag.jsonl",
                "outputs/predictions/selective_resample_gsm8k_full_margin030_107_extra4_seed101_diag.jsonl",
                "outputs/predictions/selective_resample_gsm8k_full_margin030_107_extra4_seed202_diag.jsonl",
            ],
        },
        {
            "name": "margin040_currentkeep2",
            "summary": "outputs/metrics/resample_confirm_margin040_extra4_seedaware_total3_seed2_currentkeep2.json",
            "seed_files": [
                "outputs/predictions/selective_resample_gsm8k_full_margin040_extra4_seed42_diag.jsonl",
                "outputs/predictions/selective_resample_gsm8k_full_margin040_extra4_seed101_diag.jsonl",
                "outputs/predictions/selective_resample_gsm8k_full_margin040_extra4_seed202_diag.jsonl",
            ],
        },
        {
            "name": "margin040_selective_origmaj_top7",
            "summary": "outputs/metrics/resample_confirm_margin040_extra4_seedaware_total3_seed2_currentkeep2_selective_origmaj_top7.json",
            "seed_files": [
                "outputs/predictions/selective_resample_gsm8k_full_margin040_extra4_seed42_diag.jsonl",
                "outputs/predictions/selective_resample_gsm8k_full_margin040_extra4_seed101_diag.jsonl",
                "outputs/predictions/selective_resample_gsm8k_full_margin040_extra4_seed202_diag.jsonl",
            ],
        },
    ]

    results = []

    for m in methods:
        summary = read_json(m.get("summary", "")) if m.get("summary") else None
        cost = analyze_seed_files(m.get("seed_files", []))

        acc = m.get("acc")
        fixed = broken = net = changed = None
        n_resampled = 0

        if summary:
            acc = float(summary.get("estimated_global_acc"))
            fixed = summary.get("fixed")
            broken = summary.get("broken")
            net = summary.get("net")
            changed = summary.get("changed")
            n_resampled = summary.get("n_resampled", 0)

        extra_calls = cost["extra_calls"]
        extra_tok = cost["extra_token_proxy"]

        results.append({
            "method": m["name"],
            "n_resampled": n_resampled,
            "acc": acc,
            "fixed": fixed,
            "broken": broken,
            "net": net,
            "changed": changed,
            "extra_calls": extra_calls,
            "extra_calls_per_sample": extra_calls / args.n_samples,
            "extra_token_proxy": extra_tok,
            "extra_token_proxy_per_sample": extra_tok / args.n_samples,
            "cost": cost,
        })

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("| Method | Acc | n_resampled | fixed | broken | net | changed | extra_calls | calls/sample | extra_token_proxy/sample |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in results:
        def fmt(x):
            return "-" if x is None else str(x)

        lines.append(
            f"| {r['method']} | {r['acc']:.4f} | {r['n_resampled']} | "
            f"{fmt(r['fixed'])} | {fmt(r['broken'])} | {fmt(r['net'])} | {fmt(r['changed'])} | "
            f"{r['extra_calls']} | {r['extra_calls_per_sample']:.3f} | "
            f"{r['extra_token_proxy_per_sample']:.1f} |"
        )

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
