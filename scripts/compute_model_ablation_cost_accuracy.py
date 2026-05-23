import argparse
import csv
import glob
import json
import math
import os
import re
from pathlib import Path

DATASETS = ["gsm8k", "svamp", "asdiv", "math500", "mathqa"]
SEEDS = [42, 101, 202]
BASE_N_TRAJ = 3
EXTRA_N_TRAJ_PER_SEED = 4

INPUTS = {
    "gsm8k": "data/processed/unified/model_ablation/gsm8k_scope.jsonl",
    "svamp": "data/processed/unified/model_ablation/svamp_scope.jsonl",
    "asdiv": "data/processed/unified/model_ablation/asdiv_scope.jsonl",
    "math500": "data/processed/unified/model_ablation/math500_scope.jsonl",
    "mathqa": "data/processed/unified/model_ablation/mathqa_scope.jsonl",
}

TEXT_KEYS = [
    "trajectory", "text", "reasoning", "completion", "response", "output",
    "generated_text", "final_answer", "answer"
]

QUESTION_KEYS = ["question", "problem", "input", "prompt", "context"]


def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
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
    with open(p, encoding="utf-8") as f:
        return sum(1 for x in f if x.strip())


def load_json(fp):
    p = Path(fp)
    if not p.exists():
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


def token_proxy_text(s):
    if s is None:
        return 0
    s = str(s)
    # 轻量 token proxy：英文/数字按词和符号切分；比 whitespace 更稳定
    toks = re.findall(r"\d+\.\d+|\d+|[A-Za-z]+|[^\sA-Za-z0-9]", s)
    return len(toks)


def extract_text(row, keys):
    parts = []
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        v = str(v)
        if v.strip():
            parts.append(v)
    return "\n".join(parts)


def jsonl_output_tokens(fp):
    rows = read_jsonl(fp)
    total = 0
    for r in rows:
        txt = extract_text(r, TEXT_KEYS)
        total += token_proxy_text(txt)
    return total


def jsonl_prompt_tokens(fp):
    rows = read_jsonl(fp)
    total = 0
    for r in rows:
        txt = extract_text(r, QUESTION_KEYS)
        total += token_proxy_text(txt)
    return total


def parse_seconds_from_tqdm(log_fp, expected_calls=None):
    p = Path(log_fp)
    if not p.exists():
        return {
            "latency_proxy_sec": None,
            "latency_observed_calls": 0,
            "latency_note": "missing_log",
        }

    raw = p.read_bytes().decode("utf-8", errors="ignore").replace("\r", "\n")
    durs = []

    for line in raw.splitlines():
        if "Processed prompts: 100%" not in line:
            continue
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)s/it", line)
        if m:
            durs.append(float(m.group(1)))
            continue

        # fallback: [00:07<00:00, ...]
        m = re.search(r"\[(\d+):(\d+)<", line)
        if m:
            durs.append(int(m.group(1)) * 60 + int(m.group(2)))

    if not durs:
        return {
            "latency_proxy_sec": None,
            "latency_observed_calls": 0,
            "latency_note": "no_tqdm_seconds",
        }

    # vLLM/tqdm 在 nohup 里可能重复写同一条进度，所以这里只把它当 proxy。
    if expected_calls and expected_calls > 0:
        if len(durs) >= expected_calls:
            used = durs[:expected_calls]
            note = f"clipped_from_{len(durs)}_to_{expected_calls}"
            total = sum(used)
            observed = expected_calls
        else:
            mean = sum(durs) / len(durs)
            total = mean * expected_calls
            observed = len(durs)
            note = f"scaled_from_{len(durs)}_to_{expected_calls}"
    else:
        total = sum(durs)
        observed = len(durs)
        note = "raw_sum"

    return {
        "latency_proxy_sec": total,
        "latency_observed_calls": observed,
        "latency_note": note,
    }


def get_acc_from_metric(x, keys):
    if not x:
        return None
    for k in keys:
        if k in x and isinstance(x[k], (int, float)):
            return float(x[k])
    return None


def select_confirm_metric(ds, tag, mode="best"):
    files = sorted(glob.glob(f"outputs/metrics/model_ablation/{ds}_{tag}_confirm*.json"))
    files += sorted(glob.glob(f"outputs/metrics/model_ablation/{ds}_{tag}_*confirm*.json"))
    files = sorted(set(files))

    if not files:
        return None, None

    scored = []
    for fp in files:
        x = load_json(fp)
        if not x:
            continue
        final = get_acc_from_metric(x, ["estimated_global_acc", "final_acc", "acc", "accuracy"])
        mtime = os.path.getmtime(fp)
        scored.append((fp, x, final, mtime))

    if not scored:
        return files[-1], load_json(files[-1])

    if mode == "latest":
        fp, x, _, _ = sorted(scored, key=lambda z: z[3])[-1]
        return fp, x

    # default: best final acc；论文里如果固定 setting，表中 selected_confirm_file 可追溯
    fp, x, _, _ = sorted(scored, key=lambda z: (-1e18 if z[2] is None else z[2]))[-1]
    return fp, x


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "-"
        return f"{x:.{nd}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["deepseek7b", "qwen3b"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--select_confirm", choices=["best", "latest"], default="best")
    ap.add_argument("--out_prefix", default="outputs/logs/final_summaries/model_ablation_cost_accuracy")
    args = ap.parse_args()

    rows = []

    for tag in args.tags:
        for ds in args.datasets:
            input_fp = INPUTS[ds]
            n_samples = nlines(input_fp)

            base_fp = f"data/processed/trajectories/model_ablation/{ds}_{tag}_base_3traj.jsonl"
            target_fp = f"data/processed/unified/model_ablation/{ds}_{tag}_has_disagreement.jsonl"
            base_metric_fp = f"outputs/metrics/model_ablation/{ds}_{tag}_base.json"

            base_rows = nlines(base_fp)
            target_n = nlines(target_fp)
            base_expected_calls = n_samples * BASE_N_TRAJ if n_samples else None

            extra_rows_by_seed = {}
            extra_saved_calls = 0
            extra_output_tok = 0
            extra_latency_sec = 0.0
            extra_latency_seen = False
            latency_notes = []

            for seed in SEEDS:
                extra_fp = f"data/processed/trajectories/model_ablation/{ds}_{tag}_extra_seed{seed}.jsonl"
                log_fp = f"outputs/logs/model_ablation/generate_{ds}_{tag}_extra_seed{seed}.log"

                r = nlines(extra_fp)
                extra_rows_by_seed[seed] = r
                extra_saved_calls += r
                extra_output_tok += jsonl_output_tokens(extra_fp)

                expected_this_seed = target_n * EXTRA_N_TRAJ_PER_SEED if target_n else None
                lat = parse_seconds_from_tqdm(log_fp, expected_this_seed)
                if lat["latency_proxy_sec"] is not None:
                    extra_latency_sec += lat["latency_proxy_sec"]
                    extra_latency_seen = True
                latency_notes.append(f"seed{seed}:{lat['latency_note']}")

            extra_expected_calls = target_n * EXTRA_N_TRAJ_PER_SEED * len(SEEDS) if target_n else None
            total_expected_calls = (base_expected_calls or 0) + (extra_expected_calls or 0)

            base_output_tok = jsonl_output_tokens(base_fp)
            input_prompt_tok = jsonl_prompt_tokens(input_fp)

            # prompt proxy：base 每题 3 次；extra 只对 target 题，每 seed 4 次
            target_prompt_tok = jsonl_prompt_tokens(target_fp)
            base_prompt_tok_proxy = input_prompt_tok * BASE_N_TRAJ
            extra_prompt_tok_proxy = target_prompt_tok * EXTRA_N_TRAJ_PER_SEED * len(SEEDS)

            base_lat = parse_seconds_from_tqdm(
                f"outputs/logs/model_ablation/generate_{ds}_{tag}_base.log",
                base_expected_calls,
            )

            base_metric = load_json(base_metric_fp)
            base_acc = get_acc_from_metric(base_metric, ["majority_acc", "base_acc", "acc", "accuracy"])

            confirm_fp, confirm_metric = select_confirm_metric(ds, tag, args.select_confirm)
            final_acc = get_acc_from_metric(confirm_metric, ["estimated_global_acc", "final_acc", "acc", "accuracy"])
            acc_gain = None
            if base_acc is not None and final_acc is not None:
                acc_gain = final_acc - base_acc

            row = {
                "model": tag,
                "dataset": ds,
                "n_samples": n_samples,
                "base_calls_saved": base_rows,
                "base_calls_expected": base_expected_calls,
                "target_samples": target_n,
                "target_rate": (target_n / n_samples) if n_samples else None,
                "extra_seed42_rows": extra_rows_by_seed[42],
                "extra_seed101_rows": extra_rows_by_seed[101],
                "extra_seed202_rows": extra_rows_by_seed[202],
                "extra_calls_saved": extra_saved_calls,
                "extra_calls_expected": extra_expected_calls,
                "extra_calls_per_sample_expected": (extra_expected_calls / n_samples) if n_samples and extra_expected_calls is not None else None,
                "total_calls_expected": total_expected_calls,
                "total_calls_per_sample_expected": (total_expected_calls / n_samples) if n_samples else None,
                "base_output_token_proxy": base_output_tok,
                "extra_output_token_proxy_saved": extra_output_tok,
                "base_prompt_token_proxy": base_prompt_tok_proxy,
                "extra_prompt_token_proxy_expected": extra_prompt_tok_proxy,
                "total_token_proxy_saved_or_expected": base_output_tok + extra_output_tok + base_prompt_tok_proxy + extra_prompt_tok_proxy,
                "token_proxy_per_sample": ((base_output_tok + extra_output_tok + base_prompt_tok_proxy + extra_prompt_tok_proxy) / n_samples) if n_samples else None,
                "base_latency_proxy_sec": base_lat["latency_proxy_sec"],
                "extra_latency_proxy_sec": extra_latency_sec if extra_latency_seen else None,
                "total_latency_proxy_sec": ((base_lat["latency_proxy_sec"] or 0.0) + (extra_latency_sec if extra_latency_seen else 0.0)) if (base_lat["latency_proxy_sec"] is not None or extra_latency_seen) else None,
                "latency_proxy_sec_per_sample": (((base_lat["latency_proxy_sec"] or 0.0) + (extra_latency_sec if extra_latency_seen else 0.0)) / n_samples) if n_samples and (base_lat["latency_proxy_sec"] is not None or extra_latency_seen) else None,
                "base_acc": base_acc,
                "final_acc": final_acc,
                "acc_gain": acc_gain,
                "fixed": confirm_metric.get("fixed") if confirm_metric else None,
                "broken": confirm_metric.get("broken") if confirm_metric else None,
                "net": confirm_metric.get("net") if confirm_metric else None,
                "changed": confirm_metric.get("changed") if confirm_metric else None,
                "base_metric_file": base_metric_fp if Path(base_metric_fp).exists() else "",
                "selected_confirm_file": confirm_fp or "",
                "latency_note": ";".join(latency_notes),
            }
            rows.append(row)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_fp = out_prefix.with_suffix(".csv")
    json_fp = out_prefix.with_suffix(".json")
    md_fp = out_prefix.with_suffix(".md")

    fields = list(rows[0].keys()) if rows else []
    with open(csv_fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with open(json_fp, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    show_cols = [
        "model", "dataset", "n_samples",
        "base_acc", "final_acc", "acc_gain",
        "target_samples", "target_rate",
        "extra_calls_expected", "extra_calls_per_sample_expected",
        "total_calls_per_sample_expected",
        "token_proxy_per_sample",
        "latency_proxy_sec_per_sample",
        "fixed", "broken", "net", "changed",
        "selected_confirm_file",
    ]

    with open(md_fp, "w", encoding="utf-8") as f:
        f.write("# Model ablation cost-accuracy table\n\n")
        f.write("说明：token 是轻量 proxy；latency 从 vLLM tqdm 日志估计，若日志重复会做 clipped/scaled 处理。论文中可称为 token/latency proxy，而不是精确 API 计费 token。\n\n")
        f.write("| " + " | ".join(show_cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(show_cols)) + "|\n")
        for r in rows:
            vals = []
            for c in show_cols:
                if c.endswith("file"):
                    vals.append(str(r[c]).replace("|", "\\|"))
                elif c in ["n_samples", "target_samples", "extra_calls_expected", "fixed", "broken", "net", "changed"]:
                    vals.append(fmt(r[c], 0))
                else:
                    vals.append(fmt(r[c], 4))
            f.write("| " + " | ".join(vals) + " |\n")

    print("saved:", csv_fp)
    print("saved:", json_fp)
    print("saved:", md_fp)


if __name__ == "__main__":
    main()
