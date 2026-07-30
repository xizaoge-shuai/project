import argparse
import json
import re
from pathlib import Path
from collections import defaultdict, Counter

LETTERS = ["a", "b", "c", "d", "e"]

def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sample_key(r):
    x = (
        r.get("sample_id")
        or r.get("question_id")
        or r.get("qid")
        or r.get("problem_id")
        or r.get("id")
        or r.get("question")
        or r.get("problem")
        or r.get("input")
    )
    x = str(x)
    x = re.sub(r"_traj_\d+$", "", x)
    return x

def normalize_text(x):
    x = "" if x is None else str(x).lower()
    x = x.replace("\\boxed", "")
    x = re.sub(r"[{}$]", "", x)
    x = x.replace("\\%", "%")
    x = x.replace("percent", "%")
    x = re.sub(r"\s+", " ", x).strip()
    x = x.strip(" .,\n\t;:")
    return x

def normalize_number(x):
    if x is None:
        return None
    s = str(x).lower()
    s = s.replace(",", "")
    s = s.replace("%", "")
    s = re.sub(r"\\boxed|[{}$]", "", s)
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except Exception:
        return None

def close_num(a, b):
    if a is None or b is None:
        return False
    return abs(a - b) <= max(1e-6, 1e-4 * max(1.0, abs(a), abs(b)))

def parse_options_from_string(s):
    s = str(s)
    opts = {}

    # 支持 a ) xxx, b ) xxx, c ) xxx
    pat = re.compile(
        r"(?i)([a-e])\s*[\)\].:\-]\s*(.*?)(?=(?:\s*,?\s*[a-e]\s*[\)\].:\-]\s*)|$)"
    )
    for m in pat.finditer(s):
        letter = m.group(1).lower()
        val = m.group(2).strip(" ,;")
        if val:
            opts[letter] = val

    return opts

def get_options(r):
    opts = {}

    # a/b/c/d/e 独立字段
    for l in LETTERS:
        for k in [l, l.upper(), f"option_{l}", f"option_{l.upper()}"]:
            if k in r and r[k] not in [None, ""]:
                opts[l] = r[k]

    # options / choices 字段
    for k in ["options", "choices", "choice", "answer_options"]:
        if k not in r or r[k] in [None, ""]:
            continue

        v = r[k]

        if isinstance(v, dict):
            for kk, vv in v.items():
                kk = str(kk).strip().lower()
                if kk in LETTERS:
                    opts[kk] = vv
                elif len(kk) > 0 and kk[0] in LETTERS:
                    opts[kk[0]] = vv

        elif isinstance(v, list):
            for i, vv in enumerate(v[:5]):
                opts[LETTERS[i]] = vv

        else:
            parsed = parse_options_from_string(v)
            opts.update(parsed)

    # 有些数据把选项拼在 question/problem/input/prompt 里
    if not opts:
        for k in ["question", "problem", "input", "prompt"]:
            if k in r and r[k]:
                parsed = parse_options_from_string(r[k])
                if parsed:
                    opts.update(parsed)

    return {k: str(v) for k, v in opts.items() if k in LETTERS}

def extract_raw_answer(r):
    for k in ["final_answer", "answer", "pred_answer", "prediction", "majority_answer"]:
        v = r.get(k)
        if v not in [None, ""]:
            return str(v)

    text = str(
        r.get("trajectory")
        or r.get("text")
        or r.get("reasoning")
        or r.get("output")
        or r.get("generated_text")
        or ""
    )

    pats = [
        r"Final Answer\s*[:：]\s*(.*)",
        r"\*\*Final Answer:\*\*\s*(.*)",
        r"Answer\s*[:：]\s*(.*)",
        r"the answer is\s*(.*)",
    ]

    for p in pats:
        ms = list(re.finditer(p, text, flags=re.I))
        if ms:
            ans = ms[-1].group(1).strip()
            ans = ans.split("\n")[0].strip()
            ans = re.sub(r"</?think>", "", ans).strip()
            return ans

    return ""

def map_to_letter(raw, opts):
    if raw is None:
        return None

    s = normalize_text(raw)

    # 直接是 a-e
    m = re.fullmatch(r"\(?\s*([a-e])\s*\)?", s)
    if m:
        return m.group(1)

    # option a / choice a / answer a
    m = re.search(r"\b(?:option|choice|answer)\s*\(?([a-e])\)?\b", s)
    if m:
        return m.group(1)

    # 文本完全等于某个 option
    for l, v in opts.items():
        if normalize_text(s) == normalize_text(v):
            return l

    # 数值匹配：模型输出 38，选项 a 是 38 %
    pred_num = normalize_number(s)
    num_matches = []
    for l, v in opts.items():
        opt_num = normalize_number(v)
        if close_num(pred_num, opt_num):
            num_matches.append(l)
    if len(num_matches) == 1:
        return num_matches[0]

    # ratio 匹配：1:729
    compact = re.sub(r"\s+", "", normalize_text(s))
    for l, v in opts.items():
        nv = re.sub(r"\s+", "", normalize_text(v))
        if compact and compact == nv:
            return l

    return None

def majority_label(labels):
    labels = [x for x in labels if x in LETTERS]
    if not labels:
        return None
    c = Counter(labels)
    maxc = max(c.values())
    # tie 时按 a,b,c,d,e 固定顺序，保证可复现
    for l in LETTERS:
        if c[l] == maxc:
            return l
    return None

def group_by_sample(rows):
    g = defaultdict(list)
    for r in rows:
        g[sample_key(r)].append(r)
    return g

def get_gold(r):
    for k in ["gold_answer", "answer", "correct_answer", "label"]:
        v = r.get(k)
        if v is None:
            continue
        v = str(v).strip().lower()
        if v in LETTERS:
            return v
    return None

def load_target_ids(txt_fp, target_jsonl):
    ids = set()

    p = Path(txt_fp)
    if p.exists():
        for line in open(p, encoding="utf-8"):
            if line.strip():
                ids.add(line.strip())

    if ids:
        return ids, "txt"

    p2 = Path(target_jsonl)
    if p2.exists():
        for r in read_jsonl(p2):
            ids.add(sample_key(r))

    return ids, "jsonl"

def evaluate(samples, base_g, extra_gs, target_ids, min_total, min_seed, min_margin):
    n = len(samples)
    base_ok_n = 0
    final_ok_n = 0
    changed = fixed = broken = 0
    missing_options = 0
    n_eval = 0
    details = []

    for sid, srow in samples.items():
        opts = get_options(srow)
        gold = get_gold(srow)

        if not opts:
            missing_options += 1

        base_labels = [map_to_letter(extract_raw_answer(r), opts) for r in base_g.get(sid, [])]
        base_pred = majority_label(base_labels)
        base_ok = int(base_pred == gold)

        final_pred = base_pred
        decision = "keep_base"

        if sid in target_ids:
            n_eval += 1
            total_counts = Counter()
            seed_has = Counter()

            for eg in extra_gs:
                seed_labels = [map_to_letter(extract_raw_answer(r), opts) for r in eg.get(sid, [])]
                seed_unique = set([x for x in seed_labels if x in LETTERS])

                for x in seed_labels:
                    if x in LETTERS:
                        total_counts[x] += 1

                for x in seed_unique:
                    seed_has[x] += 1

            if total_counts:
                ranked = sorted(LETTERS, key=lambda x: (-total_counts[x], LETTERS.index(x)))
                top = ranked[0]
                top_count = total_counts[top]
                second_count = total_counts[ranked[1]] if len(ranked) > 1 else 0
                margin = top_count - second_count

                if top_count >= min_total and seed_has[top] >= min_seed and margin >= min_margin:
                    final_pred = top
                    decision = f"switch_to_{top}"

        final_ok = int(final_pred == gold)

        base_ok_n += base_ok
        final_ok_n += final_ok

        if final_pred != base_pred:
            changed += 1
            if not base_ok and final_ok:
                fixed += 1
            if base_ok and not final_ok:
                broken += 1

        details.append({
            "sample_id": sid,
            "gold": gold,
            "options": opts,
            "base_labels": base_labels,
            "base_pred": base_pred,
            "base_ok": base_ok,
            "final_pred": final_pred,
            "final_ok": final_ok,
            "decision": decision,
        })

    net = fixed - broken

    metric = {
        "n_samples": n,
        "n_eval": n_eval,
        "base_acc": base_ok_n / n if n else 0,
        "estimated_global_acc": final_ok_n / n if n else 0,
        "final_acc": final_ok_n / n if n else 0,
        "gain": (final_ok_n - base_ok_n) / n if n else 0,
        "changed": changed,
        "fixed": fixed,
        "broken": broken,
        "net": net,
        "missing_options": missing_options,
        "min_total_support": min_total,
        "min_seed_support": min_seed,
        "min_margin": min_margin,
    }

    return metric, details

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--extras", nargs="+", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--target_jsonl", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="mathqa_deepseek7b_optionmap")
    args = ap.parse_args()

    scope_rows = read_jsonl(args.scope)
    samples = {sample_key(r): r for r in scope_rows}

    base_rows = read_jsonl(args.base)
    extra_rows = [read_jsonl(fp) for fp in args.extras]

    base_g = group_by_sample(base_rows)
    extra_gs = [group_by_sample(rows) for rows in extra_rows]

    target_ids, target_source = load_target_ids(args.targets, args.target_jsonl)
    if not target_ids:
        target_ids = set(samples.keys())
        target_source = "all_samples_fallback"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for min_total in [2, 3, 4]:
        for min_seed in [1, 2, 3]:
            for min_margin in [0, 1, 2]:
                metric, details = evaluate(
                    samples=samples,
                    base_g=base_g,
                    extra_gs=extra_gs,
                    target_ids=target_ids,
                    min_total=min_total,
                    min_seed=min_seed,
                    min_margin=min_margin,
                )

                name = f"{args.prefix}_total{min_total}_seed{min_seed}_margin{min_margin}"
                metric["name"] = name
                metric["target_source"] = target_source

                with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
                    json.dump(metric, f, ensure_ascii=False, indent=2)

                if min_total == 2 and min_seed == 1 and min_margin == 0:
                    write_jsonl(out_dir / f"{name}_details.jsonl", details)

                all_metrics.append(metric)

    all_metrics = sorted(all_metrics, key=lambda x: x["estimated_global_acc"], reverse=True)

    print("| name | base_acc | final_acc | gain | n_eval | changed | fixed | broken | net | missing_options |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for m in all_metrics:
        print(
            f"| {m['name']} | {m['base_acc']:.4f} | {m['estimated_global_acc']:.4f} | {m['gain']:.4f} | "
            f"{m['n_eval']} | {m['changed']} | {m['fixed']} | {m['broken']} | {m['net']} | {m['missing_options']} |"
        )

if __name__ == "__main__":
    main()
