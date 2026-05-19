import argparse, json, re, string
from pathlib import Path
from collections import defaultdict, Counter

def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]

def norm_num(x):
    s = str(x or "").replace(",", "").replace("$", "").strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return " ".join(s.lower().split())
    y = nums[-1]
    return y.rstrip("0").rstrip(".") if "." in y else y

def norm_choice(x):
    raw = str(x or "")
    candidates = re.findall(r"final answer\s*[:：]\s*([^\n\|]+)", raw, flags=re.I) + [raw]
    for c in reversed(candidates):
        m = re.search(r"\(([A-Ea-e])\)", c)
        if m: return m.group(1).lower()
        m = re.search(r"\boption\s*([A-Ea-e])\b", c, flags=re.I)
        if m: return m.group(1).lower()
        m = re.search(r"^\s*([A-Ea-e])[\)\.\:]\s*", c)
        if m: return m.group(1).lower()
        m = re.search(r"\b([A-Ea-e])\b", c)
        if len(str(c).strip()) <= 5 and m:
            return m.group(1).lower()
    s = raw.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())

def norm(x, task_type):
    return norm_choice(x) if task_type == "choice" else norm_num(x)

ap = argparse.ArgumentParser()
ap.add_argument("--baseline_details", required=True)
ap.add_argument("--extra_jsonls", nargs="+", required=True)
ap.add_argument("--target_ids", required=True)
ap.add_argument("--task_type", choices=["numeric", "choice"], required=True)
ap.add_argument("--base_acc", type=float, required=True)
ap.add_argument("--n_samples", type=int, required=True)
ap.add_argument("--min_total_support", type=int, default=2)
ap.add_argument("--min_seed_support", type=int, default=2)
ap.add_argument("--min_margin", type=int, default=1)
ap.add_argument("--out_json", required=True)
args = ap.parse_args()

base = {r["sample_id"]: r for r in read_jsonl(args.baseline_details)}
target_ids = [x.strip() for x in open(args.target_ids, encoding="utf-8") if x.strip()]

extras = defaultdict(list)
for seed_idx, fp in enumerate(args.extra_jsonls):
    for r in read_jsonl(fp):
        sid = r["sample_id"]
        ans = norm(r.get("final_answer", ""), args.task_type)
        if ans:
            extras[sid].append((seed_idx, ans))

fixed = broken = changed = cur_correct = final_correct = 0

for sid in target_ids:
    b = base[sid]
    gold = b["gold_norm"]
    cur = b["majority_answer"]
    cur_ok = int(b["majority_ok"])

    cnt = Counter(a for _, a in extras.get(sid, []))
    seed_support = defaultdict(set)
    for seed_idx, ans in extras.get(sid, []):
        seed_support[ans].add(seed_idx)

    if cnt:
        top, top_total = cnt.most_common(1)[0]
        runner = cnt.most_common(2)[1][1] if len(cnt) >= 2 else 0
        top_seed = len(seed_support[top])
    else:
        top, top_total, runner, top_seed = "", 0, 0, 0

    margin = top_total - runner
    final = cur
    if top and top != cur and top_total >= args.min_total_support and top_seed >= args.min_seed_support and margin >= args.min_margin:
        final = top

    fin_ok = int(final == gold)

    fixed += int(cur_ok == 0 and fin_ok == 1)
    broken += int(cur_ok == 1 and fin_ok == 0)
    changed += int(final != cur)
    cur_correct += cur_ok
    final_correct += fin_ok

net = fixed - broken
summary = {
    "base_acc": args.base_acc,
    "n_samples": args.n_samples,
    "n_eval": len(target_ids),
    "min_total_support": args.min_total_support,
    "min_seed_support": args.min_seed_support,
    "min_margin": args.min_margin,
    "current_acc_on_eval": cur_correct / max(1, len(target_ids)),
    "final_acc_on_eval": final_correct / max(1, len(target_ids)),
    "fixed": fixed,
    "broken": broken,
    "net": net,
    "changed": changed,
    "estimated_global_acc": args.base_acc + net / args.n_samples,
}
Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
