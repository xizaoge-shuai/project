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
ap.add_argument("--trajectories", required=True)
ap.add_argument("--task_type", choices=["numeric", "choice"], required=True)
ap.add_argument("--out_json", required=True)
ap.add_argument("--out_jsonl", required=True)
args = ap.parse_args()

rows = read_jsonl(args.trajectories)
by = defaultdict(list)
for r in rows:
    by[r["sample_id"]].append(r)

details = []
first = majority = oracle = has_dis = all_dis = 0

for sid, rs in sorted(by.items()):
    rs = sorted(rs, key=lambda x: str(x.get("trajectory_id", "")))
    gold = rs[0].get("gold_answer", rs[0].get("answer", ""))
    gold_n = norm(gold, args.task_type)
    answers = [r.get("final_answer", "") for r in rs]
    ans_n = [norm(a, args.task_type) for a in answers]
    cnt = Counter(a for a in ans_n if a)
    maj = cnt.most_common(1)[0][0] if cnt else ""

    first_ok = int((ans_n[0] if ans_n else "") == gold_n)
    maj_ok = int(maj == gold_n)
    any_ok = int(any(a == gold_n for a in ans_n))

    first += first_ok
    majority += maj_ok
    oracle += any_ok

    uniq = set(a for a in ans_n if a)
    has_dis += int(len(uniq) >= 2)
    all_dis += int(len(uniq) >= 3)

    details.append({
        "sample_id": sid,
        "gold_answer": gold,
        "gold_norm": gold_n,
        "answers": answers,
        "answers_norm": ans_n,
        "majority_answer": maj,
        "majority_ok": maj_ok,
        "first_ok": first_ok,
        "oracle_any_ok": any_ok,
    })

n = len(by)
summary = {
    "n_samples": n,
    "n_trajectories": len(rows),
    "first_acc": first / max(1, n),
    "majority_acc": majority / max(1, n),
    "oracle_any_acc": oracle / max(1, n),
    "has_disagreement": has_dis,
    "all_disagree": all_dis,
}
Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(args.out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
with open(args.out_jsonl, "w", encoding="utf-8") as f:
    for r in details:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
