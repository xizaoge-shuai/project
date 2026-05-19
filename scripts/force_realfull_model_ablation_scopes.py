import json
import re
from pathlib import Path

OUT = Path("data/processed/unified/model_ablation")
OUT.mkdir(parents=True, exist_ok=True)

def read_jsonl(fp):
    fp = Path(fp)
    if not fp.exists():
        raise FileNotFoundError(fp)
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]

def write_jsonl(fp, rows):
    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[WRITE] {fp} {len(rows)}")

def nlines(fp):
    try:
        return sum(1 for x in open(fp, encoding="utf-8") if x.strip())
    except Exception:
        return -1

def find_exact(term, n):
    ans = []
    for fp in Path("data/processed/unified").rglob("*.jsonl"):
        s = str(fp).lower()
        if term.lower() in s:
            cnt = nlines(fp)
            ans.append((str(fp), cnt))
    for fp, cnt in sorted(ans):
        if cnt == n:
            return fp
    return None

def find_min(term, n):
    ans = []
    for fp in Path("data/processed/unified").rglob("*.jsonl"):
        s = str(fp).lower()
        if term.lower() in s:
            cnt = nlines(fp)
            ans.append((str(fp), cnt))
    for fp, cnt in sorted(ans):
        if cnt >= n:
            return fp
    return None

def get_gold(r):
    for k in ["gold_answer", "answer", "target", "label", "final_answer"]:
        v = r.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""

def is_numeric_gold(x):
    s = str(x).strip().lower()
    if not s:
        return False

    # 去掉常见数字符号
    t = s.replace(",", "")
    t = t.replace("$", "").replace("€", "").replace("£", "")
    t = t.replace("%", "")
    t = t.strip()

    # 纯数字 / 小数 / 负数
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", t):
        return True

    # 分数，比如 1/2, -3/4
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", t):
        return True

    # 混合数，比如 1 1/2
    if re.fullmatch(r"[-+]?\d+\s+\d+\s*/\s*\d+", t):
        return True

    # 多个数字构成的简单答案，例如 "2, 3" 这种也算 numeric
    if re.fullmatch(r"[-+]?\d+(\.\d+)?(\s*[,;/]\s*[-+]?\d+(\.\d+)?)+", t):
        return True

    # 如果含英文字母，按 text-gold 处理，避免污染 numeric evaluator
    if re.search(r"[a-zA-Z]", t):
        return False

    # 包含至少一个数字，且剩余字符基本都是数学符号，也算 numeric
    if re.search(r"\d", t) and re.fullmatch(r"[-+*/().\d\s,/%]+", t):
        return True

    return False

# 1. GSM8K full1319
gsm = read_jsonl("data/processed/unified/gsm8k/test.jsonl")
if len(gsm) != 1319:
    raise RuntimeError(f"GSM8K expected 1319, got {len(gsm)}")
write_jsonl(OUT / "gsm8k_scope.jsonl", gsm)

# 2. SVAMP full300：从旧 Qwen7B trajectory 反推出 300 个 sample，避免 id 用 traj-level
old_svamp = Path("data/processed/trajectories/svamp/test_local_3traj_full300.jsonl")
if old_svamp.exists():
    seen, svamp = set(), []
    for line in open(old_svamp, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (
            r.get("sample_id")
            or r.get("question_id")
            or re.sub(r"_traj_\d+$", "", str(r.get("id", "")))
            or r.get("question")
        )
        if key in seen:
            continue
        seen.add(key)
        svamp.append({
            "id": key,
            "sample_id": key,
            "dataset": "svamp",
            "split": r.get("split", "test"),
            "task": "svamp",
            "question": r.get("question") or r.get("problem") or r.get("input"),
            "context": r.get("context", ""),
            "gold_answer": r.get("gold_answer") or r.get("answer"),
            "answer": r.get("gold_answer") or r.get("answer"),
        })
else:
    svamp_src = find_exact("svamp", 300)
    if not svamp_src:
        raise RuntimeError("Missing SVAMP full300 source.")
    svamp = read_jsonl(svamp_src)

if len(svamp) != 300:
    raise RuntimeError(f"SVAMP expected 300, got {len(svamp)}")
write_jsonl(OUT / "svamp_scope.jsonl", svamp)

# 3. ASDiv numeric-full2249：从 full2305 中过滤 numeric-gold
asdiv_full = Path("data/processed/unified/asdiv/test.jsonl")
asdiv_rows = read_jsonl(asdiv_full)
if len(asdiv_rows) != 2305:
    print(f"[WARN] ASDiv full expected 2305, got {len(asdiv_rows)} from {asdiv_full}")

asdiv_numeric = []
asdiv_text = []
for r in asdiv_rows:
    gold = get_gold(r)
    if is_numeric_gold(gold):
        asdiv_numeric.append(r)
    else:
        asdiv_text.append(r)

print(f"[ASDiv split] numeric={len(asdiv_numeric)} text={len(asdiv_text)}")

if len(asdiv_numeric) != 2249:
    print("[ERROR] ASDiv numeric count != 2249")
    print("First 20 text-gold examples:")
    for r in asdiv_text[:20]:
        print(json.dumps({
            "id": r.get("id") or r.get("sample_id"),
            "answer": get_gold(r),
            "question": (r.get("question") or r.get("problem") or r.get("input") or "")[:160],
        }, ensure_ascii=False))
    raise RuntimeError(f"ASDiv numeric-full expected 2249, got {len(asdiv_numeric)}")

write_jsonl(OUT / "asdiv_scope.jsonl", asdiv_numeric)

# 4. MATH500 full500
math500_src = find_exact("math500", 500)
if not math500_src:
    raise RuntimeError("Missing MATH500 full500 source.")
math500 = read_jsonl(math500_src)
write_jsonl(OUT / "math500_scope.jsonl", math500)

# 5. MathQA-500
mathqa_src = find_exact("mathqa", 500)
if mathqa_src:
    mathqa = read_jsonl(mathqa_src)
else:
    mathqa_src = find_min("mathqa", 500)
    if not mathqa_src:
        raise RuntimeError("Missing MathQA source >=500.")
    mathqa = read_jsonl(mathqa_src)[:500]
write_jsonl(OUT / "mathqa_scope.jsonl", mathqa)

# 6. BBH logical5 / formal smoke100
logical_src = None
formal_src = None
for fp in Path("data/processed/unified").rglob("*.jsonl"):
    s = str(fp).lower()
    cnt = nlines(fp)
    if "logical_deduction_five_objects" in s and cnt >= 100 and logical_src is None:
        logical_src = str(fp)
    if "formal_fallacies" in s and cnt >= 100 and formal_src is None:
        formal_src = str(fp)

if not logical_src:
    raise RuntimeError("Missing BBH logical_deduction_five_objects source >=100.")
if not formal_src:
    raise RuntimeError("Missing BBH formal_fallacies source >=100.")

logical = read_jsonl(logical_src)[:100]
formal = read_jsonl(formal_src)[:100]

write_jsonl(OUT / "bbh_logical_deduction_five_objects_scope.jsonl", logical)
write_jsonl(OUT / "bbh_formal_fallacies_scope.jsonl", formal)

print("========== REALFULL COUNTS ==========")
print("GSM8K full1319:", len(gsm))
print("SVAMP full300:", len(svamp))
print("ASDiv numeric-full2249:", len(asdiv_numeric), "from", asdiv_full)
print("ASDiv text-gold:", len(asdiv_text))
print("MATH500 full500:", len(math500), "src=", math500_src)
print("MathQA-500:", len(mathqa), "src=", mathqa_src)
print("BBH logical5 smoke100:", len(logical), "src=", logical_src)
print("BBH formal smoke100:", len(formal), "src=", formal_src)
