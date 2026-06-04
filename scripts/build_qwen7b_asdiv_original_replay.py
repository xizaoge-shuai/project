#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from collections import defaultdict

TARGET = Path("data/processed/unified/asdiv/test_numeric_has_disagreement_full2305.jsonl")
BASE_DETAILS = Path("outputs/predictions/asdiv_full2305_numeric_baseline_details.jsonl")
FINAL_DECISION = Path("outputs/predictions/asdiv_numeric_extra_confirm/numeric_full_total3_seed2_margin0.jsonl")

EXTRAS = [
    Path("data/processed/trajectories/asdiv/extra_numeric_full_seed42.jsonl"),
    Path("data/processed/trajectories/asdiv/extra_numeric_full_seed101.jsonl"),
    Path("data/processed/trajectories/asdiv/extra_numeric_full_seed202.jsonl"),
]

OUT = Path("outputs/predictions/qwen7b_replay/asdiv_numeric_full2249_original_candidates_replay.jsonl")


def norm_num(x):
    if x is None:
        return ""
    s = str(x).strip().replace(",", "").replace("$", "").replace("\\$", "")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", s)
    if nums:
        s = nums[-1]
    if "/" in s:
        try:
            a, b = s.split("/", 1)
            v = float(a) / float(b)
            return f"{v:.10f}".rstrip("0").rstrip(".")
        except Exception:
            pass
    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.10f}".rstrip("0").rstrip(".")
    except Exception:
        return re.sub(r"\s+", "", str(x).strip().lower())


def get_sid(r):
    return str(r.get("sample_id") or r.get("id") or "")


def get_text(r):
    for k in ["trajectory", "text", "reasoning", "output", "completion", "response"]:
        if r.get(k):
            return str(r[k])
    return ""


def extract_answer(r):
    for k in ["answer", "final_answer", "pred_answer", "prediction", "extracted_answer"]:
        if r.get(k) is not None:
            a = norm_num(r.get(k))
            if a:
                return a

    text = get_text(r)
    for p in [
        r"Final Answer\s*[:：]\s*([^\n]+)",
        r"Answer\s*[:：]\s*([^\n]+)",
        r"答案\s*[:：]\s*([^\n]+)",
    ]:
        m = re.findall(p, text, flags=re.I)
        if m:
            return norm_num(m[-1])

    return norm_num(text[-300:])


# 997 target ids
target_rows = [json.loads(x) for x in TARGET.open(encoding="utf-8") if x.strip()]
target_ids = [get_sid(r) for r in target_rows]
target_set = set(target_ids)

# 2249 numeric base details
base_by_id = {}
with BASE_DETAILS.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        sid = get_sid(r)
        if sid:
            base_by_id[sid] = r

# recorded final decision for 997 targets
final_by_id = {}
with FINAL_DECISION.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        sid = get_sid(r)
        if sid:
            final_by_id[sid] = r

# extra candidates: 997 * 12
extra_by_id = defaultdict(list)
for fp in EXTRAS:
    print("loading extra:", fp)
    with fp.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            sid = get_sid(r)
            if sid not in target_set:
                continue
            ans = extract_answer(r)
            if ans:
                extra_by_id[sid].append(ans)

OUT.parent.mkdir(parents=True, exist_ok=True)

missing_base = []
missing_final = []
missing_extra = []

with OUT.open("w", encoding="utf-8") as f:
    for sid in target_ids:
        b = base_by_id.get(sid)
        d = final_by_id.get(sid)

        if b is None:
            missing_base.append(sid)
            continue
        if d is None:
            missing_final.append(sid)
            continue

        # base details 里通常有 answers / answers_norm / majority_answer / majority_ok
        orig = b.get("answers_norm") or b.get("answers") or d.get("base_answers") or [b.get("majority_answer") or d.get("current_answer")]
        orig = [norm_num(x) for x in orig if norm_num(x)]

        extra = extra_by_id.get(sid, [])
        if len(extra) == 0:
            missing_extra.append(sid)

        out = {
            "sample_id": sid,
            "gold_answer": d.get("gold_answer") or b.get("gold_answer") or b.get("answer"),
            "current_answer": d.get("current_answer") or b.get("majority_answer"),
            "current_ok": d.get("current_ok") if d.get("current_ok") is not None else b.get("majority_ok"),
            "final_answer": d.get("final_answer"),
            "final_ok": d.get("final_ok"),
            "fixed": d.get("fixed"),
            "broken": d.get("broken"),
            "changed": d.get("changed"),
            "orig_answers": orig,
            "extra_answers": extra,
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print("saved:", OUT)
print("target_ids:", len(target_ids))
print("base_by_id:", len(base_by_id))
print("final_by_id:", len(final_by_id))
print("rows_out:", sum(1 for _ in OUT.open(encoding="utf-8")))
print("missing_base:", len(missing_base), missing_base[:5])
print("missing_final:", len(missing_final), missing_final[:5])
print("missing_extra:", len(missing_extra), missing_extra[:5])

lens = [len(json.loads(x).get("extra_answers", [])) for x in OUT.open(encoding="utf-8") if x.strip()]
print("extra min/avg/max:", min(lens), sum(lens)/len(lens), max(lens))
