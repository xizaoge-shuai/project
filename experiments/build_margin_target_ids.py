import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def clean(x):
    x = str(x or "").strip().replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", x)
    if not nums:
        return x
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y


def compute_margin_norm(r):
    if "margin_norm" in r:
        return float(r["margin_norm"])

    answers = r.get("answers")
    scores = r.get("scores")

    if answers is None or scores is None:
        return None

    by_ans = defaultdict(float)
    for a, s in zip(answers, scores):
        a = clean(a)
        if a == "":
            continue
        by_ans[a] += float(s)

    vals = sorted(by_ans.values(), reverse=True)
    if len(vals) < 2:
        return 1.0

    denom = sum(abs(v) for v in vals) + 1e-12
    return (vals[0] - vals[1]) / denom


def get_sid(r):
    return r.get("sample_id") or r.get("id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True)
    ap.add_argument("--out_dir", default="outputs/targets")
    ap.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50")
    args = ap.parse_args()

    rows = read_jsonl(args.details)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    missing = 0

    for r in rows:
        sid = get_sid(r)
        m = compute_margin_norm(r)
        if sid is None or m is None:
            missing += 1
            continue
        records.append((sid, m))

    if not records:
        print("No usable margin records found.")
        print("Example keys:", sorted(rows[0].keys()) if rows else [])
        raise SystemExit(1)

    print("records:", len(records), "missing:", missing)

    for t in [float(x) for x in args.thresholds.split(",") if x.strip()]:
        ids = sorted({sid for sid, m in records if m <= t})
        tag = f"{int(round(t * 100)):03d}"
        fp = out_dir / f"gsm8k_full_margin{tag}_sample_ids.txt"

        with fp.open("w", encoding="utf-8") as f:
            for sid in ids:
                f.write(sid + "\n")

        print(f"| margin<={t:.2f} | {len(ids)} | {fp} |")


if __name__ == "__main__":
    main()
