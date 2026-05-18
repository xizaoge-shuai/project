import argparse
import json
import re
from pathlib import Path
from collections import defaultdict, Counter


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def write_json(fp, obj):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean(x):
    s = str(x or "").strip().replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return s
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y


def ok(a, g):
    return clean(a) == clean(g)


def mean(xs):
    return sum(xs) / max(1, len(xs))


def traj_score(prefix_rows, k):
    rs = sorted(prefix_rows, key=lambda r: int(r.get("prefix_num_units", 0)))
    ps = [float(r.get("success_prob", 0.0)) for r in rs]
    if not ps:
        return 0.0
    if k > 0:
        ps = ps[-k:]
    return mean(ps)


def majority_answer(answers):
    vals = [clean(a) for a in answers if clean(a) != ""]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def weighted_answer(answer_score_pairs):
    by_ans = defaultdict(float)
    for ans, score in answer_score_pairs:
        ans = clean(ans)
        if ans == "":
            continue
        by_ans[ans] += float(score)
    if not by_ans:
        return ""
    return sorted(by_ans.items(), key=lambda x: x[1], reverse=True)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tail_k", type=int, default=5)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    preds = read_jsonl(args.predictions)
    trajs = read_jsonl(args.trajectories)

    pred_by_tid = defaultdict(list)
    for r in preds:
        pred_by_tid[r["trajectory_id"]].append(r)

    traj_by_sid = defaultdict(list)
    for r in trajs:
        sid = r.get("sample_id") or r.get("id")
        traj_by_sid[sid].append(r)

    details = []
    first = majority = oracle_any = pce_top1 = weighted = 0
    n = 0

    for sid, rs in sorted(traj_by_sid.items()):
        rs = sorted(rs, key=lambda r: r.get("trajectory_id", r.get("id", "")))
        gold = clean(rs[0].get("gold_answer", rs[0].get("answer", "")))

        answers = []
        scored = []

        for tr in rs:
            tid = tr.get("trajectory_id") or tr.get("id")
            ans = clean(tr.get("final_answer", tr.get("answer", "")))
            score = traj_score(pred_by_tid.get(tid, []), args.tail_k)
            answers.append(ans)
            scored.append((ans, score, tid))

        first_ans = answers[0] if answers else ""
        maj_ans = majority_answer(answers)
        top_ans = sorted(scored, key=lambda x: x[1], reverse=True)[0][0] if scored else ""
        w_ans = weighted_answer([(a, s) for a, s, _ in scored])

        first_ok = int(ok(first_ans, gold))
        maj_ok = int(ok(maj_ans, gold))
        any_ok = int(any(ok(a, gold) for a in answers))
        top_ok = int(ok(top_ans, gold))
        w_ok = int(ok(w_ans, gold))

        first += first_ok
        majority += maj_ok
        oracle_any += any_ok
        pce_top1 += top_ok
        weighted += w_ok
        n += 1

        details.append({
            "sample_id": sid,
            "dataset": args.dataset,
            "gold_answer": gold,
            "answers": answers,
            "trajectory_scores": [
                {"trajectory_id": tid, "answer": a, "score": s}
                for a, s, tid in scored
            ],
            "first_answer": first_ans,
            "majority_answer": maj_ans,
            "pce_top1_answer": top_ans,
            "weighted_answer": w_ans,
            "first_ok": first_ok,
            "majority_ok": maj_ok,
            "oracle_any_ok": any_ok,
            "pce_top1_ok": top_ok,
            "weighted_ok": w_ok,
        })

    summary = {
        "dataset": args.dataset,
        "n_samples": n,
        "tail_k": args.tail_k,
        "first_acc": first / max(1, n),
        "majority_acc": majority / max(1, n),
        "oracle_any_acc": oracle_any / max(1, n),
        "pce_top1_tail_acc": pce_top1 / max(1, n),
        "weighted_tail_acc": weighted / max(1, n),
        "weighted_gain_vs_majority": (weighted - majority) / max(1, n),
        "pce_top1_gain_vs_majority": (pce_top1 - majority) / max(1, n),
        "out_jsonl": args.out_jsonl,
    }

    write_json(args.out_json, summary)
    write_jsonl(args.out_jsonl, details)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
