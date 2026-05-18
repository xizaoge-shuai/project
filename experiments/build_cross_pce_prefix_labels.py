import argparse
import json
import re
from pathlib import Path


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clean_num(x):
    s = str(x or "").strip().replace(",", "").replace("$", "")
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    if not nums:
        return s
    y = nums[-1]
    if "." in y:
        y = y.rstrip("0").rstrip(".")
    return y


def ok(a, g):
    return clean_num(a) == clean_num(g)


def get_traj_text(r):
    for k in [
        "reasoning", "solution", "response", "generated_text", "text",
        "trajectory_text", "raw_output", "output", "completion"
    ]:
        if r.get(k):
            return str(r[k])

    # 兜底：如果轨迹文件只保留 final_answer，就构造一个极简文本
    q = str(r.get("question", ""))
    fa = str(r.get("final_answer", ""))
    return f"{q}\nFinal Answer: {fa}"


def split_units(text):
    text = str(text or "").strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]

    # 优先按行切；如果太少，再按句子切
    if len(lines) >= 3:
        return lines

    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [x.strip() for x in sents if x.strip()]
    if len(sents) >= 2:
        return sents

    return [text] if text else [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    trajs = read_jsonl(args.trajectories)
    out = []

    for r in trajs:
        sid = r.get("sample_id") or r.get("id")
        tid = r.get("trajectory_id") or r.get("id") or f"{sid}_traj"
        question = r.get("question", "")
        gold = clean_num(r.get("gold_answer", r.get("answer", "")))
        final_answer = clean_num(r.get("final_answer", ""))
        label_success = int(ok(final_answer, gold))

        text = get_traj_text(r)
        units = split_units(text)
        total = max(1, len(units))

        for i in range(total):
            prefix_units = units[: i + 1]
            prefix_text = "\n".join(prefix_units)
            prefix_num = i + 1

            out.append({
                "prefix_id": f"{tid}_atom_{i}",
                "sample_id": sid,
                "trajectory_id": tid,
                "dataset": args.dataset,
                "split": args.split,
                "task": args.dataset,
                "level": "atom_level",
                "question": question,
                "context": "",
                "prefix_text": prefix_text,
                "gold_answer": gold,
                "final_answer": final_answer,
                "prefix_num_units": prefix_num,
                "trajectory_total_units": total,
                "prefix_progress": prefix_num / total,
                "label_success": label_success,
            })

    write_jsonl(args.out_jsonl, out)

    print(json.dumps({
        "dataset": args.dataset,
        "input_trajectories": args.trajectories,
        "n_trajectories": len(trajs),
        "n_prefixes": len(out),
        "out_jsonl": args.out_jsonl,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
