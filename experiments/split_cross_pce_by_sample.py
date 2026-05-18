import argparse
import json
import random
from pathlib import Path


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sid_of(r):
    return r.get("sample_id") or r.get("id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_jsonl", required=True)
    ap.add_argument("--trajectories_jsonl", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_train", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    labels = read_jsonl(args.labels_jsonl)
    trajs = read_jsonl(args.trajectories_jsonl)

    sids = sorted({sid_of(r) for r in trajs if sid_of(r)})
    rng = random.Random(args.seed)
    rng.shuffle(sids)

    train_sids = set(sids[:args.n_train])
    test_sids = set(sids[args.n_train:])

    train_labels = [r for r in labels if sid_of(r) in train_sids]
    test_labels = [r for r in labels if sid_of(r) in test_sids]

    train_trajs = [r for r in trajs if sid_of(r) in train_sids]
    test_trajs = [r for r in trajs if sid_of(r) in test_sids]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_jsonl(out / "train_labels.jsonl", train_labels)
    write_jsonl(out / "test_labels.jsonl", test_labels)
    write_jsonl(out / "train_trajectories.jsonl", train_trajs)
    write_jsonl(out / "test_trajectories.jsonl", test_trajs)

    summary = {
        "dataset": args.dataset,
        "n_samples_total": len(sids),
        "n_train_samples": len(train_sids),
        "n_test_samples": len(test_sids),
        "n_train_labels": len(train_labels),
        "n_test_labels": len(test_labels),
        "n_train_trajectories": len(train_trajs),
        "n_test_trajectories": len(test_trajs),
        "seed": args.seed,
        "out_dir": str(out),
    }

    with open(out / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
