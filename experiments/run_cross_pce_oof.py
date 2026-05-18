import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(fp, obj):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sid_of(r):
    return r.get("sample_id") or r.get("id")


def build_text(r, feature_set):
    q = str(r.get("question", ""))
    p = str(r.get("prefix_text", r.get("context", "")))
    fa = str(r.get("final_answer", ""))

    if feature_set == "prefix_only":
        return p
    if feature_set == "prefix_plus_question":
        return f"Question:\n{q}\n\nPrefix:\n{p}"
    if feature_set == "prefix_plus_question_answer":
        return f"Question:\n{q}\n\nPrefix:\n{p}\n\nFinal answer candidate:\n{fa}"
    if feature_set == "prefix_plus_len_progress":
        return (
            f"Question:\n{q}\n\nPrefix:\n{p}\n\n"
            f"prefix_num_units={r.get('prefix_num_units')}\n"
            f"prefix_progress={r.get('prefix_progress')}"
        )
    return f"Question:\n{q}\n\nPrefix:\n{p}"


def train_and_predict(train_rows, test_rows, feature_set, max_features, min_df, C):
    X_train = [build_text(r, feature_set) for r in train_rows]
    y_train = [int(r.get("label_success", 0)) for r in train_rows]
    X_test = [build_text(r, feature_set) for r in test_rows]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=C,
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
        )),
    ])

    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_labels", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--external_train_jsonls", nargs="*", default=[])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--feature_set",
        default="prefix_plus_question_answer",
        choices=[
            "prefix_only",
            "prefix_plus_question",
            "prefix_plus_question_answer",
            "prefix_plus_len_progress",
        ],
    )
    ap.add_argument("--max_features", type=int, default=100000)
    ap.add_argument("--min_df", type=int, default=1)
    ap.add_argument("--C", type=float, default=2.0)
    args = ap.parse_args()

    target_rows = read_jsonl(args.target_labels)

    external_rows = []
    for fp in args.external_train_jsonls:
        external_rows.extend(read_jsonl(fp))

    sids = sorted({sid_of(r) for r in target_rows if sid_of(r)})
    rng = random.Random(args.seed)
    rng.shuffle(sids)

    folds = [[] for _ in range(args.n_folds)]
    for i, sid in enumerate(sids):
        folds[i % args.n_folds].append(sid)

    out_rows = []
    fold_summaries = []

    for fold_id, test_sids_list in enumerate(folds):
        test_sids = set(test_sids_list)
        train_sids = set(sids) - test_sids

        train_rows = [r for r in target_rows if sid_of(r) in train_sids]
        test_rows = [r for r in target_rows if sid_of(r) in test_sids]

        full_train_rows = external_rows + train_rows

        probs = train_and_predict(
            full_train_rows,
            test_rows,
            args.feature_set,
            args.max_features,
            args.min_df,
            args.C,
        )

        for r, p in zip(test_rows, probs):
            rr = dict(r)
            rr["success_prob"] = float(p)
            rr["pce_source"] = "target_oof" if not external_rows else "gsm8k_plus_target_oof"
            rr["fold_id"] = fold_id
            rr["feature_set"] = args.feature_set
            out_rows.append(rr)

        fold_summaries.append({
            "fold_id": fold_id,
            "n_train_target_rows": len(train_rows),
            "n_external_rows": len(external_rows),
            "n_test_rows": len(test_rows),
            "n_test_samples": len(test_sids),
        })

        print(json.dumps(fold_summaries[-1], ensure_ascii=False))

    out_rows = sorted(out_rows, key=lambda r: (sid_of(r), r.get("trajectory_id", ""), int(r.get("prefix_num_units", 0))))

    write_jsonl(args.out_jsonl, out_rows)

    summary = {
        "dataset": args.dataset,
        "n_target_rows": len(target_rows),
        "n_external_rows": len(external_rows),
        "n_samples": len(sids),
        "n_folds": args.n_folds,
        "seed": args.seed,
        "feature_set": args.feature_set,
        "out_jsonl": args.out_jsonl,
        "folds": fold_summaries,
    }

    write_json(args.out_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
