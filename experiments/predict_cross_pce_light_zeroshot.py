import argparse
import json
import pickle
from pathlib import Path

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonls", nargs="+", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--model_out", default="")
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
    ap.add_argument("--max_features", type=int, default=200000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--C", type=float, default=2.0)
    args = ap.parse_args()

    train_rows = []
    for fp in args.train_jsonls:
        train_rows.extend(read_jsonl(fp))

    test_rows = read_jsonl(args.test_jsonl)

    X_train = [build_text(r, args.feature_set) for r in train_rows]
    y_train = [int(r.get("label_success", 0)) for r in train_rows]

    X_test = [build_text(r, args.feature_set) for r in test_rows]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=args.max_features,
            min_df=args.min_df,
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=args.C,
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
        )),
    ])

    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1].tolist()

    out = []
    for r, p in zip(test_rows, probs):
        rr = dict(r)
        rr["success_prob"] = float(p)
        rr["pce_source"] = "gsm8k_light_zero_shot"
        rr["feature_set"] = args.feature_set
        out.append(rr)

    write_jsonl(args.out_jsonl, out)

    if args.model_out:
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.model_out, "wb") as f:
            pickle.dump(pipe, f)

    summary = {
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "positive_train": sum(y_train),
        "positive_rate_train": sum(y_train) / max(1, len(y_train)),
        "feature_set": args.feature_set,
        "out_jsonl": args.out_jsonl,
        "model_out": args.model_out,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
