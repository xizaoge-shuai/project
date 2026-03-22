from __future__ import annotations
import argparse
from pathlib import Path
from datasets import load_dataset
import json


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def download_gsm8k(base: Path) -> None:
    ds = load_dataset("gsm8k", "main")
    out_dir = base / "gsm8k"
    ensure_dir(out_dir)

    for split in ["train", "test"]:
        rows = []
        for i, item in enumerate(ds[split]):
            rows.append(
                {
                    "id": f"gsm8k_{split}_{i}",
                    "question": item["question"],
                    "answer": item["answer"],
                    "task": "gsm8k",
                    "context": "",
                }
            )
        write_jsonl(out_dir / f"{split}.jsonl", rows)


def download_strategyqa(base: Path) -> None:
    ds = load_dataset("metaeval/strategy-qa")
    out_dir = base / "strategyqa"
    ensure_dir(out_dir)

    for split in ds.keys():
        rows = []
        for i, item in enumerate(ds[split]):
            answer = item.get("answer", item.get("label", None))
            if isinstance(answer, bool):
                answer = "yes" if answer else "no"
            rows.append(
                {
                    "id": f"strategyqa_{split}_{i}",
                    "question": item["question"],
                    "answer": str(answer).lower(),
                    "task": "strategyqa",
                    "context": "",
                }
            )
        write_jsonl(out_dir / f"{split}.jsonl", rows)


def download_hotpotqa(base: Path) -> None:
    ds = load_dataset("hotpot_qa", "distractor")
    out_dir = base / "hotpotqa"
    ensure_dir(out_dir)

    for split in ["train", "validation"]:
        rows = []
        for i, item in enumerate(ds[split]):
            context_sentences = []
            context = item.get("context", {})
            titles = context.get("title", [])
            sentences = context.get("sentences", [])
            for t, sents in zip(titles, sentences):
                joined = " ".join(sents)
                context_sentences.append(f"{t}: {joined}")

            rows.append(
                {
                    "id": f"hotpotqa_{split}_{i}",
                    "question": item["question"],
                    "answer": item["answer"],
                    "task": "hotpotqa",
                    "context": "\n".join(context_sentences),
                }
            )
        write_jsonl(out_dir / f"{split}.jsonl", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="all", choices=["all", "gsm8k", "strategyqa", "hotpotqa"]
    )
    parser.add_argument("--output_dir", default="data/raw")
    args = parser.parse_args()

    base = Path(args.output_dir)
    ensure_dir(base)

    if args.dataset in ("all", "gsm8k"):
        print("Downloading GSM8K...")
        download_gsm8k(base)

    if args.dataset in ("all", "strategyqa"):
        print("Downloading StrategyQA...")
        download_strategyqa(base)

    if args.dataset in ("all", "hotpotqa"):
        print("Downloading HotpotQA...")
        download_hotpotqa(base)

    print(f"Done. Raw datasets saved under {base}")


if __name__ == "__main__":
    main()
