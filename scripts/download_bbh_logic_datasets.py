import json
from pathlib import Path
from datasets import load_dataset, get_dataset_config_names


TASKS = [
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "boolean_expressions",
    "formal_fallacies",
    "tracking_shuffled_objects_three_objects",
]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_bbh_task(task):
    sources = [
        ("maveriq/bigbenchhard", task),
        ("lukaemon/bbh", task),
    ]

    last_err = None
    for name, config in sources:
        try:
            ds = load_dataset(name, config)
            return name, ds
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Cannot load task={task}. Last error: {last_err}")


def main():
    print("Try available configs for maveriq/bigbenchhard / lukaemon/bbh if needed.")

    all_summary = []

    for task in TASKS:
        print("=" * 100)
        print("TASK:", task)

        source, ds = load_bbh_task(task)
        split = "train" if "train" in ds else list(ds.keys())[0]
        data = ds[split]

        raw_rows = []
        unified_rows = []

        for i, ex in enumerate(data):
            sid = f"bbh_{task}_{i}"

            q = ex.get("input", ex.get("question", ex.get("prompt", "")))
            ans = ex.get("target", ex.get("answer", ex.get("label", "")))

            raw = dict(ex)
            raw["id"] = sid
            raw["hf_source"] = source
            raw["hf_task"] = task
            raw["hf_split"] = split
            raw_rows.append(raw)

            unified_rows.append({
                "id": sid,
                "sample_id": sid,
                "question": str(q).strip(),
                "answer": str(ans).strip(),
                "gold_answer": str(ans).strip(),
                "task": "bbh_logic",
                "subtask": task,
                "context": "",
                "choices": {},
                "meta": {
                    "dataset": "bbh_logic",
                    "source": source,
                    "split": split,
                    "subtask": task,
                }
            })

        write_jsonl(f"data/raw/bbh_logic/{task}.jsonl", raw_rows)
        write_jsonl(f"data/processed/unified/bbh_logic/{task}.jsonl", unified_rows)

        all_summary.append({
            "task": task,
            "source": source,
            "split": split,
            "rows": len(unified_rows),
            "out": f"data/processed/unified/bbh_logic/{task}.jsonl",
        })

        print("source:", source)
        print("split:", split)
        print("rows:", len(unified_rows))
        print("example:", json.dumps(unified_rows[0], ensure_ascii=False, indent=2)[:1000])

    write_jsonl("outputs/logs/bbh_logic_download_summary.jsonl", all_summary)

    print("\n# Summary")
    print("| task | source | split | rows |")
    print("|---|---|---|---:|")
    for x in all_summary:
        print(f"| {x['task']} | {x['source']} | {x['split']} | {x['rows']} |")


if __name__ == "__main__":
    main()
