#!/usr/bin/env python
from pathlib import Path
import re
import shutil

FILES = [
    Path("data/build_atom_rollout_labels.py"),
    Path("pce/train.py"),
    Path("pce/train_hf.py"),
    Path("experiments/run_sample_level_selection.py"),
    Path("experiments/run_local_rewrite_backtrack.py"),
    Path("experiments/run_selective_answer_judge.py"),
]

NEW_DATASETS = [
    "gsm8k",
    "strategyqa",
    "hotpotqa",
    "svamp",
    "asdiv",
    "math500",
    "mathqa",
]

pattern = re.compile(
    r"""choices\s*=\s*\[
        (?P<body>[^\]]*?['"]gsm8k['"][^\]]*?)
        \]""",
    re.VERBOSE | re.DOTALL,
)

for path in FILES:
    if not path.exists():
        print("[MISSING]", path)
        continue

    backup = path.with_suffix(path.suffix + ".bak_cross_dataset")
    if not backup.exists():
        shutil.copy2(path, backup)

    text = path.read_text(encoding="utf-8")

    def repl(match):
        body = match.group("body")
        # 只替换包含 GSM8K 且看起来是数据集列表的 choices。
        if "strategyqa" not in body and "hotpotqa" not in body:
            return match.group(0)

        indent_match = re.search(r"\n(\s*)['\"]", body)
        indent = indent_match.group(1) if indent_match else "            "

        values = ",\n".join(
            f'{indent}"{name}"' for name in NEW_DATASETS
        )
        return "choices=[\n" + values + "\n        ]"

    updated, count = pattern.subn(repl, text)

    if count:
        path.write_text(updated, encoding="utf-8")
        print("[PATCHED]", path, "matches=", count)
    else:
        print("[UNCHANGED]", path)
