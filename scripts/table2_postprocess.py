#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def options_of(script: Path) -> set[str]:
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return set(re.findall(r"--[A-Za-z0-9_-]+", result.stdout))


def choose(
    available: set[str],
    candidates: list[str],
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in available:
            return candidate

    if required:
        raise RuntimeError(
            "none of these options exist: "
            + ", ".join(candidates)
        )

    return None


def append_optional(
    command: list[str],
    available: set[str],
    candidates: list[str],
    value: str,
) -> None:
    option = choose(
        available,
        candidates,
        required=False,
    )
    if option is not None:
        command.extend([option, value])


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    required=True,
    choices=["ensemble", "guard"],
)
parser.add_argument("--inputs", nargs="+", required=True)
parser.add_argument("--base_acc", type=float, required=True)
parser.add_argument("--n_samples", type=int, required=True)
parser.add_argument("--out_json", required=True)
parser.add_argument("--out_jsonl", required=True)
parser.add_argument("--dry_run", type=int, default=0)
args = parser.parse_args()

if args.mode == "ensemble":
    script = Path(
        "experiments/"
        "apply_resample_confirm_ensemble_seedaware.py"
    )
else:
    script = Path(
        "experiments/apply_orig_majority_guard.py"
    )

if not script.exists():
    raise SystemExit(f"missing script: {script}")

available = options_of(script)
command = [sys.executable, str(script)]

if args.mode == "ensemble":
    input_option = choose(
        available,
        [
            "--input_jsonls",
            "--resample_jsonls",
            "--inputs",
            "--jsonls",
        ],
    )
    command.append(input_option)
    command.extend(args.inputs)

    append_optional(
        command,
        available,
        ["--min_total_support", "--total_min"],
        "3",
    )
    append_optional(
        command,
        available,
        ["--min_seed_support", "--seed_min"],
        "2",
    )
    append_optional(
        command,
        available,
        [
            "--current_keep_min_support",
            "--min_current_support",
            "--current_keep",
            "--currentkeep",
        ],
        "2",
    )

else:
    input_option = choose(
        available,
        ["--input_jsonl", "--input"],
    )
    command.extend([input_option, args.inputs[0]])

    append_optional(
        command,
        available,
        [
            "--min_orig_support",
            "--orig_majority_min_support",
            "--orig_support",
            "--min_orig_count",
        ],
        "2",
    )

append_optional(
    command,
    available,
    ["--base_acc"],
    str(args.base_acc),
)
append_optional(
    command,
    available,
    ["--n_samples", "--n_eval"],
    str(args.n_samples),
)

out_json_option = choose(
    available,
    ["--out_json"],
)
out_jsonl_option = choose(
    available,
    ["--out_jsonl"],
)

command.extend([
    out_json_option,
    args.out_json,
    out_jsonl_option,
    args.out_jsonl,
])

print("COMMAND:")
print(" ".join(command))

if not args.dry_run:
    subprocess.run(command, check=True)
