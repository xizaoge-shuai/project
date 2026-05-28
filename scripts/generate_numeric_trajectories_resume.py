import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter

def read_jsonl(fp):
    p = Path(fp)
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def nlines(fp):
    p = Path(fp)
    if not p.exists():
        return 0
    return sum(1 for x in open(p, encoding="utf-8", errors="ignore") if x.strip())

def sample_key(r):
    x = (
        r.get("sample_id")
        or r.get("question_id")
        or r.get("qid")
        or r.get("problem_id")
        or r.get("id")
        or r.get("question")
        or r.get("problem")
        or r.get("input")
    )
    x = str(x)
    x = re.sub(r"_traj_\d+$", "", x)
    return x

def replace_arg(argv, name, value):
    argv = list(argv)
    if name in argv:
        i = argv.index(name)
        argv[i + 1] = str(value)
    else:
        argv += [name, str(value)]
    return argv

def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_traj", type=int, required=True)
    args, rest = ap.parse_known_args()

    inp = Path(args.input)
    out = Path(args.output)

    input_rows = read_jsonl(inp)
    expected = len(input_rows) * args.n_traj
    existing_n = nlines(out)

    print(f"[RESUME] input={inp} n_samples={len(input_rows)} n_traj={args.n_traj} expected_rows={expected}")
    print(f"[RESUME] output={out} existing_rows={existing_n}")

    if existing_n == expected:
        print(f"[RESUME][SKIP] already complete: {out}")
        return

    if existing_n == 0:
        print("[RESUME] no existing output, run original generator directly")
        cmd = [sys.executable, "scripts/generate_numeric_trajectories_local.py"] + sys.argv[1:]
        print("[RESUME][CMD]", " ".join(cmd))
        raise SystemExit(subprocess.call(cmd))

    existing_rows = read_jsonl(out)
    cnt = Counter(sample_key(r) for r in existing_rows)

    completed = {k for k, c in cnt.items() if c >= args.n_traj}
    missing_rows = [r for r in input_rows if sample_key(r) not in completed]

    print(f"[RESUME] completed_samples={len(completed)} missing_samples={len(missing_rows)}")
    print(f"[RESUME] missing_expected_rows={len(missing_rows) * args.n_traj}")

    if not missing_rows:
        print("[RESUME][WARN] no missing sample found but row count mismatch; keep existing output.")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_dir = out.parent / f".resume_tmp_{out.stem}_{ts}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    miss_inp = tmp_dir / "missing_input.jsonl"
    miss_out = tmp_dir / "missing_output.jsonl"

    write_jsonl(miss_inp, missing_rows)

    new_argv = sys.argv[1:]
    new_argv = replace_arg(new_argv, "--input", miss_inp)
    new_argv = replace_arg(new_argv, "--output", miss_out)

    cmd = [sys.executable, "scripts/generate_numeric_trajectories_local.py"] + new_argv
    print("[RESUME][CMD]", " ".join(map(str, cmd)))

    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"[RESUME][ERROR] original generator failed with code {ret}")
        raise SystemExit(ret)

    miss_n = nlines(miss_out)
    print(f"[RESUME] missing_output_rows={miss_n}")

    backup = out.with_suffix(out.suffix + f".bak_{ts}")
    shutil.copy2(out, backup)
    print(f"[RESUME] backup existing output to {backup}")

    with open(out, "a", encoding="utf-8") as fout:
        with open(miss_out, encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    fout.write(line)

    final_n = nlines(out)
    print(f"[RESUME] final_rows={final_n} expected={expected}")

    if final_n != expected:
        print("[RESUME][WARN] final rows != expected. Check duplicated/missing samples.")
    else:
        print(f"[RESUME][DONE] completed: {out}")

if __name__ == "__main__":
    main()
