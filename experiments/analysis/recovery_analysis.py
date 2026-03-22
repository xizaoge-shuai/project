from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from utils.io import read_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.traces)
    bt = [r for r in rows if r.get("backtrack_count", 0) > 0]
    if not bt:
        print({"recovery_rate": 0.0, "count": 0})
        return
    recovery_rate = sum(int(r["is_correct"]) for r in bt) / len(bt)
    print({"recovery_rate": recovery_rate, "count": len(bt)})

if __name__ == "__main__":
    main()
