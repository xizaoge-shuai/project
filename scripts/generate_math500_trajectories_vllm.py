import argparse
import json
import re
from pathlib import Path

import yaml
from vllm import LLM, SamplingParams


def read_jsonl(fp):
    return [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]


def append_jsonl(fp, rows):
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_yaml(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_key(obj, keys, default=None):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
        for v in obj.values():
            ans = find_key(v, keys, None)
            if ans is not None:
                return ans
    return default


def extract_balanced_boxed(text):
    s = str(text or "")
    key = r"\boxed{"
    idx = s.rfind(key)
    if idx < 0:
        return ""

    i = idx + len(key)
    depth = 1
    out = []

    while i < len(s):
        ch = s[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
        else:
            out.append(ch)
        i += 1

    return ""


def extract_final_answer(text):
    boxed = extract_balanced_boxed(text)
    if boxed:
        return boxed.strip()

    s = str(text or "")
    patterns = [
        r"Final Answer\s*[:：]\s*(.+)",
        r"The final answer is\s*(.+)",
        r"Therefore,\s*(.+)",
    ]
    for pat in patterns:
        m = re.findall(pat, s, flags=re.I)
        if m:
            return m[-1].strip().splitlines()[0].strip().rstrip(".")

    return ""


def build_prompt(question):
    return (
        "You are solving a math problem. Reason step by step. "
        "At the end, put only the final answer in \\boxed{...}.\n\n"
        f"Problem:\n{question}\n\n"
        "Solution:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--generator_config", required=True)
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--n_traj", type=int, default=3)
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    cfg = load_yaml(args.generator_config)

    model = (
        find_key(cfg, ["model_name_or_path", "model", "model_path", "pretrained_model_name_or_path"])
        or "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
    )

    tensor_parallel_size = int(find_key(cfg, ["tensor_parallel_size"], 1))
    max_model_len = int(find_key(cfg, ["max_model_len"], 8192))
    gpu_memory_utilization = float(find_key(cfg, ["gpu_memory_utilization"], 0.80))
    dtype = find_key(cfg, ["dtype"], "auto")
    trust_remote_code = bool(find_key(cfg, ["trust_remote_code"], True))

    rows = read_jsonl(args.input)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    out_path = Path(args.output)
    done = set()
    if out_path.exists():
        for line in out_path.open("r", encoding="utf-8"):
            if line.strip():
                try:
                    done.add(json.loads(line)["trajectory_id"])
                except Exception:
                    pass

    tasks = []
    for r in rows:
        sid = r.get("sample_id") or r.get("id")
        for j in range(args.n_traj):
            tid = f"{sid}_traj_{j}"
            if tid in done:
                continue
            tasks.append((r, sid, j, tid, build_prompt(r["question"])))

    print("model:", model)
    print("input samples:", len(rows))
    print("pending trajectories:", len(tasks))
    print("output:", args.output)

    if not tasks:
        print("[DONE] nothing to generate")
        return

    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    for start in range(0, len(tasks), args.batch_size):
        batch = tasks[start : start + args.batch_size]
        prompts = [x[4] for x in batch]
        outputs = llm.generate(prompts, sampling)

        out_rows = []
        for (r, sid, j, tid, _), out in zip(batch, outputs):
            text = out.outputs[0].text
            final_answer = extract_final_answer(text)

            out_rows.append({
                "sample_id": sid,
                "id": sid,
                "trajectory_id": tid,
                "traj_id": j,
                "dataset": args.dataset,
                "task": args.dataset,
                "question": r.get("question", ""),
                "gold_answer": r.get("gold_answer", r.get("answer", "")),
                "answer": r.get("answer", ""),
                "response": text,
                "generated_text": text,
                "final_answer": final_answer,
            })

        append_jsonl(args.output, out_rows)
        print(f"[progress] {min(start + len(batch), len(tasks))}/{len(tasks)}")

    print("[SAVED]", args.output)


if __name__ == "__main__":
    main()
