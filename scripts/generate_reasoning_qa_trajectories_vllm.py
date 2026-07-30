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


def truncate_text(s, max_chars):
    s = str(s or "")
    return s[:max_chars]


def build_prompt(r, dataset):
    q = r.get("question", "")
    ctx = r.get("context", "")

    if dataset == "strategyqa":
        return (
            "Answer the following question with careful reasoning. "
            "At the end, write exactly one of: Final Answer: yes or Final Answer: no.\n\n"
            f"Question:\n{q}\n\nReasoning:"
        )

    if dataset == "hotpotqa":
        ctx = truncate_text(ctx, 7000)
        return (
            "Answer the question using the given context. Keep the final answer short. "
            "At the end, write: Final Answer: <short answer>.\n\n"
            f"Context:\n{ctx}\n\n"
            f"Question:\n{q}\n\nReasoning:"
        )

    if dataset == "mathqa":
        return (
            "Solve the multiple-choice math problem. "
            "At the end, write only the option letter in the form Final Answer: a/b/c/d/e.\n\n"
            f"Problem:\n{q}\n\nOptions:\n{ctx}\n\nReasoning:"
        )

    return f"Question:\n{q}\n\nAnswer:"


def extract_final(text, dataset):
    s = str(text or "").strip()
    m = re.findall(r"Final Answer\s*[:：]\s*(.+)", s, flags=re.I)
    if m:
        ans = m[-1].strip().splitlines()[0].strip()
    else:
        lines = [x.strip() for x in s.splitlines() if x.strip()]
        ans = lines[-1] if lines else ""

    ans = ans.strip().strip(".").strip()

    if dataset == "strategyqa":
        low = ans.lower()
        if "yes" in low and "no" not in low:
            return "yes"
        if "no" in low and "yes" not in low:
            return "no"
        if low in {"true", "1"}:
            return "yes"
        if low in {"false", "0"}:
            return "no"
        return low

    if dataset == "mathqa":
        low = ans.lower()
        m = re.search(r"\b([abcde])\b", low)
        if m:
            return m.group(1)
        m = re.search(r"^([abcde])[\)\.:]", low)
        if m:
            return m.group(1)
        return low

    if dataset == "hotpotqa":
        # 去掉常见句子壳，只保留短答案
        ans = re.sub(r"^(the answer is|answer is)\s+", "", ans, flags=re.I).strip()
        return ans

    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--generator_config", required=True)
    ap.add_argument("--dataset", required=True, choices=["strategyqa", "hotpotqa", "mathqa"])
    ap.add_argument("--n_traj", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.9)
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
            tid = f"{sid}_traj_{j}_seed{args.seed}"
            if tid in done:
                continue
            tasks.append((r, sid, j, tid, build_prompt(r, args.dataset)))

    print("model:", model)
    print("dataset:", args.dataset)
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
            final_answer = extract_final(text, args.dataset)

            out_rows.append({
                "sample_id": sid,
                "id": sid,
                "trajectory_id": tid,
                "traj_id": j,
                "sampling_seed": args.seed,
                "dataset": args.dataset,
                "task": args.dataset,
                "question": r.get("question", ""),
                "context": r.get("context", ""),
                "choices": r.get("choices", {}),
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
