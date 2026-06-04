#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import re
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

from vllm import LLM, SamplingParams


def read_jsonl(fp):
    fp = Path(fp)
    rows = []
    if not fp.exists():
        return rows
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_ids(fp):
    fp = Path(fp)
    if not fp.exists():
        return None
    return set(x.strip() for x in fp.read_text(encoding="utf-8").splitlines() if x.strip())


def get_sid(r):
    for k in ["sample_id", "id", "qid", "question_id"]:
        if k in r and r[k] is not None:
            return str(r[k])
    return None


def extract_question(r):
    for k in ["question", "problem", "input", "prompt"]:
        if k in r and r[k]:
            return str(r[k])
    return ""


def extract_text(r):
    for k in ["trajectory", "text", "reasoning", "output", "completion", "response"]:
        if k in r and r[k]:
            return str(r[k])
    return ""


def normalize_numeric(x):
    if x is None:
        return ""
    s = str(x).strip()
    s = s.replace(",", "").replace("$", "").replace("\\$", "")
    s = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"boxed\{([^{}]+)\}", r"\1", s)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", s)
    if nums:
        s = nums[-1]
    if "/" in s:
        try:
            a, b = s.split("/", 1)
            return str(round(float(a) / float(b), 10)).rstrip("0").rstrip(".")
        except Exception:
            pass
    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.10f}".rstrip("0").rstrip(".")
    except Exception:
        return re.sub(r"\s+", "", s.lower())


def normalize_choice(x):
    if x is None:
        return ""
    s = str(x).strip().lower()
    m = re.search(r"\b([abcde])\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\(([abcde])\)", s)
    if m:
        return m.group(1)
    return s[:1]


def norm_answer(x, task_type):
    return normalize_choice(x) if task_type == "choice" else normalize_numeric(x)


def extract_answer(r, task_type):
    for k in ["answer", "final_answer", "pred_answer", "prediction", "majority_answer", "extracted_answer"]:
        if k in r and r[k] is not None:
            a = norm_answer(r[k], task_type)
            if a:
                return a

    text = extract_text(r)
    patterns = [
        r"Final Answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*is\s*([^\n]+)",
        r"Answer\s*[:：]\s*([^\n]+)",
        r"答案\s*[:：]\s*([^\n]+)",
    ]
    for p in patterns:
        m = re.findall(p, text, flags=re.I)
        if m:
            return norm_answer(m[-1], task_type)

    return norm_answer(text[-300:], task_type)


def load_config(fp):
    fp = Path(fp)
    if yaml is not None:
        return yaml.safe_load(fp.read_text(encoding="utf-8")) or {}

    cfg = {}
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        cfg[k.strip()] = v.strip().strip("'").strip('"')
    return cfg


def get_model_path(cfg):
    for k in ["model_name_or_path", "model", "model_path", "name_or_path", "path"]:
        if k in cfg and cfg[k]:
            return str(cfg[k])
    raise ValueError("Cannot find model path in generator_config. Need one of model_name_or_path/model/model_path/name_or_path/path.")


def bool_cfg(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def build_prompt(question, solution, answer, max_solution_chars):
    solution = str(solution)
    if max_solution_chars and len(solution) > max_solution_chars:
        solution = solution[-max_solution_chars:]

    return (
        "You are verifying a math reasoning path.\n\n"
        "Problem:\n"
        f"{question}\n\n"
        "Candidate reasoning:\n"
        f"{solution}\n\n"
        "Candidate final answer:\n"
        f"{answer}\n\n"
        "Is the candidate final answer correct for the problem? "
        "Answer with exactly one token: 1 for correct, 0 for incorrect.\n"
        "Answer:"
    )


def get_logprob_value(obj):
    if obj is None:
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, dict):
        if "logprob" in obj:
            return float(obj["logprob"])
        return None
    if hasattr(obj, "logprob"):
        return float(obj.logprob)
    return None


def get_token_text(obj):
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return str(obj.get("decoded_token", obj.get("token", "")))
    for attr in ["decoded_token", "token"]:
        if hasattr(obj, attr):
            return str(getattr(obj, attr))
    return ""


def ptrue_from_vllm_output(out):
    try:
        gen_text = out.outputs[0].text.strip()
    except Exception:
        gen_text = ""

    lp0 = None
    lp1 = None

    try:
        logprobs0 = out.outputs[0].logprobs[0]
    except Exception:
        logprobs0 = None

    if isinstance(logprobs0, dict):
        for _, obj in logprobs0.items():
            tok = get_token_text(obj)
            lp = get_logprob_value(obj)
            if lp is None:
                continue

            clean = tok.replace("▁", " ").strip()
            if clean == "1":
                lp1 = lp if lp1 is None else max(lp1, lp)
            elif clean == "0":
                lp0 = lp if lp0 is None else max(lp0, lp)

    if lp1 is not None and lp0 is not None:
        m = max(lp1, lp0)
        p1 = math.exp(lp1 - m)
        p0 = math.exp(lp0 - m)
        return p1 / max(p1 + p0, 1e-12), gen_text, "logprob_1_vs_0"

    # fallback：如果 top logprobs 没拿到 0/1，就用生成结果粗略兜底
    if gen_text.startswith("1"):
        return 0.90, gen_text, "fallback_generated_1"
    if gen_text.startswith("0"):
        return 0.10, gen_text, "fallback_generated_0"
    return 0.50, gen_text, "fallback_unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator_config", required=True)
    ap.add_argument("--target_jsonl", required=True)
    ap.add_argument("--extra_jsonls", nargs="+", required=True)
    ap.add_argument("--target_ids", required=True)
    ap.add_argument("--task_type", choices=["numeric", "choice"], default="numeric")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_solution_chars", type=int, default=6000)
    ap.add_argument("--logprobs", type=int, default=20)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.generator_config)
    model_path = get_model_path(cfg)

    target_ids = load_ids(args.target_ids)
    qmap = {}
    for r in read_jsonl(args.target_jsonl):
        sid = get_sid(r)
        if sid:
            qmap[sid] = extract_question(r)

    out_fp = Path(args.out_jsonl)
    out_fp.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_fp.exists():
        for r in read_jsonl(out_fp):
            done.add((str(r.get("sample_id")), int(r.get("candidate_index", -1))))

    items = []
    cand_idx_by_sid = {}

    for efp in args.extra_jsonls:
        for r in read_jsonl(efp):
            sid = get_sid(r)
            if sid is None:
                continue
            if target_ids is not None and sid not in target_ids:
                continue

            idx = cand_idx_by_sid.get(sid, 0)
            cand_idx_by_sid[sid] = idx + 1

            if (sid, idx) in done:
                continue

            question = qmap.get(sid) or extract_question(r)
            text = extract_text(r)
            ans = extract_answer(r, args.task_type)

            prompt = build_prompt(question, text, ans, args.max_solution_chars)

            items.append({
                "sample_id": sid,
                "candidate_index": idx,
                "answer": ans,
                "source_file": str(efp),
                "prompt": prompt,
            })

            if args.limit and len(items) >= args.limit:
                break
        if args.limit and len(items) >= args.limit:
            break

    print("model_path =", model_path)
    print("num_to_score =", len(items))
    print("out_jsonl =", out_fp)

    if not items:
        print("nothing to score")
        return

    llm_kwargs = {
        "model": model_path,
        "trust_remote_code": bool_cfg(cfg.get("trust_remote_code"), True),
        "dtype": cfg.get("dtype", "auto"),
        "gpu_memory_utilization": float(cfg.get("gpu_memory_utilization", 0.90)),
    }

    if cfg.get("tensor_parallel_size") is not None:
        llm_kwargs["tensor_parallel_size"] = int(cfg.get("tensor_parallel_size"))
    if cfg.get("max_model_len") is not None:
        llm_kwargs["max_model_len"] = int(cfg.get("max_model_len"))
    if cfg.get("enforce_eager") is not None:
        llm_kwargs["enforce_eager"] = bool_cfg(cfg.get("enforce_eager"))

    print("llm_kwargs =", llm_kwargs)

    llm = LLM(**llm_kwargs)
    effective_logprobs = min(int(args.logprobs), 20)
    if int(args.logprobs) != effective_logprobs:
        print(f"[WARN] vLLM max allowed sample logprobs is 20; cap logprobs {args.logprobs} -> {effective_logprobs}")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=effective_logprobs,
    )

    with out_fp.open("a", encoding="utf-8") as f:
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            prompts = [x["prompt"] for x in batch]
            outs = llm.generate(prompts, sampling_params)

            for item, out in zip(batch, outs):
                p_true, gen_text, src = ptrue_from_vllm_output(out)
                rec = {
                    "sample_id": item["sample_id"],
                    "candidate_index": item["candidate_index"],
                    "answer": item["answer"],
                    "p_true": p_true,
                    "ptrue_source": src,
                    "generated_verdict": gen_text,
                    "source_file": item["source_file"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            f.flush()
            print(f"[progress] {min(start + len(batch), len(items))}/{len(items)}")

    print("DONE", out_fp)


if __name__ == "__main__":
    main()
