import argparse
from pathlib import Path
import yaml

def patch_model_path(obj, model_path):
    hit = False
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in {"model", "model_path", "model_name", "model_name_or_path", "pretrained_model_name_or_path"}:
                obj[k] = model_path
                hit = True
            else:
                if patch_model_path(v, model_path):
                    hit = True
    elif isinstance(obj, list):
        for v in obj:
            if patch_model_path(v, model_path):
                hit = True
    return hit

ap = argparse.ArgumentParser()
ap.add_argument("--template", default="configs/model/generator_llama_local_rewrite.yaml")
ap.add_argument("--out", required=True)
ap.add_argument("--model_path", required=True)
args = ap.parse_args()

cfg = yaml.safe_load(open(args.template, "r", encoding="utf-8")) or {}
hit = patch_model_path(cfg, args.model_path)
if not hit:
    cfg["model_name_or_path"] = args.model_path

cfg["max_model_len"] = 2048
cfg["gpu_memory_utilization"] = 0.60
cfg["enforce_eager"] = True
cfg["trust_remote_code"] = True
cfg["tensor_parallel_size"] = 1

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

print("saved:", args.out)
