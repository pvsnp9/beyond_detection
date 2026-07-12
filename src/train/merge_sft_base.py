"""One-time prep for mDPO: merge each SFT LoRA adapter into its bf16 base model.

Usage: python -m src.train.merge_sft_base <qwen|gemma|llama|all>

The merged models are the policy/reference starting point for mDPO: training
loads them 4-bit and attaches a fresh LoRA, so the DPO reference (adapter
disabled) is exactly the SFT policy.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer

from config.logistics import MDPOParams
from src.hf.publish_lora_models import get_sft_best_model
from src.inference.run_inference import BaseModels

# model_key -> (sft dir under outputs/models/sft, merged dir under sft_merged_root)
MERGE_DIRS = {
    "qwen": ("qwen3_vl_8b", "qwen3_vl_8b"),
    "gemma": ("gemma3", "gemma3"),
    "llama": ("llama32", "llama32"),
}


def merge_sft_into_base(model_key: str, out_root: str | None = None) -> str:
    if model_key not in MERGE_DIRS:
        raise ValueError(f"unsupported model_key: {model_key} (choose from {list(MERGE_DIRS)})")
    sft_dirname, merged_dirname = MERGE_DIRS[model_key]
    out_root = out_root or MDPOParams().sft_merged_root
    out_dir = os.path.join(out_root, merged_dirname)

    result = get_sft_best_model(sft_dirname)
    sft_checkpoint = result.get("best_model_path")
    if not sft_checkpoint:
        raise FileNotFoundError(f"no SFT checkpoint found for {model_key} ({sft_dirname})")

    base_models = BaseModels()
    base_model_id = base_models.models[model_key]
    model_class = base_models.get_class(model_key)

    print(f"[merge] {model_key}: base={base_model_id}")
    print(f"[merge] {model_key}: sft adapter={sft_checkpoint}")
    model = model_class.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, sft_checkpoint)
    model = model.merge_and_unload()

    os.makedirs(out_dir, exist_ok=True)
    print(f"[merge] {model_key}: saving merged model to {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="4GB")

    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True, use_fast=True)
    # prefer the SFT checkpoint tokenizer (legacy DPO loader behavior)
    try:
        tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True, use_fast=True)
        if getattr(processor, "tokenizer", None) is not None:
            processor.tokenizer = tokenizer
    except Exception as exc:
        print(f"[merge] {model_key}: falling back to base tokenizer: {exc}")
    processor.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "merge_meta.json"), "w") as f:
        json.dump(
            {
                "base_model_id": base_model_id,
                "sft_checkpoint": sft_checkpoint,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    del model
    torch.cuda.empty_cache()
    print(f"[merge] {model_key}: done -> {out_dir}")
    return out_dir


def main() -> None:
    arg = (sys.argv[1] if len(sys.argv) > 1 else "all").strip().lower()
    keys = list(MERGE_DIRS) if arg == "all" else [arg]
    for key in keys:
        merge_sft_into_base(key)


if __name__ == "__main__":
    main()
