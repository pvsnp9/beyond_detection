from __future__ import annotations

import json
import os
from datetime import datetime

import torch
from transformers import BitsAndBytesConfig, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOTrainer

from config.logistics import ModelCards, build_dpo_cfg
from src.collators.dpo_inference_collator import llama_dpo_inference_collator
from src.datasets.loader import load_dpo_train_eval_dataset
from src.utils.env import resolve_tokens_and_env, set_runtime_env
from src.utils.logging import init_wandb, log_run_metadata
from src.utils.mdpo_utils import (
    _dtype_from_str,
    _get_world_size,
    _select_subset,
    build_dpo_config,
    load_sft_adapter_models,
    MDPOMetricsCallback,
    MDPOTrainMetricsCallback,
)


def main() -> None:
    cfg = build_dpo_cfg(ModelCards().llama3_2vl)
    set_seed(cfg["dpo"].seed)

    resolved = resolve_tokens_and_env(cfg["logistics"])
    set_runtime_env(resolved)

    if resolved.get("hf_token"):
        try:
            from huggingface_hub import login

            login(token=resolved["hf_token"])
        except Exception as exc:
            print(f"Hugging Face login skipped: {exc}")

    run_name = cfg["output"]["run_name"] or datetime.now().strftime("mdpo-%Y%m%d-%H%M%S")
    run = init_wandb(cfg["wandb"], run_name)

    adapter_output_dir = cfg["output"]["adapter_output_dir"] or os.path.join(
        cfg["dpo"].model_dir if hasattr(cfg["dpo"], "model_dir") else cfg["logistics"].models_output_dir,
        "mdpo",
        "llama32",
    )
    os.makedirs(adapter_output_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = cfg["dpo_extras"].tf32
    torch.backends.cudnn.allow_tf32 = cfg["dpo_extras"].tf32

    qlora_cfg = cfg["qlora"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.load_in_4bit,
        bnb_4bit_use_double_quant=qlora_cfg.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=qlora_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_dtype_from_str(qlora_cfg.bnb_4bit_compute_dtype),
    )

    model_key = "llama"
    model, processor, sft_checkpoint, base_model_id = load_sft_adapter_models(
        model_key,
        cfg,
        bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_fast=True,
    )

    if run is not None:
        run.config.update(
            {
                "full_base_model": base_model_id,
                "sft_adapter_checkpoint": sft_checkpoint,
            }
        )

    if getattr(processor, "tokenizer", None) is not None:
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "right"
    else:
        print("[DPO] Warning: processor.tokenizer missing; DPO preprocessing may fail.")

    ds = load_dpo_train_eval_dataset(
        config_name=cfg["dataset"]["lang"],
        processor=processor,
        streaming=cfg["dataset"]["streaming"],
        cache_dir=cfg["logistics"].hf_cache_dir,
    )
    train_dataset = ds.get("train", None)
    eval_dataset = ds.get("eval", None)
    if train_dataset is None or eval_dataset is None:
        print("Could not load the DPO train/eval dataset")
        raise RuntimeError("train or eval dataset is None")

    eval_dataset = _select_subset(eval_dataset, cfg["dataset"]["max_eval_samples"])
    model.config.use_cache = cfg["dpo_extras"].use_cache

    def prompt_collator(batch):
        return llama_dpo_inference_collator(
            batch,
            mode="both",
            processor=processor,
            max_length=cfg["dpo"].max_length,
        )

    dpo_args = build_dpo_config(cfg, adapter_output_dir, run_name)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )
    trainer.add_callback(MDPOTrainMetricsCallback(run))
    trainer.add_callback(
        MDPOMetricsCallback(
            model,
            processor,
            eval_dataset,
            prompt_collator,
            cfg,
        )
    )

    last_checkpoint = get_last_checkpoint(adapter_output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.model.save_pretrained(adapter_output_dir, safe_serialization=True)
    processor.save_pretrained(adapter_output_dir)

    training_meta = {
        "full_base_model": base_model_id,
        "sft_adapter_checkpoint": sft_checkpoint,
        "dataset_id": cfg["dataset"]["dataset_id"],
        "dataset_lang": cfg["dataset"]["lang"],
        "eval_split": "train_eval_split",
        "effective_batch_size": cfg["dpo"].batch_size
        * cfg["dpo"].gradient_accumulation_steps
        * _get_world_size(),
        "flash_attention_enabled": cfg["model"]["use_flash_attention"]
        and cfg["dpo_extras"].use_flash_attention_2,
        "config_source": "config/logistics.py",
        "timestamp": datetime.now().isoformat(),
    }
    try:
        with open(os.path.join(adapter_output_dir, "training_meta.json"), "w") as f:
            json.dump(training_meta, f, indent=2)
    except Exception as exc:
        print(f"Failed to write training metadata: {exc}")

    log_run_metadata(
        cfg["logistics"],
        run_name,
        adapter_output_dir,
        extra_metadata=training_meta,
    )


if __name__ == "__main__":
    main()
