from __future__ import annotations

import json
import os
from datetime import datetime

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import DPOTrainer

from config.logistics import ModelCards, build_dpo_cfg
from src.collators.qwen3_dpo_collator import Qwen3VisionDPOCollator
from src.datasets.loader import load_hf_dpo_dataset
from src.utils.env import resolve_tokens_and_env, set_runtime_env
from src.utils.logging import init_wandb, log_run_metadata
from src.utils.mdpo_utils import (
    _count_trainable_params,
    _dtype_from_str,
    _get_world_size,
    _select_subset,
    build_dpo_config,
    evaluate_mdpo,
    load_mdpo_eval_dataset,
    qwen_prompt_collator,
    MDPOMetricsCallback,
)


def _freeze_model(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def main() -> None:
    cfg = build_dpo_cfg(ModelCards().qwen3_vl_8b_instruct)

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
        "qwen3",
    )
    os.makedirs(adapter_output_dir, exist_ok=True)

    train_dataset = load_hf_dpo_dataset(
        split=cfg["dataset"]["train_split"],
        config_name=cfg["dataset"]["lang"],
        streaming=cfg["dataset"]["streaming"],
        cache_dir=cfg["logistics"].hf_cache_dir,
    )
    train_dataset = _select_subset(train_dataset, cfg["dataset"]["max_train_samples"])

    eval_dataset, eval_split = load_mdpo_eval_dataset(cfg)
    eval_dataset = _select_subset(eval_dataset, cfg["dataset"]["max_eval_samples"])

    torch.backends.cuda.matmul.allow_tf32 = cfg["dpo_extras"].tf32
    torch.backends.cudnn.allow_tf32 = cfg["dpo_extras"].tf32

    model_name_or_path = cfg["model"]["base_model_name_or_path"]
    processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    if getattr(processor, "tokenizer", None) is not None:
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    qlora_cfg = cfg["qlora"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.load_in_4bit,
        bnb_4bit_use_double_quant=qlora_cfg.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=qlora_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_dtype_from_str(qlora_cfg.bnb_4bit_compute_dtype),
    )

    model_kwargs = dict(
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=_dtype_from_str(cfg["model"]["torch_dtype"]),
    )

    if cfg["model"]["use_flash_attention"] and cfg["dpo_extras"].use_flash_attention_2:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name_or_path, attn_implementation="flash_attention_2", **model_kwargs
            )
        except Exception as exc:
            print(f"FlashAttention-2 unavailable, loading default attention: {exc}")
            model = AutoModelForImageTextToText.from_pretrained(
                model_name_or_path, **model_kwargs
            )
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)

    ref_model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)
    _freeze_model(ref_model)
    ref_model.eval()

    model.config.use_cache = cfg["dpo_extras"].use_cache
    if cfg["model"]["gradient_checkpointing"] or cfg["dpo_extras"].gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)
    lora_cfg = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        target_modules=list(lora_cfg.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    params_summary = _count_trainable_params(model)
    print(
        f"Trainable params: {params_summary['trainable_params']} / {params_summary['total_params']}"
    )

    dpo_collator = Qwen3VisionDPOCollator(
        processor=processor,
        max_length=cfg["dpo"].max_length,
        image_key=cfg["collator"]["image_key"],
        prompt_key=cfg["collator"]["prompt_key"],
        caption_key=cfg["collator"]["caption_key"],
        chosen_key=cfg["collator"]["chosen_key"],
        rejected_key=cfg["collator"]["rejected_key"],
        system_prompt=cfg["collator"]["system_prompt"],
    )
    prompt_collator = qwen_prompt_collator(processor, cfg["dpo"].max_length)

    dpo_args = build_dpo_config(cfg, adapter_output_dir, run_name)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=dpo_collator,
    )
    trainer.add_callback(MDPOMetricsCallback(model, processor, eval_dataset, prompt_collator, cfg))

    pre_metrics = evaluate_mdpo(model, processor, eval_dataset, prompt_collator, cfg)
    if run is not None:
        run.log(pre_metrics)

    last_checkpoint = get_last_checkpoint(adapter_output_dir)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    post_metrics = evaluate_mdpo(model, processor, eval_dataset, prompt_collator, cfg)
    if run is not None:
        run.log(post_metrics)

    trainer.model.save_pretrained(adapter_output_dir, safe_serialization=True)
    processor.save_pretrained(adapter_output_dir)

    training_meta = {
        "base_model_name_or_path": model_name_or_path,
        "dataset_id": cfg["dataset"]["dataset_id"],
        "dataset_lang": cfg["dataset"]["lang"],
        "eval_split": eval_split,
        "effective_batch_size": cfg["dpo"].batch_size
        * cfg["dpo"].gradient_accumulation_steps
        * _get_world_size(),
        "flash_attention_enabled": cfg["model"]["use_flash_attention"]
        and cfg["dpo_extras"].use_flash_attention_2,
        "config_source": "config/logistics.py",
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(adapter_output_dir, "training_meta.json"), "w") as f:
        json.dump(training_meta, f, indent=2)

    log_run_metadata(
        cfg["logistics"],
        run_name,
        adapter_output_dir,
        extra_metadata=training_meta,
    )


if __name__ == "__main__":
    main()
