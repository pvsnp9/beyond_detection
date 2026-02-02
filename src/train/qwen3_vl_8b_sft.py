from __future__ import annotations

import json
import os
from datetime import datetime

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from config.logistics import ModelCards, build_cfg
from src.collators.qwen3_collator import Qwen3VisionSFTCollator
from src.datasets.loader import load_hf_dataset
from src.utils.env import resolve_tokens_and_env, set_runtime_env
from src.utils.eval_sarcasm import evaluate_qwen3
from src.utils.logging import init_wandb, log_run_metadata
from src.utils.sft_utils import (
    LogCallback,
    _count_trainable_params,
    _dataset_size,
    _dtype_from_str,
    _get_world_size,
    _load_eval_dataset,
    _select_subset,
)


def _load_processor(model_name_or_path: str):
    # Some Qwen3-VL stacks behave better with slow tokenizer depending on versions.
    try:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
    except Exception as exc:
        print(f"AutoProcessor(use_fast=True) failed, retrying use_fast=False: {exc}")
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )

    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    # Critical sanity: if this is missing, you'll later see "unexpected keyword argument 'images'"
    # because you actually loaded a tokenizer-ish object.
    if not hasattr(processor, "image_processor"):
        raise RuntimeError(
            "Loaded processor has no image_processor (not a multimodal Qwen3-VL processor). "
            "This usually means your transformers version is too old or the model id isn't a VL checkpoint."
        )

    return processor


def _load_model(model_name_or_path: str, cfg, bnb_config: BitsAndBytesConfig):
    model_kwargs = dict(
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=_dtype_from_str(cfg["model"]["torch_dtype"]),
        low_cpu_mem_usage=True,
    )

    # Prefer FA2 when requested/available; otherwise SDPA; otherwise default.
    if cfg["model"]["use_flash_attention"] and cfg["sft_extras"].use_flash_attention_2:
        try:
            return Qwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except Exception as exc:
            print(f"FlashAttention-2 unavailable, loading default attention: {exc}")

    try:
        return Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            attn_implementation="sdpa",
            **model_kwargs,
        )
    except Exception as exc:
        print(f"SDPA attention unavailable, loading default attention: {exc}")
        return Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )


def main() -> None:
    try:
        cfg = build_cfg(ModelCards().qwen3_vl_8b_instruct)
        set_seed(cfg["sft"].seed)

        cfg["sft"].max_length = 2048
        cfg["mode"] = "rationale_sft"
        # these changing since we use A100 80G
        cfg["sft"].batch_size = 8
        cfg["sft"].gradient_accumulation_steps = 8 #64 batch
        cfg["sft"].lr = 2e-4

        resolved = resolve_tokens_and_env(cfg["logistics"])
        set_runtime_env(resolved)

        if resolved.get("hf_token"):
            try:
                from huggingface_hub import login

                login(token=resolved["hf_token"])
            except Exception as exc:
                print(f"Hugging Face login skipped: {exc}")

        run_name = cfg["output"]["run_name"] or datetime.now().strftime("sft-%Y%m%d-%H%M%S")
        run = init_wandb(cfg["wandb"], run_name)

        adapter_output_dir = cfg["output"]["adapter_output_dir"] or os.path.join(
            cfg["sft"].model_dir, "sft", "qwen3_vl_8b"
        )
        os.makedirs(adapter_output_dir, exist_ok=True)
        os.makedirs(cfg["logistics"].sft_log_dir, exist_ok=True)

        # --- data ---
        train_datasets = load_hf_dataset(
            cfg["dataset"]["dataset_id"],
            split=cfg["dataset"]["train_split"],
            config_name=cfg["dataset"]["lang"],
            streaming=cfg["dataset"]["streaming"],
            cache_dir=cfg["logistics"].hf_cache_dir,
            mode=cfg.get("mode")
        )
        train_datasets = _select_subset(train_datasets, cfg["dataset"]["max_train_samples"])

        eval_dataset, eval_split = _load_eval_dataset(cfg)
        eval_dataset = _select_subset(eval_dataset, cfg["dataset"]["max_eval_samples"])

        torch.backends.cuda.matmul.allow_tf32 = cfg["sft_extras"].tf32
        torch.backends.cudnn.allow_tf32 = cfg["sft_extras"].tf32

        # --- processor/model ---
        model_name_or_path = cfg["model"]["base_model_name_or_path"]
        processor = _load_processor(model_name_or_path)

        qlora_cfg = cfg["qlora"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_cfg.load_in_4bit,
            bnb_4bit_use_double_quant=qlora_cfg.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=qlora_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_dtype_from_str(qlora_cfg.bnb_4bit_compute_dtype),
        )

        model = _load_model(model_name_or_path, cfg, bnb_config)
        model.config.use_cache = False

        # Ensure pad ids are consistent (avoids occasional generation / loss weirdness)
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and tok.pad_token_id is not None:
            model.config.pad_token_id = tok.pad_token_id
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.pad_token_id = tok.pad_token_id

        # Gradient checkpointing (do before kbit prep)
        if cfg["model"]["gradient_checkpointing"] or cfg["sft_extras"].gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # QLoRA prep (PEFT API differs across versions; support both)
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=(
                    cfg["model"]["gradient_checkpointing"] or cfg["sft_extras"].gradient_checkpointing
                ),
            )
        except TypeError:
            model = prepare_model_for_kbit_training(model)

        # LoRA
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
        print(f"Trainable params: {params_summary['trainable_params']} / {params_summary['total_params']}")

        # --- collators (your new Option-B masking lives here) ---
        train_collator = Qwen3VisionSFTCollator(
            processor=processor,
            training=True,
            max_length=cfg["sft"].max_length,
            image_key=cfg["collator"]["image_key"],
            query_key=cfg["collator"]["query_key"],
            caption_key=cfg["collator"]["caption_key"],
            target_key=cfg["collator"]["target_key"],
            system_prompt=cfg["collator"]["system_prompt"],
        )
        eval_collator = Qwen3VisionSFTCollator(
            processor=processor,
            training=False,
            max_length=cfg["sft"].max_length,
            image_key=cfg["collator"]["image_key"],
            query_key=cfg["collator"]["query_key"],
            caption_key=cfg["collator"]["caption_key"],
            target_key=cfg["collator"]["target_key"],
            system_prompt=cfg["collator"]["system_prompt"],
        )

        # Sanity: ensure loss-bearing tokens exist (Option B should fix the earlier vision-token leak)
        sp = cfg["sft"]
        batch = next(iter(DataLoader(train_datasets, batch_size=sp.batch_size, collate_fn=train_collator)))
        print("supervised tokens per row:", (batch["labels"] != -100).sum(dim=1))
        assert (batch["labels"] != -100).any(), "All labels are -100 — no loss will flow."

        # --- trainer config ---
        sft_args = SFTConfig(
            output_dir=adapter_output_dir,
            run_name=run_name,
            num_train_epochs=sp.num_epochs,
            per_device_train_batch_size=sp.batch_size,
            gradient_accumulation_steps=sp.gradient_accumulation_steps,
            learning_rate=sp.lr,
            weight_decay=sp.weight_decay,
            warmup_ratio=sp.warmup_ratio,
            lr_scheduler_type="cosine",
            max_grad_norm=sp.max_grad_norm,
            # max_seq_length=sp.max_length, #droped in new trl
            bf16=cfg["sft_extras"].bf16,
            fp16=sp.fp16,
            tf32=cfg["sft_extras"].tf32,
            optim=cfg["sft_extras"].optim,
            logging_steps=sp.logging_steps,
            save_steps=sp.save_steps,
            eval_steps=sp.eval_steps,
            save_total_limit=sp.save_total_limit,
            load_best_model_at_end=sp.load_best_model_at_end,
            metric_for_best_model=sp.metric_for_best_model,
            greater_is_better=sp.greater_is_better,
            eval_strategy=cfg["sft_extras"].evaluation_strategy,
            report_to=["wandb"] if run is not None else [],
            disable_tqdm=cfg["sft_extras"].disable_tqdm,
            dataloader_num_workers=sp.num_workers,
            dataloader_pin_memory=cfg["sft_extras"].dataloader_pin_memory,
            dataloader_persistent_workers=cfg["sft_extras"].dataloader_persistent_workers,
            dataloader_prefetch_factor=cfg["sft_extras"].dataloader_prefetch_factor,
            # TRL requires this field but we're skipping dataset prep; keep it stable.
            dataset_text_field=cfg["collator"]["query_key"],
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_datasets,
            eval_dataset=eval_dataset,
            data_collator=train_collator,
            args=sft_args,
        )
        trainer.add_callback(LogCallback(sp.logging_steps))

        # pre-eval
        pre_metrics = evaluate_qwen3(model, processor, eval_dataset, eval_collator, cfg)
        if run is not None:
            run.log({f"gen_eval/pre_{k}": v for k, v in pre_metrics.items()})

        last_checkpoint = get_last_checkpoint(adapter_output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)

        # post-eval
        post_metrics = evaluate_qwen3(model, processor, eval_dataset, eval_collator, cfg)
        if run is not None:
            run.log({f"gen_eval/post_{k}": v for k, v in post_metrics.items()})

        trainer.model.save_pretrained(adapter_output_dir, safe_serialization=True)
        processor.save_pretrained(adapter_output_dir)

        training_meta = {
            "base_model_name_or_path": model_name_or_path,
            "dataset_id": cfg["dataset"]["dataset_id"],
            "lang": cfg["dataset"]["lang"],
            "effective_batch_size": sp.batch_size * sp.gradient_accumulation_steps * _get_world_size(),
            "flash_attention_enabled": cfg["model"]["use_flash_attention"] and cfg["sft_extras"].use_flash_attention_2,
            "config_source": "config/logistics.py",
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(adapter_output_dir, "training_meta.json"), "w") as f:
            json.dump(training_meta, f, indent=2)

        if not os.path.exists(os.path.join(adapter_output_dir, "adapter_config.json")):
            raise FileNotFoundError("adapter_config.json missing in adapter output dir")
        if not any(fname.startswith("adapter_model") for fname in os.listdir(adapter_output_dir)):
            raise FileNotFoundError("adapter_model weights missing in adapter output dir")

        meta = {
            "dataset_id": cfg["dataset"]["dataset_id"],
            "dataset_lang": cfg["dataset"]["lang"],
            "eval_split": eval_split,
            "train_size": _dataset_size(train_datasets),
            "eval_size": _dataset_size(eval_dataset),
            "base_model_id": model_name_or_path,
            "effective_batch_size": training_meta["effective_batch_size"],
            "flash_attention_enabled": training_meta["flash_attention_enabled"],
            "adapter_output_dir": adapter_output_dir,
            **params_summary,
        }
        log_run_metadata(run, meta)

        if run is not None:
            run.finish()

    except Exception as exc:
        raise RuntimeError(f"Qwen3-VL SFT failed: {exc}") from exc


if __name__ == "__main__":
    main()
