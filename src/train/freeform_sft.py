"""Freeform SFT: plain-text target ablation against the JSON-rationale SFT.

Trains the same models on the same dataset/query as the JSON SFT pipeline, but
the assistant target is plain text '<label>\\n<explanation>' (column
'target_text', derived at train time from target_json). Ablates whether the
structured JSON rationale scaffold helps sarcasm detection.

Usage:
    python -m src.train.freeform_sft <llama|gemma|qwen|aya>

Replaces nothing: the JSON SFT trainers (llama32sft.py, gemma3sft.py,
ayavisionsft.py, qwen3_vl_8b_sft.py) stay as-is. Run llama/gemma/aya in the
flash_attn_sft conda env (trl 0.12) and qwen in the qwen3vl env (trl 0.27);
SFTConfig kwargs are filtered against the installed trl version so the same
script works under both.

Env overrides (smoke tests):
    FREEFORM_MAX_TRAIN_SAMPLES, FREEFORM_MAX_EVAL_SAMPLES, FREEFORM_NUM_EPOCHS,
    FREEFORM_OUTPUT_DIR, FREEFORM_DISABLE_GEN_EVAL=1, FREEFORM_WANDB_PROJECT
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Mapping

import torch
from datasets import disable_caching
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from config.logistics import ModelCards, build_freeform_cfg
from src.collators.ayacollator import AyaVisionSFTCollator
from src.collators.gemma3_collator import Gemma3VisionSFTCollator
from src.collators.llama32_collator import Llama32VisionSFTCollator
from src.collators.qwen3_collator import Qwen3VisionSFTCollator
from src.datasets.loader import FREEFORM_LABELS, add_freeform_target, load_hf_dataset
from src.utils.env import resolve_tokens_and_env, set_runtime_env
from src.utils.eval_sarcasm import (
    evaluate_aya,
    evaluate_gemma,
    evaluate_llama,
    evaluate_qwen3,
)
from src.utils.logging import init_wandb, log_run_metadata
from src.utils.sft_utils import (
    EvalCallback,
    LogCallback,
    _count_trainable_params,
    _dataset_size,
    _dtype_from_str,
    _get_world_size,
    _load_eval_dataset,
    _select_subset,
)


# --------------------------------------------------------------------------
# Per-model processor/model builders (mirror the JSON SFT trainers exactly)
# --------------------------------------------------------------------------

def _default_processor(model_name_or_path: str, padding_side: str):
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if hasattr(processor, "padding_side"):
        processor.padding_side = padding_side
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = padding_side
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    return processor


def _load_processor_llama(model_name_or_path: str):
    return _default_processor(model_name_or_path, padding_side="left")


def _load_processor_gemma(model_name_or_path: str):
    return _default_processor(model_name_or_path, padding_side="right")


def _load_processor_qwen(model_name_or_path: str):
    # Lazy import: qwen3_vl_8b_sft imports Qwen3VLForConditionalGeneration at
    # module level, which only exists in the qwen3vl env (transformers>=4.57).
    from src.train.qwen3_vl_8b_sft import _load_processor as qwen_load_processor

    return qwen_load_processor(model_name_or_path)


def _load_processor_aya(model_name_or_path: str):
    processor = _default_processor(model_name_or_path, padding_side="right")
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.add_special_tokens({"additional_special_tokens": ["<image>"]})
    return processor


def _fa2_model_kwargs(cfg: dict, bnb_config: BitsAndBytesConfig) -> dict:
    return dict(
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=_dtype_from_str(cfg["model"]["torch_dtype"]),
    )


def _load_model_fa2(model_name_or_path: str, cfg: dict, bnb_config, processor):
    """FA2 with fallback to default attention (llama and aya path)."""
    model_kwargs = _fa2_model_kwargs(cfg, bnb_config)
    if cfg["model"]["use_flash_attention"] and cfg["sft_extras"].use_flash_attention_2:
        try:
            return AutoModelForImageTextToText.from_pretrained(
                model_name_or_path,
                attn_implementation="flash_attention_2",
                **model_kwargs,
            )
        except Exception as exc:
            print(f"FlashAttention-2 unavailable, loading default attention: {exc}")
    else:
        print("flash_attention_2 not set, loading default attention")
    return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)


def _load_model_llama(model_name_or_path: str, cfg: dict, bnb_config, processor):
    return _load_model_fa2(model_name_or_path, cfg, bnb_config, processor)


def _load_model_gemma(model_name_or_path: str, cfg: dict, bnb_config, processor):
    return AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        attn_implementation="eager",
        **_fa2_model_kwargs(cfg, bnb_config),
    )


def _load_model_qwen(model_name_or_path: str, cfg: dict, bnb_config, processor):
    from src.train.qwen3_vl_8b_sft import _load_model as qwen_load_model

    return qwen_load_model(model_name_or_path, cfg, bnb_config)


def _load_model_aya(model_name_or_path: str, cfg: dict, bnb_config, processor):
    model = _load_model_fa2(model_name_or_path, cfg, bnb_config, processor)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and len(tok) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))
    return model


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FreeformModelSpec:
    model_card: str
    output_dirname: str
    load_processor: Callable[[str], Any]
    load_model: Callable[[str, dict, BitsAndBytesConfig, Any], Any]
    collator_cls: type
    eval_fn: Callable[..., Dict[str, Any]]
    sft_overrides: Mapping[str, Any] = field(default_factory=dict)


_CARDS = ModelCards()

FREEFORM_MODELS: Dict[str, FreeformModelSpec] = {
    "llama": FreeformModelSpec(
        model_card=_CARDS.llama3_2vl,
        output_dirname="llama32",
        load_processor=_load_processor_llama,
        load_model=_load_model_llama,
        collator_cls=Llama32VisionSFTCollator,
        eval_fn=evaluate_llama,
        sft_overrides={"batch_size": 8, "gradient_accumulation_steps": 8, "lr": 2e-4},
    ),
    "gemma": FreeformModelSpec(
        model_card=_CARDS.gemm3_12b,
        output_dirname="gemma3",
        load_processor=_load_processor_gemma,
        load_model=_load_model_gemma,
        collator_cls=Gemma3VisionSFTCollator,
        eval_fn=evaluate_gemma,
        sft_overrides={
            "max_length": 2048,
            "batch_size": 8,
            "gradient_accumulation_steps": 8,
            "lr": 2e-4,
        },
    ),
    "qwen": FreeformModelSpec(
        model_card=_CARDS.qwen3_vl_8b_instruct,
        output_dirname="qwen3_vl_8b",
        load_processor=_load_processor_qwen,
        load_model=_load_model_qwen,
        collator_cls=Qwen3VisionSFTCollator,
        eval_fn=evaluate_qwen3,
        sft_overrides={
            "max_length": 2048,
            "batch_size": 8,
            "gradient_accumulation_steps": 8,
            "lr": 2e-4,
        },
    ),
    "aya": FreeformModelSpec(
        model_card=_CARDS.aya_model_name,
        output_dirname="aya_vision_8b",
        load_processor=_load_processor_aya,
        load_model=_load_model_aya,
        collator_cls=AyaVisionSFTCollator,
        eval_fn=evaluate_aya,
        sft_overrides={
            "max_length": 2048,
            "batch_size": 4,
            "gradient_accumulation_steps": 16,
            "lr": 2e-4,
        },
    ),
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _build_sft_config(cfg: dict, run, run_name: str, adapter_output_dir: str) -> SFTConfig:
    """Build SFTConfig kwargs filtered against the installed trl version
    (trl 0.12 in flash_attn_sft vs trl 0.27 in qwen3vl)."""
    sp = cfg["sft"]
    extras = cfg["sft_extras"]
    kwargs = dict(
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
        max_seq_length=sp.max_length,
        bf16=extras.bf16,
        fp16=sp.fp16,
        tf32=extras.tf32,
        optim=extras.optim,
        logging_steps=sp.logging_steps,
        save_steps=sp.save_steps,
        eval_steps=sp.eval_steps,
        save_total_limit=sp.save_total_limit,
        load_best_model_at_end=sp.load_best_model_at_end,
        metric_for_best_model=sp.metric_for_best_model,
        greater_is_better=sp.greater_is_better,
        eval_strategy=extras.evaluation_strategy,
        report_to=["wandb"] if run is not None else [],
        disable_tqdm=extras.disable_tqdm,
        dataloader_num_workers=sp.num_workers,
        dataloader_pin_memory=extras.dataloader_pin_memory,
        dataloader_persistent_workers=extras.dataloader_persistent_workers,
        dataloader_prefetch_factor=extras.dataloader_prefetch_factor,
        # TRL requires this field but we're skipping dataset prep; keep it stable.
        dataset_text_field=cfg["collator"]["query_key"],
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )
    fields = SFTConfig.__dataclass_fields__
    supported = {k: v for k, v in kwargs.items() if k in fields}
    dropped = sorted(set(kwargs) - set(supported))
    if dropped:
        print(f"[freeform_sft] SFTConfig kwargs unsupported by this trl version, dropped: {dropped}")
    return SFTConfig(**supported)


def _sanity_check_batch(train_dataset, train_collator, processor, batch_size: int) -> None:
    batch = next(iter(DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_collator)))
    supervised = (batch["labels"] != -100)
    print("supervised tokens per row:", supervised.sum(dim=1))
    assert supervised.any(), "All labels are -100 — no loss will flow."

    row_labels = batch["labels"][0]
    target_ids = row_labels[row_labels != -100]
    decoded = processor.tokenizer.decode(target_ids, skip_special_tokens=True).strip()
    # Aya supervises its turn-marker tokens (<|...|>), which are added (not
    # special) tokens and survive skip_special_tokens; strip them before checking.
    decoded = re.sub(r"<\|[^|]+\|>", "", decoded).strip()
    first_line = decoded.splitlines()[0].strip() if decoded else ""
    assert first_line in FREEFORM_LABELS, (
        f"Supervised span does not start with a freeform label; got {first_line!r} "
        f"(decoded: {decoded[:120]!r})"
    )
    print(f"freeform target check OK, first line: {first_line!r}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    model_key = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
    if model_key not in FREEFORM_MODELS:
        raise SystemExit(f"usage: python -m src.train.freeform_sft <{'|'.join(FREEFORM_MODELS)}>")
    spec = FREEFORM_MODELS[model_key]

    try:
        cfg = build_freeform_cfg(spec.model_card)
        # Map results stay in process-private temp files: concurrent jobs racing
        # on the same shared cache file caused SIGBUS (same pattern as mdpo.py).
        disable_caching()
        set_seed(cfg["sft"].seed)
        for attr, value in spec.sft_overrides.items():
            setattr(cfg["sft"], attr, value)

        max_train_override = os.environ.get("FREEFORM_MAX_TRAIN_SAMPLES")
        if max_train_override:
            cfg["dataset"]["max_train_samples"] = int(max_train_override)
        max_eval_override = os.environ.get("FREEFORM_MAX_EVAL_SAMPLES")
        if max_eval_override:
            cfg["dataset"]["max_eval_samples"] = int(max_eval_override)
        epochs_override = os.environ.get("FREEFORM_NUM_EPOCHS")
        if epochs_override:
            cfg["sft"].num_epochs = int(epochs_override)
        output_dir_override = os.environ.get("FREEFORM_OUTPUT_DIR")
        if output_dir_override:
            # keep smoke-test checkpoints out of the real adapter dirs
            cfg["output"]["adapter_output_dir"] = os.path.join(
                output_dir_override, spec.output_dirname
            )

        resolved = resolve_tokens_and_env(cfg["logistics"])
        set_runtime_env(resolved)
        if resolved.get("hf_token"):
            try:
                from huggingface_hub import login

                login(token=resolved["hf_token"])
            except Exception as exc:
                print(f"Hugging Face login skipped: {exc}")

        run_name = cfg["output"]["run_name"] or datetime.now().strftime(
            f"freeform-{model_key}-%Y%m%d-%H%M%S"
        )
        run = init_wandb(cfg["wandb"], run_name)

        adapter_output_dir = cfg["output"]["adapter_output_dir"] or os.path.join(
            cfg["sft"].model_dir, "freeform_sft", spec.output_dirname
        )
        os.makedirs(adapter_output_dir, exist_ok=True)
        os.makedirs(cfg["logistics"].sft_log_dir, exist_ok=True)

        # --- data (target_text derived from target_json on both splits) ---
        train_dataset = load_hf_dataset(
            cfg["dataset"]["dataset_id"],
            split=cfg["dataset"]["train_split"],
            config_name=cfg["dataset"]["lang"],
            streaming=cfg["dataset"]["streaming"],
            cache_dir=cfg["logistics"].hf_cache_dir,
        )
        train_dataset = add_freeform_target(train_dataset)
        train_dataset = _select_subset(train_dataset, cfg["dataset"]["max_train_samples"])

        eval_dataset, eval_split = _load_eval_dataset(cfg)
        eval_dataset = add_freeform_target(eval_dataset)
        eval_dataset = _select_subset(eval_dataset, cfg["dataset"]["max_eval_samples"])

        torch.backends.cuda.matmul.allow_tf32 = cfg["sft_extras"].tf32
        torch.backends.cudnn.allow_tf32 = cfg["sft_extras"].tf32

        # --- processor/model ---
        model_name_or_path = cfg["model"]["base_model_name_or_path"]
        processor = spec.load_processor(model_name_or_path)

        qlora_cfg = cfg["qlora"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_cfg.load_in_4bit,
            bnb_4bit_use_double_quant=qlora_cfg.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=qlora_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=_dtype_from_str(qlora_cfg.bnb_4bit_compute_dtype),
        )
        model = spec.load_model(model_name_or_path, cfg, bnb_config, processor)
        model.config.use_cache = False

        tok = getattr(processor, "tokenizer", None)
        if tok is not None and tok.pad_token_id is not None:
            model.config.pad_token_id = tok.pad_token_id
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.pad_token_id = tok.pad_token_id

        if cfg["model"]["gradient_checkpointing"] or cfg["sft_extras"].gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # PEFT API differs across versions; support both.
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=(
                    cfg["model"]["gradient_checkpointing"]
                    or cfg["sft_extras"].gradient_checkpointing
                ),
            )
        except TypeError:
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

        # --- collators (target_text + FREEFORM_SYSTEM_PROMPT via cfg) ---
        collator_kwargs = dict(
            processor=processor,
            max_length=cfg["sft"].max_length,
            image_key=cfg["collator"]["image_key"],
            query_key=cfg["collator"]["query_key"],
            caption_key=cfg["collator"]["caption_key"],
            target_key=cfg["collator"]["target_key"],
            system_prompt=cfg["collator"]["system_prompt"],
        )
        train_collator = spec.collator_cls(training=True, **collator_kwargs)
        eval_collator = spec.collator_cls(training=False, **collator_kwargs)

        _sanity_check_batch(train_dataset, train_collator, processor, cfg["sft"].batch_size)

        sft_args = _build_sft_config(cfg, run, run_name, adapter_output_dir)

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=train_collator,
            args=sft_args,
        )
        trainer.add_callback(LogCallback(cfg["sft"].logging_steps))
        if _env_flag("FREEFORM_DISABLE_GEN_EVAL"):
            print("[freeform_sft] FREEFORM_DISABLE_GEN_EVAL set — skipping generation eval callback")
        else:
            trainer.add_callback(
                EvalCallback(spec.eval_fn, run, cfg, eval_dataset, eval_collator, processor)
            )

        last_checkpoint = get_last_checkpoint(adapter_output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)

        trainer.model.save_pretrained(adapter_output_dir, safe_serialization=True)
        processor.save_pretrained(adapter_output_dir)

        sp = cfg["sft"]
        training_meta = {
            "mode": "freeform_sft",
            "target_format": "label\\nexplanation",
            "base_model_name_or_path": model_name_or_path,
            "dataset_id": cfg["dataset"]["dataset_id"],
            "lang": cfg["dataset"]["lang"],
            "effective_batch_size": sp.batch_size * sp.gradient_accumulation_steps * _get_world_size(),
            "flash_attention_enabled": cfg["model"]["use_flash_attention"]
            and cfg["sft_extras"].use_flash_attention_2,
            "config_source": "config/logistics.py:build_freeform_cfg",
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(adapter_output_dir, "training_meta.json"), "w") as f:
            json.dump(training_meta, f, indent=2)

        if not os.path.exists(os.path.join(adapter_output_dir, "adapter_config.json")):
            raise FileNotFoundError("adapter_config.json missing in adapter output dir")
        if not any(fname.startswith("adapter_model") for fname in os.listdir(adapter_output_dir)):
            raise FileNotFoundError("adapter_model weights missing in adapter output dir")

        meta = {
            "mode": "freeform_sft",
            "dataset_id": cfg["dataset"]["dataset_id"],
            "dataset_lang": cfg["dataset"]["lang"],
            "eval_split": eval_split,
            "train_size": _dataset_size(train_dataset),
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
        raise RuntimeError(f"Freeform SFT ({model_key}) failed: {exc}") from exc


if __name__ == "__main__":
    main()
