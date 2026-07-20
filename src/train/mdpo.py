"""Unified mDPO training entrypoint.

Usage: python -m src.train.mdpo <qwen|gemma|llama|aya>

Trains a FRESH LoRA adapter (true mDPO objective, see src/train/mdpo_trainer.py)
on top of the SFT-merged base produced by src/train/merge_sft_base.py.
Replaces the per-model scripts qwen3mdpo.py / gemma3dpo.py / llama32dpo.py.

Smoke-test env overrides: MDPO_MAX_TRAIN_SAMPLES, MDPO_NUM_EPOCHS, MDPO_OUTPUT_DIR.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

import torch
from datasets import disable_caching
from peft import LoraConfig
from transformers import BitsAndBytesConfig, set_seed
from transformers.trainer_utils import get_last_checkpoint

from config.logistics import ModelCards, build_dpo_cfg
from src.collators.ayacollator import AyaVisionSFTCollator
from src.collators.gemma3_collator import Gemma3VisionSFTCollator
from src.collators.llama32_collator import Llama32VisionSFTCollator
from src.collators.qwen3_collator import Qwen3VisionSFTCollator
from src.datasets.loader import load_dpo_train_eval_dataset
from src.train.mdpo_trainer import MDPOPreferenceCollator, MDPOTrainer
from src.train.vision_strategies import VISION_STRATEGIES
from src.utils.env import resolve_tokens_and_env, set_runtime_env
from src.utils.logging import init_wandb, log_run_metadata
from src.utils.mdpo_utils import (
    _dtype_from_str,
    _get_world_size,
    _select_subset,
    build_dpo_config,
    load_merged_policy_model,
    MDPOMetricsCallback,
    MDPOTrainMetricsCallback,
)


@dataclass(frozen=True)
class MDPOModelSpec:
    model_card: str
    merged_dirname: str          # under MDPOParams.sft_merged_root
    adapter_dirname: str         # under outputs/models/mdpo (inference expects these names)
    flash_attn: bool
    build_eval_collator: Callable[[Any, dict], Any]
    # None -> LoRAParams.target_modules list; a str is a PEFT regex over module paths
    lora_target_modules: Optional[str] = None


def _build_qwen_eval_collator(processor: Any, cfg: dict) -> Any:
    return Qwen3VisionSFTCollator(
        processor=processor, training=False, max_length=cfg["dpo"].max_length
    )


def _build_gemma_eval_collator(processor: Any, cfg: dict) -> Any:
    return Gemma3VisionSFTCollator(
        processor=processor,
        training=False,
        max_length=cfg["dpo"].max_length,
        image_key=cfg["collator"]["image_key"],
        query_key=cfg["collator"]["prompt_key"],
        caption_key=cfg["collator"]["caption_key"],
        target_key="chosen",
        system_prompt=cfg["collator"]["system_prompt"],
    )


def _build_llama_eval_collator(processor: Any, cfg: dict) -> Any:
    return Llama32VisionSFTCollator(
        processor=processor,
        training=False,
        max_length=cfg["dpo"].max_length,
        image_key=cfg["collator"]["image_key"],
        query_key=cfg["collator"]["prompt_key"],
        caption_key=cfg["collator"]["caption_key"],
        target_key="chosen",
        system_prompt=cfg["collator"]["system_prompt"],
    )


def _build_aya_eval_collator(processor: Any, cfg: dict) -> Any:
    return AyaVisionSFTCollator(
        processor=processor,
        training=False,
        max_length=cfg["dpo"].max_length,
        image_key=cfg["collator"]["image_key"],
        query_key=cfg["collator"]["prompt_key"],
        caption_key=cfg["collator"]["caption_key"],
        target_key="chosen",
        system_prompt=cfg["collator"]["system_prompt"],
    )


MDPO_MODELS = {
    "qwen": MDPOModelSpec(
        model_card=ModelCards().qwen3_vl_8b_instruct,
        merged_dirname="qwen3_vl_8b",
        adapter_dirname="qwen3",
        flash_attn=True,
        build_eval_collator=_build_qwen_eval_collator,
    ),
    "gemma": MDPOModelSpec(
        # Gemma3 training is incompatible with flash_attention_2 in this build
        model_card=ModelCards().gemm3_12b,
        merged_dirname="gemma3",
        adapter_dirname="gemma3",
        flash_attn=False,
        build_eval_collator=_build_gemma_eval_collator,
    ),
    "llama": MDPOModelSpec(
        # Mllama vision attention is incompatible with flash_attention_2 in this build
        model_card=ModelCards().llama3_2vl,
        merged_dirname="llama32",
        adapter_dirname="llama32",
        flash_attn=False,
        build_eval_collator=_build_llama_eval_collator,
        # LM-only LoRA: the generic projection names also match Mllama's vision
        # tower, whose ~6.4k-token/image unchckpointed activations OOM an 80GB
        # A100 under mDPO's two live grad forwards. Vision stays frozen.
        lora_target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    ),
    "aya": MDPOModelSpec(
        # AyaVision FA2 is untested in this build; SigLIP tower + 364px single
        # tile keeps activations small, so eager attention is affordable
        model_card=ModelCards().aya_model_name,
        merged_dirname="aya_vision_8b",
        adapter_dirname="aya",
        flash_attn=False,
        build_eval_collator=_build_aya_eval_collator,
    ),
}


def _stale_checkpoint_guard(adapter_output_dir: str) -> Optional[str]:
    last_checkpoint = get_last_checkpoint(adapter_output_dir)
    if not last_checkpoint:
        return None
    meta_path = os.path.join(adapter_output_dir, "training_meta.json")
    merged_base_path = None
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as handle:
            merged_base_path = json.load(handle).get("merged_base_path")
    if not merged_base_path:
        raise RuntimeError(
            f"Checkpoints in {adapter_output_dir} come from the pre-revision pipeline "
            "(no merged_base_path in training_meta.json). Archive them, e.g. "
            "`mv` into outputs/models/mdpo_archive_<date>/, before training."
        )
    return last_checkpoint


def main() -> None:
    model_key = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
    if model_key not in MDPO_MODELS:
        raise SystemExit(f"usage: python -m src.train.mdpo <{'|'.join(MDPO_MODELS)}>")
    spec = MDPO_MODELS[model_key]

    cfg = build_dpo_cfg(spec.model_card)
    # Avoid reusing stale tokenization/map artifacts across runs.
    disable_caching()
    set_seed(cfg["dpo"].seed)

    max_train_override = os.environ.get("MDPO_MAX_TRAIN_SAMPLES")
    if max_train_override:
        cfg["dataset"]["max_train_samples"] = int(max_train_override)
    epochs_override = os.environ.get("MDPO_NUM_EPOCHS")
    if epochs_override:
        cfg["dpo"].num_epochs = int(epochs_override)
    output_dir_override = os.environ.get("MDPO_OUTPUT_DIR")
    if output_dir_override:
        # keep smoke-test checkpoints out of the real adapter dirs
        cfg["output"]["adapter_output_dir"] = os.path.join(output_dir_override, spec.adapter_dirname)

    resolved = resolve_tokens_and_env(cfg["logistics"])
    set_runtime_env(resolved)
    if resolved.get("hf_token"):
        try:
            from huggingface_hub import login

            login(token=resolved["hf_token"])
        except Exception as exc:
            print(f"Hugging Face login skipped: {exc}")

    run_name = cfg["output"]["run_name"] or datetime.now().strftime(f"mdpo-{model_key}-%Y%m%d-%H%M%S")
    run = init_wandb(cfg["wandb"], run_name)

    adapter_output_dir = cfg["output"]["adapter_output_dir"] or os.path.join(
        cfg["dpo"].model_dir, "mdpo", spec.adapter_dirname
    )
    os.makedirs(adapter_output_dir, exist_ok=True)
    last_checkpoint = _stale_checkpoint_guard(adapter_output_dir)

    torch.backends.cuda.matmul.allow_tf32 = cfg["dpo_extras"].tf32
    torch.backends.cudnn.allow_tf32 = cfg["dpo_extras"].tf32

    qlora_cfg = cfg["qlora"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.load_in_4bit,
        bnb_4bit_use_double_quant=qlora_cfg.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=qlora_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_dtype_from_str(qlora_cfg.bnb_4bit_compute_dtype),
    )

    use_flash_attention = (
        spec.flash_attn
        and cfg["model"]["use_flash_attention"]
        and cfg["dpo_extras"].use_flash_attention_2
    )
    model, processor, merged_dir, merge_meta = load_merged_policy_model(
        model_key,
        spec.merged_dirname,
        cfg,
        bnb_config,
        use_flash_attention=use_flash_attention,
    )
    model.config.use_cache = cfg["dpo_extras"].use_cache

    if getattr(processor, "tokenizer", None) is None:
        raise RuntimeError("processor.tokenizer is required for mDPO preprocessing")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    if run is not None:
        run.config.update(
            {
                "merged_base_path": merged_dir,
                "full_base_model": merge_meta.get("base_model_id"),
                "sft_adapter_checkpoint": merge_meta.get("sft_checkpoint"),
                "mdpo_params": vars(cfg["mdpo_loss"]),
            }
        )

    ds = load_dpo_train_eval_dataset(
        config_name=cfg["dataset"]["lang"],
        processor=processor,
        streaming=cfg["dataset"]["streaming"],
        cache_dir=cfg["logistics"].hf_cache_dir,
        format_data=True,
        allowed_modalities=None,  # unimodal rows preserve abstention behavior
    )
    train_dataset = ds.get("train", None)
    eval_dataset = ds.get("eval", None)
    if train_dataset is None or eval_dataset is None:
        raise RuntimeError("could not load the DPO train/eval dataset")
    train_dataset = _select_subset(train_dataset, cfg["dataset"]["max_train_samples"])
    eval_dataset = _select_subset(eval_dataset, cfg["dataset"]["max_eval_samples"])

    dpo_args = build_dpo_config(cfg, adapter_output_dir, run_name)
    # Left-truncation can drop image placeholders while keeping image features.
    dpo_args.truncation_mode = "keep_start"

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
        target_modules=spec.lora_target_modules or list(lora_cfg.target_modules),
        task_type="CAUSAL_LM",
    )

    strategy = VISION_STRATEGIES[model_key]()
    data_collator = MDPOPreferenceCollator(
        processor=processor,
        strategy=strategy,
        mdpo_params=cfg["mdpo_loss"],
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    trainer = MDPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        strategy=strategy,
        mdpo_params=cfg["mdpo_loss"],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        data_collator=data_collator,
        peft_config=peft_config,
    )
    trainer.add_callback(MDPOTrainMetricsCallback(run))
    eval_collator = spec.build_eval_collator(processor, cfg)
    trainer.add_callback(
        MDPOMetricsCallback(trainer.model, processor, eval_dataset, eval_collator, cfg)
    )

    training_meta = {
        "adapter_style": "fresh_lora_on_merged_sft",
        "merged_base_path": merged_dir,
        "full_base_model": merge_meta.get("base_model_id"),
        "sft_adapter_checkpoint": merge_meta.get("sft_checkpoint"),
        "mdpo_params": vars(cfg["mdpo_loss"]),
        "dataset_id": cfg["dataset"]["dataset_id"],
        "dataset_lang": cfg["dataset"]["lang"],
        "eval_split": "train_eval_split",
        "effective_batch_size": cfg["dpo"].batch_size
        * cfg["dpo"].gradient_accumulation_steps
        * _get_world_size(),
        "flash_attention_enabled": use_flash_attention,
        "config_source": "config/logistics.py",
        "timestamp": datetime.now().isoformat(),
    }
    # written before training so mid-run resumes pass the stale-checkpoint guard
    with open(os.path.join(adapter_output_dir, "training_meta.json"), "w") as f:
        json.dump(training_meta, f, indent=2)

    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.model.save_pretrained(adapter_output_dir, safe_serialization=True)
    processor.save_pretrained(adapter_output_dir)

    log_run_metadata(run, training_meta)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
