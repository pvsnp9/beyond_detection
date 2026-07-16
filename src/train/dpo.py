from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from datasets import disable_caching
from peft import LoraConfig
from transformers import BitsAndBytesConfig, set_seed

from config.logistics import build_std_dpo_cfg
from src.datasets.loader import load_dpo_train_eval_dataset
from src.train.mdpo import MDPO_MODELS, _stale_checkpoint_guard
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


class RandomRejectedCollator(MDPOPreferenceCollator):
    # dpo_random: swap each row's rejected tokens with another same-modality
    # example's, resampled fresh every batch

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pools: Optional[Dict[str, List[tuple]]] = None

    def build_pools(self, train_dataset: Any) -> None:
        # column access only: row iteration would decode every PIL image
        pools = defaultdict(list)
        for row_id, modality, tokens in zip(
            train_dataset["id"], train_dataset["modality"], train_dataset["rejected_input_ids"]
        ):
            pools[str(modality).strip().lower()].append((row_id, tokens))
        for modality, pool in pools.items():
            if len(pool) < 2:
                raise RuntimeError(f"modality '{modality}' needs >=2 rows for random negatives")
        self.pools = dict(pools)

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.pools is None:
            raise RuntimeError("call build_pools(trainer.train_dataset) before training")
        rng = random.Random()  # urandom-seeded per call: fresh, worker-safe draws
        swapped = []
        for example in examples:
            pool = self.pools[str(example.get("modality", "both")).strip().lower()]
            while True:
                row_id, tokens = pool[rng.randrange(len(pool))]
                if row_id != example["id"]:
                    break
            example = dict(example)  # never mutate the dataset row
            example["rejected_input_ids"] = tokens
            swapped.append(example)
        return super().torch_call(swapped)


class RandomRejectedTrainer(MDPOTrainer):
    # random collator for the train dataloader only; eval keeps true pairs so
    # eval_rewards/margins-based model selection stays comparable across variants

    def __init__(self, *args: Any, train_collator: RandomRejectedCollator, **kwargs: Any) -> None:
        self._train_collator = train_collator
        super().__init__(*args, **kwargs)
        if self.precompute_ref_log_probs:
            # cached ref logps would score the original rejected, not the swapped one
            raise ValueError("dpo_random requires precompute_ref_log_probs=False")

    def get_train_dataloader(self) -> Any:
        original = self.data_collator
        self.data_collator = self._train_collator
        try:
            return super().get_train_dataloader()
        finally:
            self.data_collator = original


def main() -> None:
    args = [arg.strip().lower() for arg in sys.argv[1:]]
    model_key = args[0] if args else ""
    if model_key not in MDPO_MODELS or args[1:] not in ([], ["random"]):
        raise SystemExit(f"usage: python -m src.train.dpo <{'|'.join(MDPO_MODELS)}> [random]")
    variant = "dpo_random" if len(args) > 1 else "dpo"
    spec = MDPO_MODELS[model_key]

    cfg = build_std_dpo_cfg(spec.model_card, variant)
    # Avoid reusing stale tokenization/map artifacts across runs.
    disable_caching()
    set_seed(cfg["dpo"].seed)

    max_train_override = os.environ.get("DPO_MAX_TRAIN_SAMPLES")
    if max_train_override:
        cfg["dataset"]["max_train_samples"] = int(max_train_override)
    epochs_override = os.environ.get("DPO_NUM_EPOCHS")
    if epochs_override:
        cfg["dpo"].num_epochs = int(epochs_override)
    output_dir_override = os.environ.get("DPO_OUTPUT_DIR")
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

    run_name = datetime.now().strftime(f"{variant}-{model_key}-%Y%m%d-%H%M%S")
    run = init_wandb(cfg["wandb"], run_name)

    adapter_output_dir = cfg["output"]["adapter_output_dir"] or os.path.join(
        cfg["dpo"].model_dir, variant, spec.adapter_dirname
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
        raise RuntimeError("processor.tokenizer is required for DPO preprocessing")
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    if run is not None:
        run.config.update(
            {
                "variant": variant,
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

    trainer_kwargs = dict(
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
    if variant == "dpo":
        trainer = MDPOTrainer(**trainer_kwargs)
    else:
        random_collator = RandomRejectedCollator(
            processor=processor,
            strategy=strategy,
            mdpo_params=cfg["mdpo_loss"],
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        trainer = RandomRejectedTrainer(train_collator=random_collator, **trainer_kwargs)
        # after __init__: pools need the process_row-mapped train dataset
        random_collator.build_pools(trainer.train_dataset)
    trainer.add_callback(MDPOTrainMetricsCallback(run))
    eval_collator = spec.build_eval_collator(processor, cfg)
    trainer.add_callback(
        MDPOMetricsCallback(trainer.model, processor, eval_dataset, eval_collator, cfg)
    )

    training_meta = {
        "variant": variant,
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
