from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import itertools

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoProcessor, TrainerCallback

from src.hf.publish_lora_models import get_sft_best_model
from src.inference.run_inference import BaseModels
from src.utils.sft_utils import _count_trainable_params, _dtype_from_str, _get_world_size


def _select_subset(dataset, max_samples: Optional[int]):
    if max_samples is None:
        return dataset
    if hasattr(dataset, "select"):
        return dataset.select(range(min(len(dataset), max_samples)))
    if hasattr(dataset, "take"):
        return list(dataset.take(max_samples))
    return list(itertools.islice(dataset, max_samples))



def build_dpo_config(cfg: Dict[str, Any], output_dir: str, run_name: str):
    from trl import DPOConfig

    dpo_cfg = cfg.get("dpo", {})
    extras = cfg.get("dpo_extras", {})
    wandb_cfg = cfg.get("wandb") or {}
    wandb_enabled = bool(wandb_cfg.get("enabled", True)) and os.environ.get("WANDB_DISABLED") != "true"

    return DPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=dpo_cfg.num_epochs,
        per_device_train_batch_size=dpo_cfg.batch_size,
        gradient_accumulation_steps=dpo_cfg.gradient_accumulation_steps,
        gradient_checkpointing=extras.gradient_checkpointing,
        learning_rate=dpo_cfg.learning_rate,
        weight_decay=dpo_cfg.weight_decay,
        warmup_ratio=dpo_cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=dpo_cfg.max_grad_norm,
        max_length=dpo_cfg.max_length,
        beta=dpo_cfg.beta,
        loss_type=dpo_cfg.loss_type,
        bf16=extras.bf16,
        fp16=dpo_cfg.fp16,
        tf32=extras.tf32,
        optim=extras.optim,
        logging_steps=dpo_cfg.logging_steps,
        save_steps=dpo_cfg.save_steps,
        eval_steps=dpo_cfg.eval_steps,
        eval_strategy=extras.eval_strategy,
        save_total_limit=dpo_cfg.save_total_limit,
        load_best_model_at_end=dpo_cfg.load_best_model_at_end,
        metric_for_best_model=dpo_cfg.metric_for_best_model,
        greater_is_better=dpo_cfg.greater_is_better,
        report_to=["wandb"] if wandb_enabled else [],
        disable_tqdm=extras.disable_tqdm,
        dataloader_num_workers=dpo_cfg.num_workers,
        dataloader_pin_memory=extras.dataloader_pin_memory,
        dataloader_persistent_workers=extras.dataloader_persistent_workers,
        dataloader_prefetch_factor=extras.dataloader_prefetch_factor,
        remove_unused_columns=dpo_cfg.remove_unused_columns,
    )




def _find_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    stack = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0:
                chunk = text[start : i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    continue
    return None


def _normalize_fact(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _is_consecutive_ids(visual_facts: Any) -> bool:
    if not isinstance(visual_facts, list):
        return False
    ids = []
    for fact in visual_facts:
        if not isinstance(fact, dict):
            return False
        if "id" not in fact:
            return False
        ids.append(fact["id"])
    if not ids:
        return True
    try:
        ids = [int(i) for i in ids]
    except Exception:
        return False
    return ids == list(range(0, len(ids)))


def _json_schema_checks(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    required_keys = {
        "label",
        "need_explanation",
        "visual_facts",
        "evidence_fact_ids",
        "incongruity",
        "explanation",
    }
    errors: List[str] = []
    if not required_keys.issubset(payload.keys()):
        errors.append("missing_keys")
    if "visual_facts" in payload and not _is_consecutive_ids(payload.get("visual_facts")):
        errors.append("non_consecutive_ids")
    return len(errors) == 0, errors


def _hallucination_flag(payload: Dict[str, Any], example: Dict[str, Any]) -> bool:
    model_facts = payload.get("visual_facts") if isinstance(payload, dict) else None
    if not isinstance(model_facts, list) or not model_facts:
        return False

    provided = example.get("visual_facts", [])
    if not isinstance(provided, list):
        return True

    normalized = [_normalize_fact(f) for f in provided if f]
    if not normalized:
        return True

    for fact in model_facts:
        if not isinstance(fact, dict):
            return True
        fact_text = _normalize_fact(fact.get("fact", ""))
        if not fact_text:
            return True
        if not any(fact_text in ref or ref in fact_text for ref in normalized):
            return True
    return False


def _pivot_consistent(payload: Dict[str, Any]) -> bool:
    label = str(payload.get("label", "")).strip().lower()
    incongruity = str(payload.get("incongruity", "")).strip()
    if label == "sarcastic":
        return bool(incongruity)
    if label == "non_sarcastic":
        return incongruity == ""
    return False


def _length_ratio(a: str, b: str) -> float:
    if not b:
        return 0.0
    return len(a) / max(1, len(b))


@torch.no_grad()
def evaluate_mdpo(
    model: Any,
    processor: Any,
    eval_dataset: Iterable[Dict[str, Any]],
    prompt_collator,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    dpo_cfg = cfg.get("dpo", {})
    max_samples = cfg.get("dataset", {}).get("max_eval_samples")
    batch_size = getattr(dpo_cfg, "eval_batch_size", None) or dpo_cfg.batch_size

    was_training = model.training
    model.eval()

    eval_examples = _select_subset(eval_dataset, max_samples)
    if not isinstance(eval_examples, list):
        eval_examples = list(eval_examples)

    total = 0
    valid_json = 0
    schema_errors: Dict[str, int] = {}
    hallucinated = 0
    pivot_ok = 0
    consecutive_ids_ok = 0
    length_ratios: List[float] = []
    chosen_rejected_ratios: List[float] = []

    def _process_example(text: str, example: Dict[str, Any]):
        nonlocal total, valid_json, hallucinated, pivot_ok, consecutive_ids_ok
        total += 1
        payload = _find_json_object(text)
        if not isinstance(payload, dict):
            schema_errors["invalid_json"] = schema_errors.get("invalid_json", 0) + 1
            return
        valid_json += 1

        ok_schema, errs = _json_schema_checks(payload)
        if not ok_schema:
            for err in errs:
                schema_errors[err] = schema_errors.get(err, 0) + 1

        if _is_consecutive_ids(payload.get("visual_facts")):
            consecutive_ids_ok += 1

        if not _hallucination_flag(payload, example):
            hallucinated += 0
        else:
            hallucinated += 1

        if _pivot_consistent(payload):
            pivot_ok += 1

        chosen = str(example.get("chosen", "") or "")
        if chosen:
            length_ratios.append(_length_ratio(text, chosen))
        rejected = str(example.get("rejected", "") or "")
        if chosen and rejected:
            chosen_rejected_ratios.append(_length_ratio(chosen, rejected))

    batch_examples: List[Dict[str, Any]] = []
    for example in eval_examples:
        batch_examples.append(example)
        if len(batch_examples) < batch_size:
            continue
        model_inputs = prompt_collator(batch_examples)
        model_inputs = {
            k: v.to(model.device) if torch.is_tensor(v) else v
            for k, v in model_inputs.items()
        }
        generated = model.generate(
            **model_inputs,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=cfg.get("eval_decode", {}).get("max_new_tokens", 256),
        )
        texts = processor.batch_decode(generated, skip_special_tokens=True)
        for text, ex in zip(texts, batch_examples):
            _process_example(text, ex)
        batch_examples = []

    if batch_examples:
        model_inputs = prompt_collator(batch_examples)
        model_inputs = {
            k: v.to(model.device) if torch.is_tensor(v) else v
            for k, v in model_inputs.items()
        }
        generated = model.generate(
            **model_inputs,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=cfg.get("eval_decode", {}).get("max_new_tokens", 256),
        )
        texts = processor.batch_decode(generated, skip_special_tokens=True)
        for text, ex in zip(texts, batch_examples):
            _process_example(text, ex)

    metrics: Dict[str, Any] = {}
    metrics["eval/num_samples"] = total
    metrics["eval/json_valid_count"] = valid_json
    metrics["eval/json_validity"] = valid_json / max(1, total)
    metrics["eval/hallucination_rate"] = hallucinated / max(1, total)
    metrics["eval/pivot_consistency"] = pivot_ok / max(1, total)
    metrics["eval/visual_facts_id_consecutive_rate"] = consecutive_ids_ok / max(1, total)
    if length_ratios:
        metrics["eval/length_ratio_output_target"] = sum(length_ratios) / len(length_ratios)
    if chosen_rejected_ratios:
        metrics["eval/length_ratio_chosen_rejected"] = (
            sum(chosen_rejected_ratios) / len(chosen_rejected_ratios)
        )
    for key, value in schema_errors.items():
        metrics[f"eval/schema_error_types/{key}"] = value / max(1, total)

    if was_training:
        model.train()
    return metrics


def load_sft_adapter_models(
    model_key: str,
    cfg: Dict[str, Any],
    bnb_config: Any,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    use_fast: bool = True,
) -> Tuple[Any, Any, str, str]:
    
    models_dirs = {
        "aya": "aya_vision_8b",
        "llama": "llama32",
        "gemma": "gemma3",
        "qwen": "qwen3_vl_8b",
    }

    result = get_sft_best_model(models_dirs.get(model_key))
    checkpoint_path = result.get("best_model_path")
    if not checkpoint_path:
        raise FileNotFoundError(f"no SFT checkpoint found for {model_key}")

    base_models = BaseModels()
    base_model_id = base_models.models.get(model_key)
    if not base_model_id:
        raise ValueError(f"unknown model_key: {model_key}")
    
    print(f"[DPO] Loading processor from base model: {base_model_id}")
    processor = AutoProcessor.from_pretrained(
        base_model_id,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast,
        )
        if getattr(processor, "tokenizer", None) is not None:
            processor.tokenizer = tokenizer
            print(f"[DPO] Attached tokenizer from: {checkpoint_path}")
        else:
            print("[DPO] Processor has no tokenizer attribute; using base tokenizer.")
    except Exception as exc:
        print(f"[DPO] Failed to load tokenizer from checkpoint; using base processor: {exc}")
    
    
    model_kwargs = dict(
        dtype=_dtype_from_str(cfg["model"]["torch_dtype"]),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        quantization_config=bnb_config,
    )

    model_class = base_models.get_class(model_key)
    print(f"[DPO] Loading base model: {base_model_id}")
    
    if cfg["model"]["use_flash_attention"] and cfg["dpo_extras"].use_flash_attention_2:
        try:
            policy_model = model_class.from_pretrained(
                base_model_id, attn_implementation="flash_attention_2", **model_kwargs
            )
        except Exception as exc:
            print(f"FlashAttention-2 unavailable, loading default attention: {exc}")
            policy_model = model_class.from_pretrained(base_model_id, **model_kwargs)
    else:
        policy_model = model_class.from_pretrained(base_model_id, **model_kwargs)
    

    # Enable gradient checkpointing from DPOConfigExtras
    if cfg["dpo_extras"].gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()

    policy_model = prepare_model_for_kbit_training(policy_model)

    print(f"[DPO] Attaching SFT adapter from: {checkpoint_path}")
    policy_model = PeftModel.from_pretrained(
        policy_model, 
        checkpoint_path, 
        is_trainable=True
    )

    print("\n" + "="*40)
    print("TRAINABLE PARAMETERS STATUS:")
    policy_model.print_trainable_parameters()
    print("="*40 + "\n")

    return policy_model, processor, checkpoint_path, base_model_id






__all__ = [
    "_count_trainable_params",
    "_dtype_from_str",
    "_get_world_size",
    "_select_subset",
    "build_dpo_config",
    "evaluate_mdpo",
    "load_sft_adapter_models",
    "MDPOMetricsCallback",
    "MDPOTrainMetricsCallback",
    "MDPOStepEvalCallback",
]


class MDPOMetricsCallback(TrainerCallback):
    def __init__(
        self,
        model: Any,
        processor: Any,
        eval_dataset: Any,
        prompt_collator: Any,
        cfg: Dict[str, Any],
    ) -> None:
        self.model = model
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.prompt_collator = prompt_collator
        self.cfg = cfg

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        extra = evaluate_mdpo(
            self.model,
            self.processor,
            self.eval_dataset,
            self.prompt_collator,
            self.cfg,
        )
        if metrics is not None:
            metrics.update(extra)
        return control


class MDPOTrainMetricsCallback(TrainerCallback):
    def __init__(self, run: Any) -> None:
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.run is None or not logs:
            return control
        keys = ("reward", "logps", "margin", "loss", "kl", "entropy", "grad_norm", "lr")
        filtered = {
            k: v for k, v in logs.items() if any(t in k.lower() for t in keys)
        }
        if filtered:
            self.run.log(filtered)
        return control


class MDPOStepEvalCallback(TrainerCallback):
    def __init__(
        self,
        model: Any,
        processor: Any,
        eval_dataset: Any,
        prompt_collator: Any,
        cfg: Dict[str, Any],
        step_interval: int,
        run: Any,
    ) -> None:
        self.model = model
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.prompt_collator = prompt_collator
        self.cfg = cfg
        self.step_interval = max(1, int(step_interval))
        self.run = run

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0:
            return control
        if state.global_step % self.step_interval != 0:
            return control
        metrics = evaluate_mdpo(
            self.model,
            self.processor,
            self.eval_dataset,
            self.prompt_collator,
            self.cfg,
        )
        if self.run is not None:
            self.run.log(metrics)
        return control
