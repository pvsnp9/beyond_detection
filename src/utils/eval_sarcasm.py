from __future__ import annotations

import json
import re
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Optional

import itertools

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

LABEL_ORDER = ("not_sarcastic", "sarcastic", "unknown")


def _normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "sarcastic" if value else "not_sarcastic"
    if isinstance(value, (int, float)):
        v = int(value)
        if v == 0:
            return "not_sarcastic"
        if v == 1:
            return "sarcastic"
        if v == 2:
            return "unknown"
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "sarcastic"}:
        return "sarcastic"
    if text in {"2", "unknown", "unk", "uncertain", "cannot_determine", "cannot_determine_from_missing_modality"}:
        return "unknown"
    if text in {"0", "false", "no", "n", "not_sarcastic", "nonsarcastic"}:
        return "not_sarcastic"
    if "not sarcastic" in text or "non-sarcastic" in text:
        return "not_sarcastic"
    if "unknown" in text or "cannot determine" in text:
        return "unknown"
    if "sarcastic" in text:
        return "sarcastic"
    return None


def _cfg_get(container: Any, key: str, default: Any = None) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _extract_gold(example: Dict[str, Any]) -> Optional[str]:
    for key in ("label", "labels", "is_sarcastic", "sarcasm", "gold", "target"):
        if key in example:
            label = _normalize_label(example.get(key))
            if label is not None:
                return label
    target_json = example.get("target_json")
    if isinstance(target_json, str):
        try:
            payload = json.loads(target_json)
        except json.JSONDecodeError:
            payload = None
    elif isinstance(target_json, dict):
        payload = target_json
    else:
        payload = None
    if isinstance(payload, dict):
        for key in ("label", "labels", "sarcastic", "is_sarcastic", "sarcasm"):
            if key in payload:
                return _normalize_label(payload.get(key))
    return None


def _parse_prediction(text: str) -> Optional[str]:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    def _label_from_payload(payload: Any) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        for key in ("label", "labels", "sarcastic", "is_sarcastic", "sarcasm"):
            if key in payload:
                normalized = _normalize_label(payload.get(key))
                if normalized is not None:
                    return normalized
        return None

    # 1) Direct JSON parse.
    try:
        payload = json.loads(raw)
        label = _label_from_payload(payload)
        if label is not None:
            return label
    except Exception:
        pass

    # 2) JSON block inside extra text (e.g., model preamble + JSON).
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            label = _label_from_payload(payload)
            if label is not None:
                return label
        except Exception:
            pass
    return _normalize_label(raw)


def _build_metrics(preds: Iterable[str], golds: Iterable[str]) -> Dict[str, Any]:
    pred_list = list(preds)
    gold_list = list(golds)
    metrics: Dict[str, Any] = {
        "num_eval_samples": len(gold_list),
        "pred_dist": {label: pred_list.count(label) for label in LABEL_ORDER},
        "gold_dist": {label: gold_list.count(label) for label in LABEL_ORDER},
        "num_failed_preds": pred_list.count("failed"),
    }
    if not gold_list:
        return metrics

    metrics["accuracy"] = accuracy_score(gold_list, pred_list)
    metrics["macro_f1"] = f1_score(
        gold_list,
        pred_list,
        labels=list(LABEL_ORDER),
        average="macro",
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        gold_list,
        pred_list,
        labels=list(LABEL_ORDER),
        average=None,
        zero_division=0,
    )
    for idx, label in enumerate(LABEL_ORDER):
        metrics[f"precision_{label}"] = float(precision[idx])
        metrics[f"recall_{label}"] = float(recall[idx])
        metrics[f"f1_{label}"] = float(f1[idx])
        metrics[f"support_{label}"] = int(support[idx])

    return metrics


def _select_gemma_dtype(cfg: Dict[str, Any], model: Any) -> torch.dtype:
    sft_cfg = cfg.get("sft", {})
    extras_cfg = cfg.get("sft_extras", {})
    if _cfg_get(extras_cfg, "bf16", False):
        return torch.bfloat16
    if _cfg_get(sft_cfg, "fp16", False):
        return torch.float16
    model_dtype = getattr(model, "dtype", None)
    if model_dtype in (torch.float16, torch.bfloat16):
        return model_dtype
    return torch.float16


def _cast_gemma_pixel_values(
    model: Any,
    model_inputs: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    pixel_values = model_inputs.get("pixel_values")
    if not torch.is_tensor(pixel_values):
        return
    target_dtype = _select_gemma_dtype(cfg, model)
    model_inputs["pixel_values"] = pixel_values.to(dtype=target_dtype)


def _gemma_autocast_context(model: Any, cfg: Dict[str, Any]):
    model_device = getattr(model, "device", None)
    if not isinstance(model_device, torch.device):
        return nullcontext()
    if model_device.type != "cuda":
        return nullcontext()
    autocast_dtype = _select_gemma_dtype(cfg, model)
    return torch.autocast(device_type="cuda", dtype=autocast_dtype)


@torch.no_grad()
def evaluate_llama(
    model: Any,
    processor: Any,
    eval_dataset: Iterable[Dict[str, Any]],
    collator: Any,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    decode_cfg = cfg.get("eval_decode", {})
    dataset_cfg = cfg.get("dataset", {})
    sft_cfg = cfg.get("sft", {})
    max_samples = _cfg_get(dataset_cfg, "max_eval_samples")
    batch_size = _cfg_get(sft_cfg, "batch_size", 1)

    was_training = model.training
    prev_use_cache = model.config.use_cache
    prev_padding_side = processor.tokenizer.padding_side

    model.eval()
    model.config.use_cache = True  
    processor.tokenizer.padding_side = "left" # Required for decoder-only generate

    try:
        preds = []
        golds = []

        if max_samples:
            if hasattr(eval_dataset, "select"):
                eval_dataset = eval_dataset.select(range(min(len(eval_dataset), max_samples)))
            elif hasattr(eval_dataset, "take"):
                eval_dataset = list(eval_dataset.take(max_samples))
            else:
                eval_dataset = list(itertools.islice(eval_dataset, max_samples))

        if hasattr(eval_dataset, "__len__"):
            loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collator)
            for batch in tqdm(loader, desc="Eval", leave=False):
                batch = {
                    k: v.to(model.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                generated = model.generate(
                    **batch,
                    do_sample=decode_cfg.get("do_sample", False),
                    temperature=decode_cfg.get("temperature", 0.0),
                    top_p=decode_cfg.get("top_p", 1.0),
                    max_new_tokens=decode_cfg.get("max_new_tokens", 8),
                )
                generated_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch["input_ids"], generated)
                ]
                texts = processor.batch_decode(generated_trimmed, skip_special_tokens=True)
                for text in texts:
                    preds.append(_parse_prediction(text) or "failed")
            for example in eval_dataset:
                golds.append(_extract_gold(example) or "unknown")
        else:
            batch_examples = []
            seen = 0
            for example in eval_dataset:
                batch_examples.append(example)
                seen += 1
                if max_samples and seen >= max_samples:
                    break
                if len(batch_examples) < batch_size:
                    continue
                model_inputs = collator(batch_examples)
                model_inputs = {
                    k: v.to(model.device) if torch.is_tensor(v) else v
                    for k, v in model_inputs.items()
                }
                generated = model.generate(
                    **model_inputs,
                    do_sample=decode_cfg.get("do_sample", False),
                    temperature=decode_cfg.get("temperature", 0.0),
                    top_p=decode_cfg.get("top_p", 1.0),
                    max_new_tokens=decode_cfg.get("max_new_tokens", 8),
                )
                generated_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(model_inputs["input_ids"], generated)
                ]
                texts = processor.batch_decode(generated_trimmed, skip_special_tokens=True)
                for text, ex in zip(texts, batch_examples):
                    preds.append(_parse_prediction(text) or "failed")
                    golds.append(_extract_gold(ex) or "unknown")
                batch_examples = []

            if batch_examples:
                model_inputs = collator(batch_examples)
                model_inputs = {
                    k: v.to(model.device) if torch.is_tensor(v) else v
                    for k, v in model_inputs.items()
                }
                generated = model.generate(
                    **model_inputs,
                    do_sample=decode_cfg.get("do_sample", False),
                    temperature=decode_cfg.get("temperature", 0.0),
                    top_p=decode_cfg.get("top_p", 1.0),
                    max_new_tokens=decode_cfg.get("max_new_tokens", 8),
                )
                generated_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(model_inputs["input_ids"], generated)
                ]
                texts = processor.batch_decode(generated_trimmed, skip_special_tokens=True)
                for text, ex in zip(texts, batch_examples):
                    preds.append(_parse_prediction(text) or "failed")
                    golds.append(_extract_gold(ex) or "unknown")

        return _build_metrics(preds, golds)
    finally:
        model.config.use_cache = prev_use_cache
        processor.tokenizer.padding_side = prev_padding_side
        if was_training:
            model.train()


@torch.no_grad()
def evaluate_gemma(
    model: Any,
    processor: Any,
    eval_dataset: Iterable[Dict[str, Any]],
    collator: Any,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    decode_cfg = cfg.get("eval_decode", {})
    dataset_cfg = cfg.get("dataset", {})
    sft_cfg = cfg.get("sft", {})
    max_samples = _cfg_get(dataset_cfg, "max_eval_samples")
    batch_size = _cfg_get(sft_cfg, "batch_size", 1)
    
    was_training = model.training
    prev_use_cache = model.config.use_cache
    prev_padding_side = processor.tokenizer.padding_side
    
    processor.tokenizer.padding_side = "left"
    
    model.eval()
    model.config.use_cache = True

    try:
        preds = []
        golds = []

        if max_samples:
            if hasattr(eval_dataset, "select"):
                eval_dataset = eval_dataset.select(range(min(len(eval_dataset), max_samples)))
            elif hasattr(eval_dataset, "take"):
                eval_dataset = list(eval_dataset.take(max_samples))
            else:
                eval_dataset = list(itertools.islice(eval_dataset, max_samples))

        if hasattr(eval_dataset, "__len__"):
            loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collator)
            for batch in tqdm(loader, desc="Eval", leave=False):
                batch = {
                    k: v.to(model.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }
                _cast_gemma_pixel_values(model, batch, cfg)
                with _gemma_autocast_context(model, cfg):
                    generated = model.generate(
                        **batch,
                        do_sample=decode_cfg.get("do_sample", False),
                        temperature=decode_cfg.get("temperature", 0.0),
                        top_p=decode_cfg.get("top_p", 1.0),
                        max_new_tokens=decode_cfg.get("max_new_tokens", 8),
                    )
                generated_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch["input_ids"], generated)
                ]
                texts = processor.batch_decode(generated_trimmed, skip_special_tokens=True)
                for text in texts:
                    preds.append(_parse_prediction(text) or "failed")
            for example in eval_dataset:
                golds.append(_extract_gold(example) or "unknown")
        else:
            batch_examples = []
            seen = 0
            for example in eval_dataset:
                batch_examples.append(example)
                seen += 1
                if max_samples and seen >= max_samples:
                    break
                if len(batch_examples) < batch_size:
                    continue
                model_inputs = collator(batch_examples)
                model_inputs = {
                    k: v.to(model.device) if torch.is_tensor(v) else v
                    for k, v in model_inputs.items()
                }
                _cast_gemma_pixel_values(model, model_inputs, cfg)
                with _gemma_autocast_context(model, cfg):
                    generated = model.generate(
                        **model_inputs,
                        do_sample=decode_cfg.get("do_sample", False),
                        temperature=decode_cfg.get("temperature", 0.0),
                        top_p=decode_cfg.get("top_p", 1.0),
                        max_new_tokens=decode_cfg.get("max_new_tokens", 8),
                    )
                generated_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(model_inputs["input_ids"], generated)
                ]
                texts = processor.batch_decode(generated_trimmed, skip_special_tokens=True)
                for text, ex in zip(texts, batch_examples):
                    preds.append(_parse_prediction(text) or "failed")
                    golds.append(_extract_gold(ex) or "unknown")
                batch_examples = []

            if batch_examples:
                model_inputs = collator(batch_examples)
                model_inputs = {
                    k: v.to(model.device) if torch.is_tensor(v) else v
                    for k, v in model_inputs.items()
                }
                _cast_gemma_pixel_values(model, model_inputs, cfg)
                with _gemma_autocast_context(model, cfg):
                    generated = model.generate(
                        **model_inputs,
                        do_sample=decode_cfg.get("do_sample", False),
                        temperature=decode_cfg.get("temperature", 0.0),
                        top_p=decode_cfg.get("top_p", 1.0),
                        max_new_tokens=decode_cfg.get("max_new_tokens", 8),
                    )
                generated_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(model_inputs["input_ids"], generated)
                ]
                texts = processor.batch_decode(generated_trimmed, skip_special_tokens=True)
                for text, ex in zip(texts, batch_examples):
                    preds.append(_parse_prediction(text) or "failed")
                    golds.append(_extract_gold(ex) or "unknown")

        return _build_metrics(preds, golds)
    finally:
        model.config.use_cache = prev_use_cache
        processor.tokenizer.padding_side = prev_padding_side
        
        if was_training:
            model.train()


@torch.no_grad()
def evaluate_qwen3(
    model: Any,
    processor: Any,
    eval_dataset: Iterable[Dict[str, Any]],
    collator: Any,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    
    
    was_training = model.training
    prev_use_cache = model.config.use_cache
    prev_padding_side = processor.tokenizer.padding_side
    
    processor.tokenizer.padding_side = "left"
    
    model.eval()
    model.config.use_cache = True

    try:
        try:
            eval_cfg = cfg.get("eval", {})
            decode_cfg = cfg.get("eval_decode", {})
            dataset_cfg = cfg.get("dataset", {})
            sft_cfg = cfg.get("sft", {})
            max_samples = _cfg_get(dataset_cfg, "max_eval_samples")
            batch_size = _cfg_get(sft_cfg, "batch_size", 1)
            max_new_tokens = _cfg_get(
                eval_cfg, "max_new_tokens", _cfg_get(decode_cfg, "max_new_tokens", 256)
            )
            do_sample = _cfg_get(eval_cfg, "do_sample", _cfg_get(decode_cfg, "do_sample", False))
            temperature = _cfg_get(
                eval_cfg, "temperature", _cfg_get(decode_cfg, "temperature", 0.0)
            )
            top_p = _cfg_get(eval_cfg, "top_p", _cfg_get(decode_cfg, "top_p", 1.0))

            preds = []
            golds = []

            if max_samples:
                if hasattr(eval_dataset, "select"):
                    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), max_samples)))
                elif hasattr(eval_dataset, "take"):
                    eval_dataset = list(eval_dataset.take(max_samples))
                else:
                    eval_dataset = list(itertools.islice(eval_dataset, max_samples))

            if hasattr(eval_dataset, "__len__"):
                loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collator)
                for batch in tqdm(loader, desc="Eval", leave=False):
                    batch = {
                        k: v.to(model.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }
                    batch.pop("token_type_ids", None)
                    generated = model.generate(
                        **batch,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                    )
                    generated_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch["input_ids"], generated)
                    ]
                    texts = processor.batch_decode(
                        generated_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for text in texts:
                        preds.append(_parse_prediction(text) or "failed")
                for example in eval_dataset:
                    golds.append(_extract_gold(example) or "unknown")
            else:
                batch_examples = []
                seen = 0
                for example in eval_dataset:
                    batch_examples.append(example)
                    seen += 1
                    if max_samples and seen >= max_samples:
                        break
                    if len(batch_examples) < batch_size:
                        continue
                    model_inputs = collator(batch_examples)
                    model_inputs = {
                        k: v.to(model.device) if torch.is_tensor(v) else v
                        for k, v in model_inputs.items()
                    }
                    model_inputs.pop("token_type_ids", None)
                    generated = model.generate(
                        **model_inputs,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                    )
                    generated_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(model_inputs["input_ids"], generated)
                    ]
                    texts = processor.batch_decode(
                        generated_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for text, ex in zip(texts, batch_examples):
                        preds.append(_parse_prediction(text) or "failed")
                        golds.append(_extract_gold(ex) or "unknown")
                    batch_examples = []

                if batch_examples:
                    model_inputs = collator(batch_examples)
                    model_inputs = {
                        k: v.to(model.device) if torch.is_tensor(v) else v
                        for k, v in model_inputs.items()
                    }
                    model_inputs.pop("token_type_ids", None)
                    generated = model.generate(
                        **model_inputs,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                    )
                    generated_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(model_inputs["input_ids"], generated)
                    ]
                    texts = processor.batch_decode(
                        generated_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for text, ex in zip(texts, batch_examples):
                        preds.append(_parse_prediction(text) or "failed")
                        golds.append(_extract_gold(ex) or "unknown")

            return _build_metrics(preds, golds)
        except Exception as exc:
            raise RuntimeError(f"evaluate_qwen3 failed: {exc}") from exc
    finally:
        model.config.use_cache = prev_use_cache
        processor.tokenizer.padding_side = prev_padding_side

        if was_training:
            model.train()


@torch.no_grad()
def evaluate_aya(
    model: Any,
    processor: Any,
    eval_dataset: Iterable[Dict[str, Any]],
    collator: Any,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    was_training = model.training

    rev_use_cache = model.config.use_cache
    prev_padding_side = processor.tokenizer.padding_side

    model.eval()
    model.config.use_cache = True
    processor.tokenizer.padding_side = "left"
    try:
        try:
            eval_cfg = cfg.get("eval", {})
            decode_cfg = cfg.get("eval_decode", {})
            dataset_cfg = cfg.get("dataset", {})
            sft_cfg = cfg.get("sft", {})
            max_samples = _cfg_get(dataset_cfg, "max_eval_samples")
            batch_size = _cfg_get(sft_cfg, "batch_size", 1)
            max_new_tokens = _cfg_get(
                eval_cfg, "max_new_tokens", _cfg_get(decode_cfg, "max_new_tokens", 256)
            )
            do_sample = _cfg_get(eval_cfg, "do_sample", _cfg_get(decode_cfg, "do_sample", False))
            temperature = _cfg_get(
                eval_cfg, "temperature", _cfg_get(decode_cfg, "temperature", 0.2)
            )
            top_p = _cfg_get(eval_cfg, "top_p", _cfg_get(decode_cfg, "top_p", 1.0))

            preds = []
            golds = []

            if max_samples:
                if hasattr(eval_dataset, "select"):
                    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), max_samples)))
                elif hasattr(eval_dataset, "take"):
                    eval_dataset = list(eval_dataset.take(max_samples))
                else:
                    eval_dataset = list(itertools.islice(eval_dataset, max_samples))

            if hasattr(eval_dataset, "__len__"):
                loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collator)
                for batch in tqdm(loader, desc="Eval", leave=False):
                    batch = {
                        k: v.to(model.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }
                    image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
                    if image_token_id is None and getattr(processor, "tokenizer", None) is not None:
                        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
                    if image_token_id is not None and image_token_id != processor.tokenizer.unk_token_id:
                        image_token_counts = (batch["input_ids"] == image_token_id).sum(dim=1)
                        print(
                            "eval_image_tokens_per_row="
                            f"{image_token_counts.tolist()} image_token_id={image_token_id}"
                        )
                    else:
                        print(f"eval_image_token_id_unresolved={image_token_id}")
                    if getattr(processor, "tokenizer", None) is not None:
                        tokens = processor.tokenizer.convert_ids_to_tokens(
                            batch["input_ids"][0].tolist()
                        )
                        image_like = [
                            (i, tok)
                            for i, tok in enumerate(tokens)
                            if "image" in tok or tok in {"<", ">", "image"}
                        ]
                        print(f"eval_image_like_tokens_sample={image_like[:12]}")
                    if "pixel_values" in batch:
                        print(f"eval_pixel_values_shape={tuple(batch['pixel_values'].shape)}")
                    print(f"eval_input_ids_shape={tuple(batch['input_ids'].shape)}")
                    _cast_gemma_pixel_values(model, batch, cfg)
                    autocast_dtype = _select_gemma_dtype(cfg, model)
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        generated = model.generate(
                            **batch,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=max_new_tokens,
                        )
                    prompt_len = batch["input_ids"].shape[1]
                    new_tokens = [row[prompt_len:] for row in generated]
                    texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
                    for text in texts:
                        preds.append(_parse_prediction(text) or "failed")
                for example in eval_dataset:
                    golds.append(_extract_gold(example) or "unknown")
            else:
                batch_examples = []
                seen = 0
                for example in eval_dataset:
                    batch_examples.append(example)
                    seen += 1
                    if max_samples and seen >= max_samples:
                        break
                    if len(batch_examples) < batch_size:
                        continue
                    model_inputs = collator(batch_examples)
                    model_inputs = {
                        k: v.to(model.device) if torch.is_tensor(v) else v
                        for k, v in model_inputs.items()
                    }
                    _cast_gemma_pixel_values(model, model_inputs, cfg)
                    autocast_dtype = _select_gemma_dtype(cfg, model)
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        generated = model.generate(
                            **model_inputs,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=max_new_tokens,
                        )
                    prompt_len = model_inputs["input_ids"].shape[1]
                    new_tokens = [row[prompt_len:] for row in generated]
                    texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
                    for text, ex in zip(texts, batch_examples):
                        preds.append(_parse_prediction(text) or "failed")
                        golds.append(_extract_gold(ex) or "unknown")
                    batch_examples = []

                if batch_examples:
                    model_inputs = collator(batch_examples)
                    model_inputs = {
                        k: v.to(model.device) if torch.is_tensor(v) else v
                        for k, v in model_inputs.items()
                    }
                    _cast_gemma_pixel_values(model, model_inputs, cfg)
                    autocast_dtype = _select_gemma_dtype(cfg, model)
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        generated = model.generate(
                            **model_inputs,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=max_new_tokens,
                        )
                    prompt_len = model_inputs["input_ids"].shape[1]
                    new_tokens = [row[prompt_len:] for row in generated]
                    texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
                    for text, ex in zip(texts, batch_examples):
                        preds.append(_parse_prediction(text) or "failed")
                        golds.append(_extract_gold(ex) or "unknown")

            return _build_metrics(preds, golds)
        except Exception as exc:
            raise RuntimeError(f"evaluate_aya failed: {exc}") from exc
    finally:
        model.config.use_cache = prev_use_cache
        processor.tokenizer.padding_side = prev_padding_side
        if was_training:
            model.train()
