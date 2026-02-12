from __future__ import annotations

from typing import Any, Iterable

import torch
from transformers import AutoProcessor

from config.logistics import Logistics, ModelCards
from src.collators.ayacollator import AyaVisionSFTCollator
from src.collators.gemma3_collator import Gemma3VisionSFTCollator
from src.collators.llama32_collator import Llama32VisionSFTCollator
from src.collators.qwen3_collator import Qwen3VisionSFTCollator
from src.datasets.loader import load_hf_dataset


def _iter_dataset(dataset) -> Iterable[dict]:
    if hasattr(dataset, "select"):
        for item in dataset:
            yield item
        return
    for item in dataset:
        yield item


def _collect_samples(dataset, max_samples: int, image_key: str, target_key: str) -> list[dict]:
    samples: list[dict] = []
    for ex in _iter_dataset(dataset):
        image = ex.get(image_key)
        target = ex.get(target_key)
        if image is None:
            continue
        if not isinstance(target, str) or not target.strip():
            continue
        samples.append(ex)
        if len(samples) >= max_samples:
            break
    return samples


def _find_image_tensor_keys(outputs: dict) -> list[str]:
    keys: list[str] = []
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor) and ("pixel" in key or "image" in key):
            keys.append(key)
    return sorted(keys)


def _check_padding_after_supervised(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    pad_id: int | None,
    ignore_index: int,
) -> list[bool | None]:
    if pad_id is None:
        return [None for _ in range(labels.shape[0])]
    results: list[bool | None] = []
    for row_labels, row_ids in zip(labels, input_ids):
        supervised_positions = (row_labels != ignore_index).nonzero(as_tuple=True)[0]
        if supervised_positions.numel() == 0:
            results.append(None)
            continue
        last_idx = int(supervised_positions[-1].item())
        has_pad_after = (row_ids[last_idx + 1 :] == pad_id).any().item()
        results.append(bool(has_pad_after))
    return results


def _print_output_summary(
    outputs: dict,
    ignore_index: int,
    pad_id: int | None,
    processor: Any,
) -> None:
    print(f"output_keys={sorted(outputs.keys())}")
    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"]
    labels = outputs["labels"]
    print(f"input_ids_shape={tuple(input_ids.shape)}")
    print(f"attention_mask_shape={tuple(attention_mask.shape)}")
    print(f"labels_shape={tuple(labels.shape)}")
    supervised_tokens = int((labels != ignore_index).sum().item())
    per_example = (labels != ignore_index).sum(dim=1).tolist()
    print(f"supervised_tokens_total={supervised_tokens}")
    print(f"supervised_tokens_per_example={per_example}")
    padding_after = _check_padding_after_supervised(
        labels,
        input_ids,
        pad_id,
        ignore_index,
    )
    print(f"pad_after_supervised_per_example={padding_after}")
    tokenizer = getattr(processor, "tokenizer", processor)
    decoded_by_example: list[str] = []
    for row_labels, row_ids in zip(labels, input_ids):
        keep = row_labels != ignore_index
        kept_ids = row_ids[keep].detach().cpu().tolist()
        if pad_id is not None:
            kept_ids = [tid for tid in kept_ids if tid != pad_id]
        decoded_by_example.append(tokenizer.decode(kept_ids, skip_special_tokens=False))
    print("supervised_tokens_decoded_per_example=")
    for idx, text in enumerate(decoded_by_example):
        print(f"[{idx}] {text}")
    image_tensor_keys = _find_image_tensor_keys(outputs)
    print(f"image_tensor_keys={image_tensor_keys}")
    for key in image_tensor_keys:
        tensor = outputs[key]
        print(f"{key}_shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def _assert_required(outputs: dict, ignore_index: int) -> None:
    assert "input_ids" in outputs, "Missing input_ids"
    assert "attention_mask" in outputs, "Missing attention_mask"
    assert "labels" in outputs, "Missing labels"
    assert outputs["input_ids"].shape == outputs["labels"].shape, "labels/input_ids shape mismatch"
    assert _find_image_tensor_keys(outputs), "Missing image tensor data in outputs"
    assert torch.any(outputs["labels"] != ignore_index).item(), "No supervised tokens in labels"


def _load_dataset_with_fallback(logistics: Logistics):
    dataset_id = logistics.hf_sft_ds_id
    cache_dir = logistics.hf_cache_dir
    langs = list(getattr(logistics, "langs", [])) or ["en"]
    last_exc: Exception | None = None
    for lang in langs + [None]:
        try:
            dataset = load_hf_dataset(
                dataset_id,
                split="train",
                config_name=lang,
                cache_dir=cache_dir,
            )
            print(f"dataset_loaded config_name={lang}")
            return dataset, dataset_id, lang
        except Exception as exc:
            last_exc = exc
            print(f"dataset_load_failed config_name={lang} err={exc}")
    raise RuntimeError(f"Failed to load dataset_id={dataset_id}") from last_exc


def _run_smoke_test(
    model_id: str,
    collator_cls: Any,
    processor_kwargs: dict,
    samples: list[dict],
) -> None:
    print("\n=== COLLATOR SMOKE ===")
    print(f"model_id={model_id}")
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    collator = collator_cls(processor=processor, training=True, max_length=2048)
    pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    ignore_index = getattr(collator, "ignore_index", -100)
    outputs = collator(samples)
    _print_output_summary(outputs, ignore_index, pad_id, processor)
    _assert_required(outputs, ignore_index)


def main() -> None:
    logistics = Logistics()
    mc = ModelCards()
    dataset, dataset_id, lang = _load_dataset_with_fallback(logistics)
    print(f"dataset_id={dataset_id}")
    print(f"dataset_lang={lang}")

    samples = _collect_samples(
        dataset,
        max_samples=4,
        image_key="image",
        target_key="target_json",
    )
    print(f"samples_loaded={len(samples)}")
    if not samples:
        raise RuntimeError("No valid samples found (need image + non-empty target_json).")
    for idx, sample in enumerate(samples):
        sample_id = (
            sample.get("id")
            or sample.get("image_id")
            or sample.get("image_path")
            or sample.get("uid")
            or sample.get("uuid")
        )
        print(f"sample_idx={idx} sample_id={sample_id}")

    collator_map = [
        (mc.aya_model_name, AyaVisionSFTCollator, {"trust_remote_code": True}),
        (mc.llama3_2vl, Llama32VisionSFTCollator, {}),
        (mc.gemm3_12b, Gemma3VisionSFTCollator, {"trust_remote_code": True}),
        (mc.qwen3_vl_8b_instruct, Qwen3VisionSFTCollator, {"trust_remote_code": True}),
    ]

    for model_id, collator_cls, processor_kwargs in collator_map:
        _run_smoke_test(model_id, collator_cls, processor_kwargs, samples)


if __name__ == "__main__":
    main()
