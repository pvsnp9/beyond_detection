from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import argparse
import traceback

import torch
from transformers import AutoProcessor

from config.logistics import Logistics, ModelCards
from src.collators.ayacollator import AyaVisionSFTCollator
from src.collators.gemma3_collator import Gemma3VisionSFTCollator
from src.collators.llama32_collator import Llama32VisionSFTCollator
from src.collators.qwen3_collator import Qwen3VisionSFTCollator
from src.datasets.loader import load_hf_dataset


def _iter_dataset(dataset) -> Iterable[dict]:
    for item in dataset:
        yield item


def _sample_id(sample: dict) -> Any:
    return (
        sample.get("id")
        or sample.get("image_id")
        or sample.get("image_path")
        or sample.get("uid")
        or sample.get("uuid")
    )


def _bad_image_reason(image: Any) -> Optional[str]:
    if image is None:
        return "none"
    if hasattr(image, "size") and hasattr(image, "mode"):
        return None
    shape = getattr(image, "shape", None)
    if shape is None:
        return "missing_shape"
    if len(shape) == 2:
        return "grayscale_2d"
    if len(shape) == 3 and shape[-1] not in (1, 3, 4):
        return "bad_channel_count"
    if len(shape) not in (2, 3):
        return "unsupported_ndim"
    return None


def _get_model_max_length(processor) -> Optional[int]:
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        return None
    max_len = getattr(tok, "model_max_length", None)
    if max_len is None:
        return None
    if isinstance(max_len, int) and max_len > 0 and max_len < 100000:
        return max_len
    return None


def _decode_supervised(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    processor: Any,
    ignore_index: int,
) -> str:
    tok = getattr(processor, "tokenizer", processor)
    keep = labels != ignore_index
    kept_ids = input_ids[keep].detach().cpu().tolist()
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is not None:
        kept_ids = [tid for tid in kept_ids if tid != pad_id]
    return tok.decode(kept_ids, skip_special_tokens=False)


def _prepare_full_text_and_images(
    collator: Any,
    processor: Any,
    example: dict,
    model_key: str,
) -> Tuple[Any, Any, bool]:
    user_text = collator._user_text(example)
    target = (example.get(collator.target_key) or "").strip()
    if not target:
        raise ValueError("missing target_json for training")

    if model_key == "aya":
        image = collator._normalize_image(example[collator.image_key])
        user_messages = collator._messages(user_text, image)
        full_messages = user_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": target}]}
        ]
        return full_messages, None, True

    user_messages = collator._user_message(user_text)
    full_messages = user_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": target}]}
    ]
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    image = example[collator.image_key]
    if hasattr(collator, "_normalize_image"):
        image = collator._normalize_image(image)

    if model_key in {"llama", "gemma"}:
        return full_text, [[image]], False
    return full_text, [image], False


def _full_length_for_example(
    collator: Any,
    processor: Any,
    example: dict,
    model_key: str,
) -> int:
    text_or_messages, images, use_template = _prepare_full_text_and_images(
        collator, processor, example, model_key
    )
    if use_template:
        # Aya tokenize=True path appears fragile; compute length via processor(text+images).
        text = processor.apply_chat_template(
            [text_or_messages],
            add_generation_prompt=False,
            tokenize=False,
        )
        text_inputs = text if isinstance(text, list) else [text]
        inputs = processor(
            text=text_inputs,
            images=[collator._normalize_image(example[collator.image_key])],
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
    else:
        inputs = processor(
            text=text_or_messages,
            images=images,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=model_key == "qwen",
        )
    return int(inputs["input_ids"].shape[-1])


def _run_collator_for_example(
    collator_cls: Any,
    processor: Any,
    example: dict,
    max_length: int,
) -> dict:
    collator = collator_cls(processor=processor, training=True, max_length=max_length)
    return collator([example])


def _process_model_split(
    model_id: str,
    model_key: str,
    collator_cls: Any,
    processor_kwargs: dict,
    dataset,
    split: str,
    lang: Optional[str],
) -> None:
    print("\n=== MAXLEN STATS ===")
    print(f"model_id={model_id}")
    print(f"split={split} lang={lang}")

    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    model_max_len = _get_model_max_length(processor)
    max_len_used = model_max_len or 1536
    print(f"model_max_length={model_max_len} max_len_used={max_len_used}")

    collator = collator_cls(processor=processor, training=True, max_length=max_len_used)
    ignore_index = getattr(collator, "ignore_index", -100)

    max_full_len = 0
    trimmed_count = 0
    total = 0
    bad_image_counts: dict[str, int] = {}
    length_bins = {
        "<=1024": 0,
        ">1024<=1536": 0,
        ">1536<=2048": 0,
        ">2048": 0,
    }

    for example in _iter_dataset(dataset):
        image = example.get(collator.image_key)
        bad_reason = _bad_image_reason(image)
        if bad_reason is not None:
            bad_image_counts[bad_reason] = bad_image_counts.get(bad_reason, 0) + 1
            continue
        if not (example.get(collator.target_key) or "").strip():
            continue
        total += 1
        try:
            full_len = _full_length_for_example(collator, processor, example, model_key)
        except Exception as exc:
            image_type = type(image)
            image_shape = getattr(image, "shape", None)
            image_mode = getattr(image, "mode", None)
            image_size = getattr(image, "size", None)
            image_bad_reason = _bad_image_reason(image)
            print(
                "full_len_failed "
                f"sample_id={_sample_id(example)} "
                f"model_id={model_id} "
                f"image_type={image_type} "
                f"image_shape={image_shape} "
                f"image_mode={image_mode} "
                f"image_size={image_size} "
                f"image_bad_reason={image_bad_reason} "
                f"err={exc}"
            )
            continue
        if full_len > max_full_len:
            max_full_len = full_len
        if full_len <= 1024:
            length_bins["<=1024"] += 1
        elif full_len <= 1536:
            length_bins[">1024<=1536"] += 1
        elif full_len <= 2048:
            length_bins[">1536<=2048"] += 1
        else:
            length_bins[">2048"] += 1

        if full_len > max_len_used:
            trimmed_count += 1
            try:
                outputs = _run_collator_for_example(
                    collator_cls, processor, example, max_len_used
                )
                labels = outputs["labels"][0]
                input_ids = outputs["input_ids"][0]
                supervised_tokens = int((labels != ignore_index).sum().item())
                decoded = _decode_supervised(labels, input_ids, processor, ignore_index)
                print(
                    "trimmed_sample "
                    f"id={_sample_id(example)} full_len={full_len} "
                    f"max_len_used={max_len_used} supervised_tokens={supervised_tokens}"
                )
                print(f"trimmed_supervised_decoded={decoded}")
            except Exception as exc:
                print(f"trimmed_decode_failed id={_sample_id(example)} err={exc}")

    print(f"total_examples={total}")
    print(f"max_full_length={max_full_len}")
    print(f"trimmed_count={trimmed_count}")
    print(f"bad_image_counts={bad_image_counts}")
    print(f"length_bins={length_bins}")


def main() -> None:
    logistics = Logistics()
    mc = ModelCards()
    splits = list(getattr(logistics, "splits", ["train"]))
    langs = list(getattr(logistics, "langs", ["en"]))

    collators = [
        ("aya", mc.aya_model_name, AyaVisionSFTCollator, {"trust_remote_code": True}),
        ("llama", mc.llama3_2vl, Llama32VisionSFTCollator, {}),
        ("gemma", mc.gemm3_12b, Gemma3VisionSFTCollator, {"trust_remote_code": True}),
        ("qwen", mc.qwen3_vl_8b_instruct, Qwen3VisionSFTCollator, {"trust_remote_code": True}),
    ]

    for split in splits:
        for lang in langs:
            dataset = load_hf_dataset(
                logistics.hf_sft_ds_id,
                split=split,
                config_name=lang,
                cache_dir=logistics.hf_cache_dir,
            )
            for model_key, model_id, collator_cls, processor_kwargs in collators:
                _process_model_split(
                    model_id=model_id,
                    model_key=model_key,
                    collator_cls=collator_cls,
                    processor_kwargs=processor_kwargs,
                    dataset=dataset,
                    split=split,
                    lang=lang,
                )


def _debug_aya_failures(
    split: str,
    lang: str,
    max_samples: int,
    max_failures: int,
) -> None:
    logistics = Logistics()
    mc = ModelCards()

    dataset = load_hf_dataset(
        logistics.hf_sft_ds_id,
        split=split,
        config_name=lang,
        cache_dir=logistics.hf_cache_dir,
    )
    processor = AutoProcessor.from_pretrained(
        mc.aya_model_name,
        trust_remote_code=True,
    )
    collator = AyaVisionSFTCollator(processor=processor, training=True, max_length=1536)

    failures = 0
    checked = 0
    for example in _iter_dataset(dataset):
        if checked >= max_samples or failures >= max_failures:
            break
        checked += 1
        image = example.get(collator.image_key)
        if _bad_image_reason(image) is not None:
            continue
        if not (example.get(collator.target_key) or "").strip():
            continue

        try:
            _ = _full_length_for_example(collator, processor, example, "aya")
        except Exception as exc:
            failures += 1
            sample_id = _sample_id(example)
            print("\n=== AYA FULL_LEN DEBUG ===")
            print(f"sample_id={sample_id} err={exc}")
            try:
                image_norm = collator._normalize_image(image)
                print(f"image_norm_type={type(image_norm)} size={getattr(image_norm, 'size', None)} mode={getattr(image_norm, 'mode', None)}")
            except Exception as img_exc:
                print(f"image_normalize_failed err={img_exc}")

            try:
                user_text = collator._user_text(example)
                user_messages = collator._messages(user_text, collator._normalize_image(image))
                full_messages = user_messages + [
                    {"role": "assistant", "content": [{"type": "text", "text": (example.get(collator.target_key) or '').strip()}]}
                ]
                text = processor.apply_chat_template(
                    [full_messages],
                    add_generation_prompt=False,
                    tokenize=False,
                )
                print(f"chat_template_text_type={type(text)}")
                if isinstance(text, list):
                    print(f"chat_template_text_len={len(text)}")
                    if text:
                        print(f"chat_template_text0_prefix={text[0][:200]}")
                else:
                    print(f"chat_template_text_prefix={text[:200]}")
            except Exception as tmpl_exc:
                print(f"apply_chat_template_failed err={tmpl_exc}")
                print(traceback.format_exc())

            try:
                ip = getattr(processor, "image_processor", None)
                if ip is None:
                    print("image_processor_missing=True")
                else:
                    ip_out = ip(collator._normalize_image(image), return_tensors="pt")
                    print(f"image_processor_keys={sorted(ip_out.keys())}")
                    for key, value in ip_out.items():
                        if isinstance(value, torch.Tensor):
                            print(f"{key}_shape={tuple(value.shape)} dtype={value.dtype}")
            except Exception as ip_exc:
                print(f"image_processor_failed err={ip_exc}")
                print(traceback.format_exc())

            try:
                if isinstance(text, list):
                    text_inputs = text
                else:
                    text_inputs = [text]
                proc_out = processor(
                    text=text_inputs,
                    images=[collator._normalize_image(image)],
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                print(f"processor_call_keys={sorted(proc_out.keys())}")
                for key, value in proc_out.items():
                    if isinstance(value, torch.Tensor):
                        print(f"{key}_shape={tuple(value.shape)} dtype={value.dtype}")
            except Exception as proc_exc:
                print(f"processor_call_failed err={proc_exc}")
                print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collator max length diagnostics")
    parser.add_argument("--debug-aya", action="store_true", help="Debug Aya full_len failures")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--lang", default="en", help="Dataset config name/lang")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples to inspect for debug")
    parser.add_argument("--max-failures", type=int, default=5, help="Max failures to print")
    args = parser.parse_args()

    if args.debug_aya:
        _debug_aya_failures(
            split=args.split,
            lang=args.lang,
            max_samples=args.max_samples,
            max_failures=args.max_failures,
        )
    else:
        main()
