import json
from typing import List, Optional

import torch
from transformers import AutoProcessor

from config.logistics import Logistics, ModelCards
from config.queries import Queries
from src.collators.gemma3_collator import Gemma3VisionSFTCollator
from src.datasets.loader import load_hf_dataset


def _format_full_text(
    collator: Gemma3VisionSFTCollator,
    example: dict,
    processor,
) -> str:
    user_text = collator._user_text(example)
    messages = collator._user_message(user_text)
    target = (example.get(collator.target_key) or "").strip()
    full_messages = messages + [
        {"role": "assistant", "content": [{"type": "text", "text": target}]}
    ]
    return processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _print_mask_check(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int,
    processor,
) -> None:
    masked_prompt = torch.all(labels[:prompt_len] == -100).item()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        masked_padding = True
        print("pad_token_id_missing=True")
    else:
        masked_padding = torch.all(labels[input_ids == pad_id] == -100).item()
    has_assistant_loss = torch.any(labels[prompt_len:] != -100).item()
    print(f"mask_prompt_ok={masked_prompt}")
    print(f"mask_padding_ok={masked_padding}")
    print(f"assistant_loss_present={has_assistant_loss}")
    assert masked_prompt, "Expected prompt tokens to be masked (-100)."
    if pad_id is not None:
        assert masked_padding, "Expected padding tokens to be masked (-100)."
    assert has_assistant_loss, "Expected assistant tokens to contribute to loss."


def _summarize_batch(outputs: dict, label_outputs: bool) -> None:
    print(f"processor_keys={sorted(outputs.keys())}")
    if label_outputs:
        labels = outputs["labels"]
        input_ids = outputs["input_ids"]
        print(f"labels_shape={tuple(labels.shape)}")
        print(f"input_ids_shape={tuple(input_ids.shape)}")


def _select_samples(dataset, max_samples: int) -> List[dict]:
    if hasattr(dataset, "select"):
        return dataset.select(range(max_samples))
    return list(dataset.take(max_samples))


def _percentile(sorted_vals: List[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    idx = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]


def compute_length_stats(
    dataset,
    collator: Gemma3VisionSFTCollator,
    processor,
    max_samples: Optional[int] = None,
) -> None:
    lengths: List[int] = []
    text_lengths: List[int] = []
    image_token_lengths: List[int] = []
    widths: List[int] = []
    heights: List[int] = []
    total = len(dataset) if hasattr(dataset, "__len__") else None
    limit = max_samples if max_samples is not None else total

    for i, example in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        image = collator._normalize_image(example["image"])
        if hasattr(image, "size"):
            width, height = image.size
            widths.append(int(width))
            heights.append(int(height))

        full_text = _format_full_text(collator, example, processor)
        inputs = processor(
            text=full_text,
            images=[[image]],
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        full_len = int(inputs["input_ids"].shape[-1])
        lengths.append(full_len)

        user_text = collator._user_text(example)
        text_only_messages = []
        if collator.system_prompt:
            text_only_messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": collator.system_prompt}],
                }
            )
        text_only_messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            }
        )
        text_only_messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": (example.get("target_json") or "").strip()}
                ],
            }
        )
        text_only = processor.apply_chat_template(
            text_only_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        text_inputs = processor(
            text=text_only,
            images=None,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        text_len = int(text_inputs["input_ids"].shape[-1])
        text_lengths.append(text_len)
        image_token_lengths.append(max(0, full_len - text_len))

    lengths.sort()
    text_lengths.sort()
    image_token_lengths.sort()
    widths.sort()
    heights.sort()

    print("\n=== LENGTH STATS ===")
    print(f"samples={len(lengths)}")
    print(
        "input_ids_len_p50/p90/p95/p99="
        f"{_percentile(lengths, 50)}/"
        f"{_percentile(lengths, 90)}/"
        f"{_percentile(lengths, 95)}/"
        f"{_percentile(lengths, 99)}"
    )
    print(
        "text_len_p50/p90/p95/p99="
        f"{_percentile(text_lengths, 50)}/"
        f"{_percentile(text_lengths, 90)}/"
        f"{_percentile(text_lengths, 95)}/"
        f"{_percentile(text_lengths, 99)}"
    )
    print(
        "image_token_len_p50/p90/p95/p99="
        f"{_percentile(image_token_lengths, 50)}/"
        f"{_percentile(image_token_lengths, 90)}/"
        f"{_percentile(image_token_lengths, 95)}/"
        f"{_percentile(image_token_lengths, 99)}"
    )
    if widths and heights:
        print(
            "image_w_p50/p90/p95/p99="
            f"{_percentile(widths, 50)}/"
            f"{_percentile(widths, 90)}/"
            f"{_percentile(widths, 95)}/"
            f"{_percentile(widths, 99)}"
        )
        print(
            "image_h_p50/p90/p95/p99="
            f"{_percentile(heights, 50)}/"
            f"{_percentile(heights, 90)}/"
            f"{_percentile(heights, 95)}/"
            f"{_percentile(heights, 99)}"
        )

    recommended = _percentile(lengths, 95)
    print(f"recommended_max_length_p95={recommended}")


def get_max_length_from_dataset(
    dataset,
    collator: Gemma3VisionSFTCollator,
    processor,
    max_samples: Optional[int] = None,
) -> int:
    max_len = 0
    printed = 0
    total = len(dataset) if hasattr(dataset, "__len__") else None
    limit = max_samples if max_samples is not None else total

    for i, example in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        image = collator._normalize_image(example["image"])
        full_text = _format_full_text(collator, example, processor)
        prompt_text = processor.apply_chat_template(
            collator._user_message(collator._user_text(example)),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=full_text,
            images=[[image]],
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        prompt_inputs = processor(
            text=prompt_text,
            images=[[image]],
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        full_len = int(inputs["input_ids"].shape[-1])
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        if full_len >= 1024 and printed < 10:
            print(f"\nlong_sequence idx={i} full_len={full_len} prompt_len={prompt_len}")
            print("full_text=")
            print(full_text)
            print("prompt_text=")
            print(prompt_text)
            print("full_text_decoded=")
            print(
                processor.tokenizer.decode(
                    inputs["input_ids"][0], skip_special_tokens=False
                )
            )
            print("prompt_text_decoded=")
            print(
                processor.tokenizer.decode(
                    prompt_inputs["input_ids"][0], skip_special_tokens=False
                )
            )
            printed += 1
        if full_len > max_len:
            max_len = full_len

    return max_len


def main() -> None:
    try:
        logistics = Logistics()
        mc = ModelCards()
        dataset_id = logistics.hf_sft_ds_id
        cache_dir = logistics.hf_cache_dir
        processor = AutoProcessor.from_pretrained(
            mc.gemm3_12b,
            trust_remote_code=True,
        )
        # Processor sanity info
        print("Loaded processor:", mc.gemm3_12b)
        print("Tokenizer padding token:", processor.tokenizer.pad_token, "| pad_id:", processor.tokenizer.pad_token_id)
        print("Model max length:", getattr(processor.tokenizer, "model_max_length", "unknown"))
        print(f"vocab_size={len(processor.tokenizer)}")
        print(f"padding_side={processor.tokenizer.padding_side}")
        print(f"special_tokens_map={processor.tokenizer.special_tokens_map}")
        print(f"additional_special_tokens={processor.tokenizer.additional_special_tokens}")
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            crop_count = getattr(image_processor, "num_crops", None)
            crop_size = getattr(image_processor, "crop_size", None)
            print(f"image_processor_num_crops={crop_count}")
            print(f"image_processor_crop_size={crop_size}")

        dataset = load_hf_dataset(
            dataset_id,
            split="train",
            config_name="en",
            cache_dir=cache_dir,
        )
        samples = _select_samples(dataset, 4)

        print(f"dataset_id={dataset_id}")
        print(f"samples_loaded={len(samples)}")
        print(f"system_prompt={Queries().SYSTEM_PROMPT[:120]}...")

        train_collator = Gemma3VisionSFTCollator(
            processor=processor,
            training=True,
            max_length=2048,
        )

        print("\n=== TRAINING ===")
        train_outputs = train_collator(samples)
        _summarize_batch(train_outputs, label_outputs=True)
        assert "labels" in train_outputs, "Training outputs should include labels."
        assert "input_ids" in train_outputs, "Training outputs should include input_ids."
        assert (
            train_outputs["labels"].shape == train_outputs["input_ids"].shape
        ), "Labels and input_ids must have matching shapes."

        first = samples[0]
        prompt_text = processor.apply_chat_template(
            train_collator._user_message(train_collator._user_text(first)),
            tokenize=False,
            add_generation_prompt=True,
        )
        print("\nformatted_prompt_text_train=")
        print(prompt_text)
        full_text = _format_full_text(train_collator, first, processor)
        print("\nformatted_full_text=")
        print(full_text)
        target_in_text = first["target_json"] in full_text
        print("target_json_in_text=" + str(target_in_text))
        assert target_in_text, "Expected target_json to be included in assistant text."

        prompt_text = processor.apply_chat_template(
            train_collator._user_message(train_collator._user_text(first)),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_len = train_collator._prompt_length(first["image"], prompt_text)
        _print_mask_check(
            train_outputs["labels"][0],
            train_outputs["input_ids"][0],
            prompt_len,
            processor,
        )
        label_ids = train_outputs["labels"][0][train_outputs["labels"][0] != -100].tolist()
        if label_ids:
            print("\nlabel_tokens_decoded=")
            print(processor.tokenizer.decode(label_ids, skip_special_tokens=False))
        else:
            print("label_tokens_decoded=")
            print("<none>")

        infer_collator = Gemma3VisionSFTCollator(
            processor=processor,
            training=False,
            max_length=2048,
        )

        print("\n=== INFERENCE ===")
        infer_outputs = infer_collator(samples)
        _summarize_batch(infer_outputs, label_outputs=False)
        assert "labels" not in infer_outputs, "Inference outputs should not include labels."
        assert "input_ids" in infer_outputs, "Inference outputs should include input_ids."
        print("formatted_prompt_text=")
        print(
            processor.apply_chat_template(
                infer_collator._user_message(infer_collator._user_text(first)),
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        print(
            "assistant_output_expected="
            + json.dumps(first["target_json"])[:120]
            + "..."
        )

        # max_len = get_max_length_from_dataset(dataset, train_collator, processor)
        # print(f"max_length_dataset={max_len}")
        # compute_length_stats(dataset, train_collator, processor)
    except Exception as exc:
        raise RuntimeError(f"Gemma3 collator test failed: {exc}") from exc


if __name__ == "__main__":
    main()
