import json
from typing import List, Optional, Any

import torch
from transformers import AutoProcessor

from config.logistics import Logistics, ModelCards
from config.queries import Queries
from src.collators.llama32_collator import Llama32VisionSFTCollator
from src.datasets.loader import load_hf_dataset


def _format_full_text(collator: Llama32VisionSFTCollator, example: dict, processor) -> str:
    modality = collator._get_modality(example)
    user_text = collator._user_text(example, modality)
    messages = collator._user_message(user_text, modality)
    target = (example.get(collator.target_key) or "").strip()
    full_messages = messages + [
        {"role": "assistant", "content": [{"type": "text", "text": target}]}
    ]
    return processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )





def _summarize_batch(
    outputs: dict,
    label_outputs: bool,
) -> None:
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


def _select_samples_by_modality(
    dataset,
    modalities: tuple[str, ...] = ("both", "text", "image"),
    max_scan: int = 20000,
) -> List[dict]:
    selected = {}
    scanned = 0

    for row in dataset:
        scanned += 1
        modality = (row.get("modality") or "").strip().lower()
        if modality in modalities and modality not in selected:
            selected[modality] = row
        if len(selected) == len(modalities) or scanned >= max_scan:
            break

    missing = [m for m in modalities if m not in selected]
    if missing:
        raise ValueError(
            f"Could not find all required modalities {modalities}. "
            f"missing={missing}, scanned={scanned}"
        )

    return [selected[m] for m in modalities]


def _percentile(sorted_vals: List[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    idx = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]


def compute_length_stats(
    dataset,
    collator: Llama32VisionSFTCollator,
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
        image = example["image"]
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
                "content": [{"type": "text", "text": (example.get("target_json") or "").strip()}],
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



def _print_mask_check(labels: torch.Tensor, input_ids: torch.Tensor, processor: Any) -> None:
    # 1. Check if padding is masked
    pad_id = processor.tokenizer.pad_token_id
    # We only check padding if it exists in input_ids
    padding_mask = input_ids == pad_id
    if padding_mask.any():
        masked_padding = torch.all(labels[padding_mask] == -100).item()
    else:
        masked_padding = True
        
    # 2. Check if the prompt is masked and assistant is visible
    # We decode the labels (filtering -100) to see what the model is actually training on
    trainable_tokens = labels[labels != -100].tolist()
    trainable_text = processor.tokenizer.decode(trainable_tokens)
    
    # Validation logic
    has_assistant_loss = len(trainable_tokens) > 0
    # A successful mask should NOT contain the system or user prompt text
    prompt_masked_well = "Analyze the relationship" not in trainable_text 
    
    print(f"mask_padding_ok={masked_padding}")
    print(f"assistant_loss_present={has_assistant_loss}")
    print(f"trainable_text_snippet='{trainable_text}'")
    
    assert masked_padding, "Expected padding tokens to be masked (-100)."
    assert has_assistant_loss, "Expected assistant tokens to contribute to loss."
    assert prompt_masked_well, "Prompt text found in trainable labels! Masking failed."


def _decode_ids(ids: torch.Tensor, processor) -> str:
    ids = ids.detach().cpu().tolist()
    return processor.tokenizer.decode(ids, skip_special_tokens=False)


def _decode_labels(labels: torch.Tensor, input_ids: torch.Tensor, processor) -> str:
    keep = labels != -100
    kept_ids = input_ids[keep].detach().cpu().tolist()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        kept_ids = [token_id for token_id in kept_ids if token_id != pad_id]
    return processor.tokenizer.decode(kept_ids, skip_special_tokens=False)


def _assert_cross_attention_alignment(
    outputs: dict,
    samples: List[dict],
    collator: Llama32VisionSFTCollator,
) -> None:
    if "cross_attention_mask" not in outputs:
        print("cross_attention_mask_present=False")
        return

    cross_attention_mask = outputs["cross_attention_mask"]
    input_ids = outputs["input_ids"]
    print(f"cross_attention_mask_shape={tuple(cross_attention_mask.shape)}")
    print(f"input_ids_shape={tuple(input_ids.shape)}")

    assert cross_attention_mask.ndim >= 2, "cross_attention_mask must have sequence dimension."
    assert (
        cross_attention_mask.shape[0] == input_ids.shape[0]
    ), "cross_attention_mask batch size must match input_ids."
    assert (
        cross_attention_mask.shape[1] == input_ids.shape[1]
    ), "cross_attention_mask seq_len must match input_ids seq_len."

    text_rows = [
        idx
        for idx, sample in enumerate(samples)
        if collator._get_modality(sample) == "text"
    ]
    image_rows = [
        idx
        for idx, sample in enumerate(samples)
        if collator._get_modality(sample) in {"both", "image"}
    ]
    if text_rows:
        text_sum = cross_attention_mask[text_rows].sum().item()
        print(f"cross_attention_mask_text_rows_total={text_sum}")
        assert (
            text_sum == 0
        ), "cross_attention_mask for text-only rows should be zeroed."
    if image_rows:
        image_sum = cross_attention_mask[image_rows].sum().item()
        print(f"cross_attention_mask_image_rows_total={image_sum}")
        assert (
            image_sum > 0
        ), "cross_attention_mask for image rows should contain active attention."


def _assert_validation_errors(
    collator: Llama32VisionSFTCollator,
    sample_text: dict,
    sample_image: dict,
) -> None:
    try:
        collator([])
        raise AssertionError("Expected ValueError for empty batch.")
    except ValueError as exc:
        print(f"empty_batch_error={exc}")
        assert "empty batch" in str(exc)

    missing_image = dict(sample_image)
    missing_image.pop(collator.image_key, None)
    try:
        collator([missing_image])
        raise AssertionError("Expected ValueError for missing image.")
    except ValueError as exc:
        print(f"missing_image_error={exc}")
        assert "missing" in str(exc) and collator.image_key in str(exc)

    missing_target = dict(sample_text)
    missing_target[collator.target_key] = ""
    try:
        collator([missing_target])
        raise AssertionError("Expected ValueError for missing target_json in training.")
    except ValueError as exc:
        print(f"missing_target_error={exc}")
        assert "missing target_json for training" in str(exc)




def main() -> None:
    logistics = Logistics()
    mc = ModelCards()
    dataset_id = logistics.hf_sft_ds_id
    cache_dir = logistics.hf_cache_dir
    processor = AutoProcessor.from_pretrained(mc.llama3_2vl)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Decoder-only generation must use left padding.
        tokenizer.padding_side = "left"


    # Processor sanity info
    print("Loaded processor:", mc.llama3_2vl)
    print("Tokenizer padding token:", processor.tokenizer.pad_token, "| pad_id:", processor.tokenizer.pad_token_id)
    print("Tokenizer eos_token_id:", processor.tokenizer.eos_token_id)
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
    samples = _select_samples_by_modality(dataset, ("both", "text", "image"))
    modalities = [
        (sample.get("modality") or "").strip().lower()
        for sample in samples
    ]

    print(f"dataset_id={dataset_id}")
    print(f"samples_loaded={len(samples)}")
    print(f"sample_modalities={modalities}")
    print(f"system_prompt={Queries().SYSTEM_PROMPT[:120]}...")

    train_collator = Llama32VisionSFTCollator(
        processor=processor,
        training=True,
        max_length=2048,
    )
    infer_collator = Llama32VisionSFTCollator(
        processor=processor,
        training=False,
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
    _assert_cross_attention_alignment(train_outputs, samples, train_collator)

    for idx, sample in enumerate(samples):
        modality = train_collator._get_modality(sample)
        user_text = train_collator._user_text(sample, modality)
        user_messages = train_collator._user_message(user_text, modality)
        user_content_types = [item["type"] for item in user_messages[-1]["content"]]

        print(f"\n--- sample[{idx}] modality={modality} ---")
        print(f"user_content_types={user_content_types}")
        if modality == "text":
            assert user_content_types == ["text"], "Text-only sample should omit image content."
        else:
            assert "image" in user_content_types, "Image-capable sample should include image content."

        prompt_text = processor.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print("\nformatted_prompt_text_train=")
        print(prompt_text)

        full_text = _format_full_text(train_collator, sample, processor)
        print("\nformatted_full_text=")
        print(full_text)
        target_in_text = (sample.get("target_json") or "") in full_text
        print("target_json_in_text=" + str(target_in_text))
        assert target_in_text, "Expected target_json to be included in assistant text."

        print("\n--- Masking Verification ---")
        _print_mask_check(
            train_outputs["labels"][idx],
            train_outputs["input_ids"][idx],
            processor,
        )

        print(f"\ndecoded_input_ids_sample{idx}=")
        print(_decode_ids(train_outputs["input_ids"][idx], processor))

        print(f"\ndecoded_labels_sample{idx}_loss_tokens_only=")
        print(
            _decode_labels(
                train_outputs["labels"][idx],
                train_outputs["input_ids"][idx],
                processor,
            )
        )

        label_ids = train_outputs["labels"][idx][train_outputs["labels"][idx] != -100].tolist()
        if label_ids:
            print("\nlabel_tokens_decoded=")
            print(processor.tokenizer.decode(label_ids, skip_special_tokens=False))
        else:
            print("label_tokens_decoded=")
            print("<none>")

        labels = train_outputs["labels"][idx]
        input_ids = train_outputs["input_ids"][idx]
        non_masked = (labels != -100).nonzero(as_tuple=True)[0]
        if len(non_masked) > 0:
            first_loss_idx = non_masked[0].item()
            transition_tokens = input_ids[max(0, first_loss_idx - 5) : first_loss_idx + 5].tolist()
            transition_text = processor.tokenizer.decode(transition_tokens, skip_special_tokens=False)
            print(f"Token transition at index: {first_loss_idx}")
            print(f"Transition context: '...{transition_text}...'")
        else:
            print("Token transition at index: <none>")

    sample_text = next(sample for sample in samples if train_collator._get_modality(sample) == "text")
    sample_image = next(sample for sample in samples if train_collator._get_modality(sample) in {"both", "image"})
    _assert_validation_errors(train_collator, sample_text, sample_image)

    print("\n=== INFERENCE ===")
    infer_outputs = infer_collator(samples)
    _summarize_batch(infer_outputs, label_outputs=False)
    assert "labels" not in infer_outputs, "Inference outputs should not include labels."
    assert "input_ids" in infer_outputs, "Inference outputs should include input_ids."
    _assert_cross_attention_alignment(infer_outputs, samples, infer_collator)
    for idx, sample in enumerate(samples):
        modality = infer_collator._get_modality(sample)
        print(f"formatted_prompt_text_sample{idx}_modality_{modality}=")
        print(
            processor.apply_chat_template(
                infer_collator._user_message(
                    infer_collator._user_text(sample, modality),
                    modality,
                ),
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        print(
            f"assistant_output_expected_sample{idx}="
            + json.dumps(sample.get("target_json", ""))[:120]
            + "..."
        )

    # compute_length_stats(dataset, train_collator, processor)


def run_main():
    main()

if __name__ == "__main__":
    main()
