import json
import inspect

import torch
from transformers import AutoProcessor

from config.logistics import Logistics, ModelCards
from config.queries import Queries
from src.collators.qwen3_collator import Qwen3VisionSFTCollator
from src.datasets.loader import load_hf_dataset


def _format_full_text(
    collator: Qwen3VisionSFTCollator,
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


def _get_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def _print_mask_check(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int,
    processor,
) -> None:
    masked_prompt = torch.all(labels[:prompt_len] == -100).item()
    tok = _get_tokenizer(processor)
    pad_id = tok.pad_token_id
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


def _select_samples(dataset, max_samples: int) -> list[dict]:
    if max_samples is None: return list(dataset)
    if hasattr(dataset, "select"):
        return list(dataset.select(range(max_samples)))
    if hasattr(dataset, "take"):
        return list(dataset.take(max_samples))
    return list(dataset)[:max_samples]


def _compute_max_lengths(
    collator: Qwen3VisionSFTCollator,
    dataset,
    processor,
    max_samples: int = 256,
) -> tuple[int, int]:
    """
    NOTE: This computes lengths using tokenizer-only. With VL models, input_ids
    may include extra vision tokens, so these are approximate/debug-only.
    """
    samples = _select_samples(dataset, max_samples)
    max_full = 0
    max_prompt = 0
    tok = _get_tokenizer(processor)

    for example in samples:
        user_text = collator._user_text(example)
        user_messages = collator._user_message(user_text)

        prompt_text = processor.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_len = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
        max_prompt = max(max_prompt, prompt_len)

        if collator.training:
            full_text = _format_full_text(collator, example, processor)
            full_len = len(tok(full_text, add_special_tokens=False)["input_ids"])
            max_full = max(max_full, full_len)

    return max_full, max_prompt


def _decode_ids(ids: torch.Tensor, processor) -> str:
    tok = _get_tokenizer(processor)
    ids = ids.detach().cpu().tolist()
    return tok.decode(ids, skip_special_tokens=False)


def _decode_labels(labels: torch.Tensor, input_ids: torch.Tensor, processor) -> str:
    """
    Decode only the tokens that contribute to loss (labels != -100).
    Uses input_ids to recover the actual token ids for kept positions.
    """
    tok = _get_tokenizer(processor)
    pad_id = tok.pad_token_id

    keep = labels != -100
    kept_ids = input_ids[keep].detach().cpu().tolist()

    if pad_id is not None:
        kept_ids = [t for t in kept_ids if t != pad_id]

    return tok.decode(kept_ids, skip_special_tokens=False)


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    n = len(needle)
    if n == 0:
        return -1
    for j in range(0, len(haystack) - n + 1):
        if haystack[j : j + n] == needle:
            return j
    return -1


def _assistant_prompt_len_from_input_ids(input_ids: torch.Tensor, processor) -> int:
    """
    Option B mirror: find '<|im_start|>assistant' marker inside the already-built input_ids,
    and treat everything through that marker as prompt (to be masked).
    """
    tok = _get_tokenizer(processor)
    candidates = [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
    ]
    marker_variants = [tok(c, add_special_tokens=False)["input_ids"] for c in candidates]

    ids = input_ids.detach().cpu().tolist()
    for marker_ids in marker_variants:
        start = _find_subsequence(ids, marker_ids)
        if start != -1:
            return start + len(marker_ids)

    raise RuntimeError(
        "Could not find '<|im_start|>assistant' marker in input_ids "
        "(likely truncation or unexpected chat template)."
    )


def main() -> None:
    try:
        logistics = Logistics()
        mc = ModelCards()

        dataset_id = logistics.hf_sft_ds_id
        cache_dir = logistics.hf_cache_dir

        processor = AutoProcessor.from_pretrained(
            mc.qwen3_vl_8b_instruct,
            trust_remote_code=True,
        )
        tok = _get_tokenizer(processor)

        print("Loaded processor:", mc.qwen3_vl_8b_instruct)
        print("processor_type=", type(processor))
        print("has_image_processor=", hasattr(processor, "image_processor"))
        print("Tokenizer padding token:", tok.pad_token, "| pad_id:", tok.pad_token_id)
        print("Model max length:", getattr(tok, "model_max_length", "unknown"))
        print(f"vocab_size={len(tok)}")
        print(f"padding_side={tok.padding_side}")

        dataset = load_hf_dataset(
            dataset_id,
            split="train",
            config_name="en",
            cache_dir=cache_dir,
        )
        samples = _select_samples(dataset, 4)

        print(f"dataset_id={dataset_id}")
        print(f"samples_loaded={len(samples)}")
        print(f"system_prompt={Queries().SYSTEM_PROMPT}...")

        train_collator = Qwen3VisionSFTCollator(
            processor=processor,
            training=True,
            max_length=2048,
        )

        # Quick schema sanity
        first = samples[0]
        assert train_collator.image_key in first, f"Missing {train_collator.image_key} in sample[0]"
        assert (first.get(train_collator.target_key) or "").strip(), "Missing target_json in sample[0]"

        print("\n=== TRAINING ===")
        train_outputs = train_collator(samples)
        _summarize_batch(train_outputs, label_outputs=True)

        assert "labels" in train_outputs, "Training outputs should include labels."
        assert "input_ids" in train_outputs, "Training outputs should include input_ids."
        assert train_outputs["labels"].shape == train_outputs["input_ids"].shape, (
            "Labels and input_ids must have matching shapes."
        )

        full_text = _format_full_text(train_collator, first, processor)
        print("formatted_full_text=")
        print(full_text)

        target_str = (first.get("target_json") or "").strip()
        target_in_text = target_str in full_text
        print("target_json_in_text=" + str(target_in_text))
        assert target_in_text, "Expected target_json to be included in assistant text."

        # Option B: compute prompt_len by finding assistant marker inside input_ids
        prompt_len = _assistant_prompt_len_from_input_ids(
            train_outputs["input_ids"][0],
            processor,
        )
        print(f"prompt_len_from_input_ids={prompt_len}")

        _print_mask_check(
            train_outputs["labels"][0],
            train_outputs["input_ids"][0],
            prompt_len,
            processor,
        )

        # --- Decode and print ---
        print("\ndecoded_input_ids_sample0=")
        print(_decode_ids(train_outputs["input_ids"][0], processor))

        print("\ndecoded_labels_sample0_loss_tokens_only=")
        print(
            _decode_labels(
                train_outputs["labels"][0],
                train_outputs["input_ids"][0],
                processor,
            )
        )

        infer_collator = Qwen3VisionSFTCollator(
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

        print("assistant_output_expected=" + target_str[:120] + "...")

        # max_full, max_prompt = _compute_max_lengths(train_collator, dataset, processor, max_samples=len(dataset))
        # print(f"max_full_length_tokens={max_full}")
        # print(f"max_prompt_length_tokens={max_prompt}")

    except Exception as exc:
        raise RuntimeError(f"Qwen3 collator test failed: {exc}") from exc


if __name__ == "__main__":
    main()
