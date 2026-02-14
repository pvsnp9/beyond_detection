import inspect

import torch
from transformers import AutoProcessor

from config.logistics import Logistics, ModelCards
from config.queries import Queries
from src.collators.qwen3_dpo_collator import Qwen3VisionDPOCollator
from src.datasets.loader import load_hf_dpo_dataset


def _get_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def _format_full_text(
    collator: Qwen3VisionDPOCollator,
    example: dict,
    processor,
    answer_key: str,
) -> str:
    user_text = collator._user_text(example)
    messages = collator._user_message(user_text)
    answer = (example.get(answer_key) or "").strip()
    full_messages = messages + [
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
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


def _summarize_batch(outputs: dict) -> None:
    print(f"processor_keys={sorted(outputs.keys())}")
    print(f"prompt_input_ids_shape={tuple(outputs['prompt_input_ids'].shape)}")
    print(f"chosen_input_ids_shape={tuple(outputs['chosen_input_ids'].shape)}")
    print(f"rejected_input_ids_shape={tuple(outputs['rejected_input_ids'].shape)}")
    print(f"chosen_labels_shape={tuple(outputs['chosen_labels'].shape)}")
    print(f"rejected_labels_shape={tuple(outputs['rejected_labels'].shape)}")


def _select_samples(dataset, max_samples: int) -> list[dict]:
    if max_samples is None:
        return list(dataset)
    if hasattr(dataset, "select"):
        return list(dataset.select(range(max_samples)))
    if hasattr(dataset, "take"):
        return list(dataset.take(max_samples))
    return list(dataset)[:max_samples]


def _decode_ids(ids: torch.Tensor, processor) -> str:
    tok = _get_tokenizer(processor)
    ids = ids.detach().cpu().tolist()
    return tok.decode(ids, skip_special_tokens=False)


def _decode_labels(labels: torch.Tensor, input_ids: torch.Tensor, processor) -> str:
    tok = _get_tokenizer(processor)
    pad_id = tok.pad_token_id
    keep = labels != -100
    kept_ids = input_ids[keep].detach().cpu().tolist()
    if pad_id is not None:
        kept_ids = [t for t in kept_ids if t != pad_id]
    return tok.decode(kept_ids, skip_special_tokens=False)


def _assert_exact_label_span(
    processor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    prompt_text: str,
    full_text: str,
) -> None:
    if not full_text.startswith(prompt_text):
        raise AssertionError("prompt_text is not a prefix of full_text in test")
    target_text = full_text[len(prompt_text) :]
    target_ids = processor.tokenizer(target_text, add_special_tokens=False)["input_ids"]
    if not target_ids:
        raise AssertionError("target text tokenized to empty sequence in test")

    ids = input_ids.detach().cpu().tolist()
    keep = (labels != -100).detach().cpu().tolist()
    kept_ids = [tid for tid, k in zip(ids, keep) if k]

    if kept_ids != target_ids:
        raise AssertionError("labels do not match target token span exactly")

    start_idx = None
    for idx in range(len(ids) - len(target_ids) + 1):
        if ids[idx : idx + len(target_ids)] == target_ids:
            start_idx = idx
            break
    if start_idx is None:
        raise AssertionError("target token span not found in input_ids")

    for i, k in enumerate(keep):
        if start_idx <= i < start_idx + len(target_ids):
            if not k:
                raise AssertionError("expected assistant target tokens to be unmasked")
        else:
            if k:
                raise AssertionError("expected non-target tokens to be masked")


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    n = len(needle)
    if n == 0:
        return -1
    for j in range(0, len(haystack) - n + 1):
        if haystack[j : j + n] == needle:
            return j
    return -1


def _assistant_prompt_len_from_input_ids(input_ids: torch.Tensor, processor) -> int:
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

        dataset = load_hf_dpo_dataset(
            split="train",
            config_name="en",
            cache_dir=cache_dir,
        )
        samples = _select_samples(dataset, 4)

        print(f"dataset_id={logistics.hf_dpo_ds_id}")
        print(f"samples_loaded={len(samples)}")
        print(f"system_prompt={Queries().SYSTEM_PROMPT}...")

        collator = Qwen3VisionDPOCollator(
            processor=processor,
            max_length=2048,
        )

        first = samples[0]
        assert collator.image_key in first, f"Missing {collator.image_key} in sample[0]"
        assert (first.get(collator.chosen_key) or "").strip(), "Missing chosen in sample[0]"
        assert (first.get(collator.rejected_key) or "").strip(), "Missing rejected in sample[0]"

        print("\n=== DPO TRAINING ===")
        outputs = collator(samples)
        _summarize_batch(outputs)

        assert outputs["chosen_labels"].shape == outputs["chosen_input_ids"].shape
        assert outputs["rejected_labels"].shape == outputs["rejected_input_ids"].shape

        chosen_text = _format_full_text(collator, first, processor, collator.chosen_key)
        rejected_text = _format_full_text(collator, first, processor, collator.rejected_key)

        print("formatted_chosen_text=")
        print(chosen_text)
        print("formatted_rejected_text=")
        print(rejected_text)

        chosen_str = (first.get(collator.chosen_key) or "").strip()
        rejected_str = (first.get(collator.rejected_key) or "").strip()
        assert chosen_str in chosen_text, "Expected chosen to be included in assistant text."
        assert rejected_str in rejected_text, "Expected rejected to be included in assistant text."

        chosen_prompt_len = _assistant_prompt_len_from_input_ids(
            outputs["chosen_input_ids"][0],
            processor,
        )
        rejected_prompt_len = _assistant_prompt_len_from_input_ids(
            outputs["rejected_input_ids"][0],
            processor,
        )
        print(f"chosen_prompt_len_from_input_ids={chosen_prompt_len}")
        print(f"rejected_prompt_len_from_input_ids={rejected_prompt_len}")

        _print_mask_check(
            outputs["chosen_labels"][0],
            outputs["chosen_input_ids"][0],
            chosen_prompt_len,
            processor,
        )
        _print_mask_check(
            outputs["rejected_labels"][0],
            outputs["rejected_input_ids"][0],
            rejected_prompt_len,
            processor,
        )

        prompt_text = processor.apply_chat_template(
            collator._user_message(collator._user_text(first)),
            tokenize=False,
            add_generation_prompt=True,
        )
        _assert_exact_label_span(
            processor,
            outputs["chosen_input_ids"][0],
            outputs["chosen_labels"][0],
            prompt_text,
            chosen_text,
        )
        _assert_exact_label_span(
            processor,
            outputs["rejected_input_ids"][0],
            outputs["rejected_labels"][0],
            prompt_text,
            rejected_text,
        )

        print("\ndecoded_chosen_input_ids_sample0=")
        print(_decode_ids(outputs["chosen_input_ids"][0], processor))

        print("\ndecoded_chosen_labels_sample0_loss_tokens_only=")
        print(
            _decode_labels(
                outputs["chosen_labels"][0],
                outputs["chosen_input_ids"][0],
                processor,
            )
        )

        print("\ndecoded_rejected_input_ids_sample0=")
        print(_decode_ids(outputs["rejected_input_ids"][0], processor))

        print("\ndecoded_rejected_labels_sample0_loss_tokens_only=")
        print(
            _decode_labels(
                outputs["rejected_labels"][0],
                outputs["rejected_input_ids"][0],
                processor,
            )
        )

    except Exception as exc:
        raise RuntimeError(f"Qwen3 DPO collator test failed: {exc}") from exc


if __name__ == "__main__":
    main()
