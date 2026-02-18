from transformers import AutoProcessor

from config.logistics import Logistics, ModelCards
from src.collators.dpo_inference_collator import (
    aya_dpo_inference_collator,
    gemma_dpo_inference_collator,
    llama_dpo_inference_collator,
    qwen_dpo_inference_collator,
)
from src.datasets.loader import load_hf_dpo_dataset


def _select_samples(dataset, max_samples: int) -> list[dict]:
    if max_samples is None:
        return list(dataset)
    if hasattr(dataset, "select"):
        return list(dataset.select(range(max_samples)))
    if hasattr(dataset, "take"):
        return list(dataset.take(max_samples))
    return list(dataset)[:max_samples]


def _print_messages(messages_batch, images_batch) -> None:
    print(f"messages_batch_len={len(messages_batch)}")
    print(f"images_batch_len={0 if images_batch is None else len(images_batch)}")
    print("messages_sample0=")
    print(messages_batch[0])
    print("messages_sample1=")
    print(messages_batch[1])


def _decode_and_print(inputs, processor) -> None:
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        print("input_ids_missing=True")
        return
    decoded = processor.batch_decode(input_ids, skip_special_tokens=False)
    print("decoded_sample0=")
    print(decoded[0])
    print("decoded_sample1=")
    print(decoded[1])


def _run_collator(
    name: str,
    collator_fn,
    processor,
    samples: list[dict],
    modes: list[str],
) -> None:
    print(f"\n=== {name} ===")
    for mode in modes:
        print(f"\n-- mode={mode} --")
        messages_batch, images_batch = collator_fn(
            samples,
            mode=mode,
            processor=None,
        )
        _print_messages(messages_batch, images_batch)
        inputs = collator_fn(
            samples,
            mode=mode,
            processor=processor,
            max_length=2048,
        )
        print(f"input_keys={sorted(inputs.keys())}")
        _decode_and_print(inputs, processor)


def main() -> None:
    try:
        logistics = Logistics()
        mc = ModelCards()

        dataset = load_hf_dpo_dataset(
            split="test",
            config_name="en",
            cache_dir=logistics.hf_cache_dir,
        )
        samples = _select_samples(dataset, 2)
        if len(samples) < 2:
            raise RuntimeError("Need at least 2 samples for this test.")

        modes = ["text", "image", "both"]

        processors = {
            "aya": AutoProcessor.from_pretrained(mc.aya_model_name, use_fast=True),
            "qwen": AutoProcessor.from_pretrained(
                mc.qwen3_vl_8b_instruct, trust_remote_code=True
            ),
            "llama": AutoProcessor.from_pretrained(mc.llama3_2vl, use_fast=True),
            "gemma": AutoProcessor.from_pretrained(mc.gemm3_12b, use_fast=True),
        }

        _run_collator("aya_dpo_inference_collator", aya_dpo_inference_collator, processors["aya"], samples, modes)
        _run_collator("qwen_dpo_inference_collator", qwen_dpo_inference_collator, processors["qwen"], samples, modes)
        _run_collator("llama_dpo_inference_collator", llama_dpo_inference_collator, processors["llama"], samples, modes)
        _run_collator("gemma_dpo_inference_collator", gemma_dpo_inference_collator, processors["gemma"], samples, modes)

    except Exception as exc:
        raise RuntimeError(f"DPO inference collator test failed: {exc}") from exc


if __name__ == "__main__":
    main()
