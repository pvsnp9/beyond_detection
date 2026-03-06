import argparse

from transformers import AutoProcessor

from config.logistics import Logistics, ModelCards
from src.datasets.loader import load_dpo_train_eval_dataset


def _select_samples(dataset, max_samples: int) -> list[dict]:
    if hasattr(dataset, "select"):
        return list(dataset.select(range(max_samples)))
    if hasattr(dataset, "take"):
        return list(dataset.take(max_samples))
    return list(dataset)[:max_samples]


def _print_example_info(split_name: str, example: dict, idx: int, format_data: bool) -> None:
    required_keys = ["query", "image", "chosen", "rejected"]
    if format_data:
        required_keys.extend(["prompt", "images"])
    missing = [key for key in required_keys if key not in example]
    assert not missing, f"Missing keys in {split_name}[{idx}]: {missing}"

    print(f"\n{split_name}[{idx}] key types:")
    for key, value in example.items():
        value_type = type(value)
        print(f"- {key}: {value_type}")

    print(f"{split_name}[{idx}] data:")
    print(
        f"Query:{example.get('query', 'No')}\n"
        f"Chosen:{example.get('chosen', 'No')}\n"
        f"Rejected:{example.get('rejected', 'No')}\n"
        f"Image:{example.get('image', 'No')}"
    )
    print(example)

def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect DPO train/eval samples.")
    parser.add_argument(
        "--format-data",
        action="store_true",
        help="Apply dataset formatting (expects `prompt` and `images` fields).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=2,
        help="Number of samples to print from each split.",
    )
    return parser.parse_args()


def main() -> None:
    args = _build_args()
    logistics = Logistics()
    processor = None
    if args.format_data:
        model_cards = ModelCards()
        processor = AutoProcessor.from_pretrained(
            model_cards.qwen3_vl_8b_instruct,
            trust_remote_code=True,
        )
        print("Loaded processor:", model_cards.qwen3_vl_8b_instruct)

    dataset = load_dpo_train_eval_dataset(
        config_name="en",
        processor=processor,
        cache_dir=logistics.hf_cache_dir,
        format_data=args.format_data,
    )

    train_samples = _select_samples(dataset["train"], args.sample_count)
    eval_samples = _select_samples(dataset["eval"], args.sample_count)

    print(f"Trains cols: {train_samples[0].keys()}\nEval cols:{eval_samples[0].keys()}")
    print(f"train_samples={len(train_samples)}")
    print(f"eval_samples={len(eval_samples)}")

    for idx, sample in enumerate(train_samples):
        _print_example_info("train", sample, idx, args.format_data)
    for idx, sample in enumerate(eval_samples):
        _print_example_info("eval", sample, idx, args.format_data)


if __name__ == "__main__":
    main()
