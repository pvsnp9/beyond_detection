from transformers import AutoProcessor

from config.logistics import Logistics, ModelCards
from src.datasets.loader import load_dpo_train_eval_dataset


def _select_samples(dataset, max_samples: int) -> list[dict]:
    if hasattr(dataset, "select"):
        return list(dataset.select(range(max_samples)))
    if hasattr(dataset, "take"):
        return list(dataset.take(max_samples))
    return list(dataset)[:max_samples]


def _print_example_info(split_name: str, example: dict, idx: int) -> None:
    required_keys = ["prompt", "images", "chosen", "rejected"]
    missing = [key for key in required_keys if key not in example]
    assert not missing, f"Missing keys in {split_name}[{idx}]: {missing}"

    print(f"\n{split_name}[{idx}] key types:")
    for key,value in example.items():
        value_type = type(value)
        if key == "images" and isinstance(value, list) and value:
            print(f"- {key}: {value_type} item_type={type(value[0])}")
        else:
            print(f"- {key}: {value_type}")

    print(f"{split_name}[{idx}] data:")
    print(f"Prompt:{example.get('prompt', 'No')}\nChosen:{example.get('chosen', 'No')}\nRejected:{example.get('rejected', 'No')}\nImages:{example.get('images', 'No')}")
    print(example)

def main() -> None:
    logistics = Logistics()
    mc = ModelCards()

    processor = AutoProcessor.from_pretrained(
        mc.qwen3_vl_8b_instruct,
        trust_remote_code=True,
    )
    print("Loaded processor:", mc.qwen3_vl_8b_instruct)

    dataset = load_dpo_train_eval_dataset(
        config_name="en",
        processor=processor,
        cache_dir=logistics.hf_cache_dir,
    )

    train_samples = _select_samples(dataset["train"], 2)
    eval_samples = _select_samples(dataset["eval"], 2)

    print(f"Trains cols: {train_samples[0].keys()}\nEval cols:{eval_samples[0].keys()}")
    print(f"train_samples={len(train_samples)}")
    print(f"eval_samples={len(eval_samples)}")

    for idx, sample in enumerate(train_samples):
        _print_example_info("train", sample, idx)
    for idx, sample in enumerate(eval_samples):
        _print_example_info("eval", sample, idx)


if __name__ == "__main__":
    main()
