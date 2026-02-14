"""HF dataset loader helpers."""

from typing import Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset

from config.logistics import Logistics

# Convenient return type alias for the datasets loader.
DatasetLike = Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]


def load_hf_dataset(
    dataset_id: str,
    *,
    split: Optional[str] = "train",
    config_name: Optional[str] = None,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    download_mode: Optional[str] = None,
    mode: Optional[str] = None,
) -> DatasetLike:
    """Load a Hugging Face dataset without altering its schema."""
    dataset = load_dataset(
        dataset_id,
        name=config_name,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
        download_mode=download_mode,
    )
    if mode == "rationale_sft":
        allowed_tasks = {"explanation", "detection_explanation"}

        def _keep_example(example):
            task = example.get("task") if isinstance(example, dict) else None
            return task in allowed_tasks

        if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
            dataset = type(dataset)(
                {split_name: split_ds.filter(_keep_example) for split_name, split_ds in dataset.items()}
            )
        else:
            dataset = dataset.filter(_keep_example)
    try:
        dataset_len = len(dataset)
    except TypeError:
        dataset_len = None
    if dataset_len is not None:
        print(f"Dataset size: {dataset_len}")
    return dataset


def load_hf_dpo_dataset(
    *,
    split: Optional[str] = "train",
    config_name: Optional[str] = None,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    download_mode: Optional[str] = None,
) -> DatasetLike:
    """Load the default HF DPO dataset without altering its schema."""
    logistics = Logistics()
    return load_hf_dataset(
        logistics.hf_dpo_ds_id,
        split=split,
        config_name=config_name,
        streaming=streaming,
        cache_dir=cache_dir,
        download_mode=download_mode,
    )
