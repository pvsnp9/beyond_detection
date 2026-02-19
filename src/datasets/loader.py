"""HF dataset loader helpers."""

from typing import Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset

from config.logistics import Logistics
from src.collators.dpo_data import format_hf_dpo_data

# Convenient return type alias for the datasets loader.
DatasetLike = Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]


def load_hf_dataset(
    dataset_id: str,
    split: Optional[str] = "train",
    config_name: Optional[str] = None,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    download_mode: Optional[str] = None,
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
    try:
        dataset_len = len(dataset)
    except TypeError:
        dataset_len = None
    if dataset_len is not None:
        print(f"Actual [{split}] Dataset size: {dataset_len}")
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


def load_dpo_train_eval_dataset(
    config_name: Optional[str] = None,
    processor: object = None,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    download_mode: Optional[str] = None,
    eval_size: int = 67,
) -> DatasetDict:
    """Load DPO train dataset, derive eval split, and map into model format."""
    if processor is None:
        raise ValueError("processor is required")
    logistics = Logistics()
    dataset = load_hf_dataset(
        logistics.hf_dpo_ds_id,
        split="train",
        config_name=config_name,
        streaming=streaming,
        cache_dir=cache_dir,
        download_mode=download_mode,
    )

    if isinstance(dataset, IterableDataset):
        eval_dataset = dataset.take(eval_size)
        train_dataset = dataset.skip(eval_size)
    else:
        dataset_len = len(dataset)
        actual_eval_size = min(eval_size, dataset_len)
        eval_dataset = dataset.select(range(actual_eval_size))
        train_dataset = dataset.select(range(actual_eval_size, dataset_len))

    train_dataset = train_dataset.map(
        lambda example: format_hf_dpo_data(example, processor),
        # load_from_cache_file=False
        # remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda example: format_hf_dpo_data(example, processor),
        # load_from_cache_file=False
        # remove_columns=eval_dataset.column_names,
    )
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})
