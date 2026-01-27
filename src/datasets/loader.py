"""HF dataset loader helpers."""

from typing import Optional, Union

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset

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
) -> DatasetLike:
    """Load a Hugging Face dataset without altering its schema."""
    return load_dataset(
        dataset_id,
        name=config_name,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
        download_mode=download_mode,
    )
