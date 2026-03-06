"""HF dataset loader helpers."""

from math import floor
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
    eval_size: int = 48,
    format_data: bool = False,
    allowed_modalities: Optional[tuple[str, ...]] = ("both",),
) -> DatasetDict:
    """Load DPO train dataset, optionally filter modalities, derive eval split, and map if requested."""
    if format_data and processor is None:
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

    normalized_modalities = None
    if allowed_modalities:
        normalized_modalities = {
            str(modality).strip().lower()
            for modality in allowed_modalities
            if str(modality).strip()
        }
        if not normalized_modalities:
            normalized_modalities = None

    if normalized_modalities:
        if isinstance(dataset, IterableDataset):
            dataset = dataset.filter(
                lambda example: str(example.get("modality", "")).strip().lower() in normalized_modalities
            )
        elif "modality" in dataset.column_names:
            dataset = dataset.filter(
                lambda example: str(example.get("modality", "")).strip().lower() in normalized_modalities,
                load_from_cache_file=False,
            )

    if isinstance(dataset, IterableDataset):
        actual_eval_size = max(0, eval_size)
        eval_dataset = dataset.take(actual_eval_size)
        train_dataset = dataset.skip(actual_eval_size)
    else:
        dataset_len = len(dataset)
        actual_eval_size = min(max(0, eval_size), dataset_len)
        if actual_eval_size == 0:
            eval_dataset = dataset.select([])
            train_dataset = dataset
        elif "modality" not in dataset.column_names:
            eval_dataset = dataset.select(range(actual_eval_size))
            train_dataset = dataset.select(range(actual_eval_size, dataset_len))
        else:
            modalities = dataset["modality"]
            bucket_to_indices = {"both": [], "text": [], "image": []}
            for idx, modality in enumerate(modalities):
                normalized_modality = str(modality or "").strip().lower()
                if normalized_modality in bucket_to_indices:
                    bucket_to_indices[normalized_modality].append(idx)

            ratio_by_modality = {"both": 0.8, "text": 0.1, "image": 0.1}
            target_counts = {
                key: floor(actual_eval_size * ratio_by_modality[key]) for key in ratio_by_modality
            }
            assigned = sum(target_counts.values())
            if assigned < actual_eval_size:
                # Keep remainder in "both" so small eval sizes still prioritize multimodal samples.
                target_counts["both"] += actual_eval_size - assigned

            eval_indices = []
            for modality in ("both", "text", "image"):
                eval_indices.extend(bucket_to_indices[modality][: target_counts[modality]])

            if len(eval_indices) < actual_eval_size:
                selected_set = set(eval_indices)
                for idx in range(dataset_len):
                    if idx in selected_set:
                        continue
                    eval_indices.append(idx)
                    selected_set.add(idx)
                    if len(eval_indices) == actual_eval_size:
                        break

            eval_index_set = set(eval_indices)
            train_indices = [idx for idx in range(dataset_len) if idx not in eval_index_set]
            eval_dataset = dataset.select(eval_indices)
            train_dataset = dataset.select(train_indices)

    if format_data:
        train_dataset = train_dataset.map(
            lambda example: format_hf_dpo_data(example, processor),
            load_from_cache_file=False,
            # remove_columns=train_dataset.column_names,
        )
        eval_dataset = eval_dataset.map(
            lambda example: format_hf_dpo_data(example, processor),
            load_from_cache_file=False,
            # remove_columns=eval_dataset.column_names,
        )
    return DatasetDict({"train": train_dataset, "eval": eval_dataset})
