"""
Verify whether test data IDs are leaked into train/validation for each language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Optional

from config.logistics import Logistics

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Skipping invalid JSON in {path} at line {line_num}: {exc}")
    except FileNotFoundError:
        print(f"Missing file: {path}")
    except Exception as exc:
        print(f"Failed to read {path}: {exc}")


def _build_id_set(records: Iterable[Dict[str, Any]], source: str) -> set:
    ids = set()
    for rec in records:
        rec_id = rec.get("id")
        if rec_id is None:
            print(f"Skipping record without id in {source}: {rec}")
            continue
        ids.add(rec_id)
    return ids


def _collect_test_records(records: Iterable[Dict[str, Any]]) -> Iterable[Tuple[Any, Dict[str, Any]]]:
    for rec in records:
        rec_id = rec.get("id")
        if rec_id is None:
            print(f"Skipping test record without id: {rec}")
            continue
        yield rec_id, rec


def verify_language(lang: str, logistics: Logistics, dataset_kind: str = "sft") -> None:
    try:
        base_dir = (
            Path(logistics.project_root_dir)
            / logistics.processed_data_dir
            / dataset_kind
            / lang
        )
        train_path = base_dir / "train.jsonl"
        val_path = base_dir / "validation.jsonl"
        test_path = base_dir / "test.jsonl"

        train_ids = _build_id_set(_load_jsonl(train_path), "train")
        val_ids = _build_id_set(_load_jsonl(val_path), "validation")

        output_dir = (
            Path(logistics.project_root_dir)
            / logistics.processed_data_dir
            / "verification"
            / dataset_kind
            / lang
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "leaked_test_data.jsonl"

        leaked_records = []
        for rec_id, rec in _collect_test_records(_load_jsonl(test_path)):
            leaked_in = []
            if rec_id in train_ids:
                leaked_in.append("train")
            if rec_id in val_ids:
                leaked_in.append("validation")
            if leaked_in:
                leaked_records.append(
                    {
                        "source": "test",
                        "id": rec_id,
                        "leaked_in": leaked_in,
                        "content": rec,
                    }
                )

        if leaked_records:
            with output_path.open("w", encoding="utf-8") as handle:
                for record in leaked_records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(record)
        else:
            print("No data leakage found")
    except Exception as exc:
        print(f"Verification failed for {dataset_kind} lang={lang}: {exc}")


def _get_hf_config(dataset_id: str, lang: Optional[str]) -> Optional[str]:
    if dataset_id == "coderchen01/MMSD2.0":
        return "mmsd-v2"
    if dataset_id == "alita9/muse-sarcasm-explanation":
        return None
    if dataset_id == "alita9/sarcnet":
        return lang
    return None


# checks original HS sets
def check_hf_split_overlaps(logistics: Logistics) -> None:
    if load_dataset is None:
        print("Skipping HF overlap check: datasets package is not installed.")
        return

    splits = ["train", "validation", "test"]
    for name, dataset_id in logistics.hf_datatset_ids.items():
        lang_list = logistics.langs if dataset_id == "alita9/sarcnet" else [None]
        for lang in lang_list:
            config_name = _get_hf_config(dataset_id, lang)
            try:
                ids_by_split = {}
                for split in splits:
                    ds = load_dataset(dataset_id, config_name, split=split)
                    ids_by_split[split] = {
                        str(row.get("id")) for row in ds if row.get("id") is not None
                    }

                train_ids = ids_by_split["train"]
                val_ids = ids_by_split["validation"]
                test_ids = ids_by_split["test"]
                print(
                    f"HF overlap check: {name} ({dataset_id}) "
                    f"config={config_name}"
                )
                print(f"  train/test overlap: {len(train_ids & test_ids)}")
                print(f"  train/val overlap: {len(train_ids & val_ids)}")
                print(f"  val/test overlap: {len(val_ids & test_ids)}")
            except Exception as exc:
                print(
                    f"Failed HF overlap check for {dataset_id} "
                    f"config={config_name}: {exc}"
                )


def main() -> None:
    logistics = Logistics()
    for lang in logistics.langs:
        verify_language(lang, logistics, "sft")
        # verify_language(lang, logistics, "dpo")
    # check_hf_split_overlaps(logistics)


# if __name__ == "__main__":
#     main()
