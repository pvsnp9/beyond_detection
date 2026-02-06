"""
Clean combined SFT/DPO datasets by removing cross-split overlaps.

Policy:
- Keep any ID present in test; remove from train/validation.
- If an ID is in both train and validation, keep it in train and drop from validation.
Writes cleaned files to data/generated/combined_clean/<kind>/<lang>/<split>.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Any, Tuple

from config.logistics import Logistics


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"Skipping invalid JSON in {path} at line {line_num}: {exc}")
    except Exception as exc:
        print(f"Failed to read {path}: {exc}")
    return records


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _get_id(rec: Dict[str, Any]) -> Any:
    return rec.get("id")


def _dedupe_splits(
    train: List[Dict[str, Any]],
    val: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    test_ids = {i for i in (_get_id(r) for r in test) if i is not None}
    train_ids = {i for i in (_get_id(r) for r in train) if i is not None}
    val_ids = {i for i in (_get_id(r) for r in val) if i is not None}

    drop_train = {i for i in train_ids if i in test_ids}
    drop_val = {i for i in val_ids if i in test_ids or i in train_ids}

    cleaned_train = [r for r in train if _get_id(r) not in drop_train]
    cleaned_val = [r for r in val if _get_id(r) not in drop_val]

    stats = {
        "drop_train": len(drop_train),
        "drop_val": len(drop_val),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "train_clean": len(cleaned_train),
        "val_clean": len(cleaned_val),
    }
    return cleaned_train, cleaned_val, test, stats


def clean_combined_splits() -> None:
    logistics = Logistics()
    src_root = Path(logistics.project_root_dir) / logistics.combined_data_dir
    dst_root = Path(logistics.project_root_dir) / logistics.data_generation_dir / "combined_clean"

    if not src_root.exists():
        print(f"Missing combined dir: {src_root}")
        return

    for kind in ["sft", "dpo"]:
        for lang in logistics.langs:
            in_dir = src_root / kind / lang
            if not in_dir.exists():
                continue

            train = _read_jsonl(in_dir / "train.jsonl")
            val = _read_jsonl(in_dir / "validation.jsonl")
            test = _read_jsonl(in_dir / "test.jsonl")

            cleaned_train, cleaned_val, cleaned_test, stats = _dedupe_splits(
                train, val, test
            )

            out_dir = dst_root / kind / lang
            _write_jsonl(out_dir / "train.jsonl", cleaned_train)
            _write_jsonl(out_dir / "validation.jsonl", cleaned_val)
            _write_jsonl(out_dir / "test.jsonl", cleaned_test)

            print(
                f"{kind}/{lang} "
                f"train {stats['train_count']}->{stats['train_clean']} "
                f"val {stats['val_count']}->{stats['val_clean']} "
                f"test {stats['test_count']} "
                f"dropped train={stats['drop_train']} val={stats['drop_val']}"
            )


def main() -> None:
    clean_combined_splits()


if __name__ == "__main__":
    main()
