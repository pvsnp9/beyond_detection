"""
Visual verification for processed SFT/DPO datasets before HF publish.
Prints split counts and a few sample rows per split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Iterable

from config.logistics import Logistics


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _print_split_summary(base_dir: Path, split: str, sample_size: int) -> None:
    path = base_dir / f"{split}.jsonl"
    if not path.exists():
        print(f"{path} missing")
        return
    count = 0
    samples = []
    for rec in _read_jsonl(path):
        count += 1
        if len(samples) < sample_size:
            samples.append(rec)
    print(f"{path} count={count}")
    for idx, rec in enumerate(samples, start=1):
        print(f"sample[{idx}]: {json.dumps(rec, ensure_ascii=False)}")


def verify_processed(kind: str, sample_size: int = 2) -> None:
    logistics = Logistics()
    for lang in logistics.langs:
        base_dir = Path(logistics.project_root_dir) / logistics.processed_data_dir / kind / lang
        print(f"================== {kind} {lang} ==================\n")
        for split in logistics.splits:
            _print_split_summary(base_dir, split, sample_size)


def main() -> None:
    verify_processed("sft")
    verify_processed("dpo")


if __name__ == "__main__":
    main()
