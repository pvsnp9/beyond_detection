import json
import os
import sys
from pathlib import Path

from config.logistics import Logistics


def read_sample_jsonl(path: Path, max_rows: int = 5) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= max_rows:
                break
    return rows


def main() -> int:
    logistics = Logistics()
    input_path = (
        Path(logistics.project_root_dir)
        / logistics.processed_data_dir
        / "dpo"
        / "en"
        / "train.jsonl"
    )
    output_path = (
        Path(logistics.project_root_dir)
        / logistics.processed_data_dir
        / "dpo"
        / "en"
        / "complete_train.jsonl"
    )

    if not input_path.exists():
        print(f"SMOKE FAIL: input missing: {input_path}", file=sys.stderr)
        return 1
    if not input_path.is_file():
        print(f"SMOKE FAIL: input not a file: {input_path}", file=sys.stderr)
        return 1
    if not output_path.parent.exists():
        print(f"SMOKE FAIL: output dir missing: {output_path.parent}", file=sys.stderr)
        return 1
    if not os.access(output_path.parent, os.W_OK):
        print(f"SMOKE FAIL: output dir not writable: {output_path.parent}", file=sys.stderr)
        return 1

    rows = read_sample_jsonl(input_path)
    if not rows:
        print(f"SMOKE FAIL: no rows found in {input_path}", file=sys.stderr)
        return 1

    if not all("id" in r for r in rows):
        print("SMOKE FAIL: some sample rows missing id field", file=sys.stderr)
        return 1
    if not any(isinstance(r.get("chosen"), str) and r.get("chosen").strip() for r in rows):
        print("SMOKE FAIL: no sample rows with non-empty chosen text", file=sys.stderr)
        return 1

    print("SMOKE OK: gen_rejected inputs/paths look valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
