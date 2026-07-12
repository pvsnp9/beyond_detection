import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REQUIRED_OUTPUT_KEYS = {"label", "explanation", "missing_modalities"}
VALID_LABELS = {"sarcastic", "non_sarcastic", "unknown"}
PATH_PATTERN = re.compile(
    r"outputs/results/(?P<model>[^/]+)/(?P<timestamp>\d{8}_\d{6})/(?P<type>[^/]+)/(?P<run_mode>[^/]+)/[^/]+\.jsonl$"
)


@dataclass
class ParsedRow:
    raw: Dict[str, Any]
    target: Optional[Dict[str, Any]] = None
    output: Optional[Dict[str, Any]] = None
    parse_status: str = "ok"  # ok | json_decode | not_dict | missing_keys | bad_label_value | output_missing


def parse_row(record: Dict[str, Any]) -> ParsedRow:
    row = ParsedRow(raw=record)

    target_raw = record.get("target_json")
    if isinstance(target_raw, str):
        try:
            row.target = json.loads(target_raw)
        except json.JSONDecodeError:
            row.target = None
    elif isinstance(target_raw, dict):
        row.target = target_raw

    output_raw = record.get("output")
    if output_raw is None:
        row.parse_status = "output_missing"
        return row

    if isinstance(output_raw, dict):
        parsed = output_raw
    elif isinstance(output_raw, str):
        try:
            parsed = json.loads(output_raw)
        except json.JSONDecodeError:
            row.parse_status = "json_decode"
            return row
    else:
        row.parse_status = "json_decode"
        return row

    if not isinstance(parsed, dict):
        row.parse_status = "not_dict"
        return row

    missing = REQUIRED_OUTPUT_KEYS - set(parsed.keys())
    if missing:
        row.parse_status = "missing_keys"
        row.output = parsed
        return row

    label = parsed.get("label")
    if not isinstance(label, str) or label not in VALID_LABELS:
        row.parse_status = "bad_label_value"
        row.output = parsed
        return row

    row.output = parsed
    return row


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"__line_decode_failed__": True, "raw": line})
    return records


def parse_path_triple(jsonl_path: Path) -> Optional[Dict[str, str]]:
    text = str(jsonl_path.resolve())
    m = PATH_PATTERN.search(text)
    if not m:
        return None
    return m.groupdict()
