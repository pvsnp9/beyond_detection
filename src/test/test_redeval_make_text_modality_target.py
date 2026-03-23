import json
from pathlib import Path

import pytest

from src.data_generation.redeval import (
    IMAGE_MISSING_EXPLANATION,
    TEXT_MISSING_EXPLANATION,
    make_text_modality_target,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _base_target() -> dict:
    return {
        "need_explanation": True,
        "visual_facts": [{"id": 1, "fact": "fact one"}],
        "evidence_fact_ids": [1],
        "text_literal": "literal text",
        "incongruity": "mismatch",
        "label": "sarcastic",
        "explanation": "old explanation",
        "missing_modalities": [],
    }


def test_make_text_modality_target_transforms_rows_and_preserves_other_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "ood.jsonl"
    rows = [
        {
            "id": "drop-both",
            "image": "/tmp/drop.jpg",
            "caption": "drop me",
            "label_gt": "sarcastic",
            "modality": "both",
            "target_json": json.dumps(_base_target()),
        },
        {
            "id": "text-1",
            "image": "/tmp/text.jpg",
            "caption": "caption for text modality",
            "label_gt": "non_sarcastic",
            "modality": "text",
            "teacher": '{"x":1}',
            "quality_flags": ["keep_this"],
            "target_json": json.dumps(_base_target()),
        },
        {
            "id": "image-1",
            "image": "/tmp/image.jpg",
            "caption": "caption for image modality",
            "label_gt": "sarcastic",
            "modality": "image",
            "teacher": '{"y":2}',
            "quality_flags": [],
            "target_json": _base_target(),
        },
    ]
    _write_jsonl(input_path, rows)

    stats = make_text_modality_target(input_path=input_path)
    assert stats == {
        "total_in": 3,
        "kept_text": 1,
        "kept_image": 1,
        "kept_both": 1,
        "dropped_both_or_other": 0,
        "written": 3,
    }

    output_rows = _read_jsonl(input_path)
    assert len(output_rows) == 3
    assert {row["modality"] for row in output_rows} == {"text", "image", "both"}

    by_modality = {row["modality"]: row for row in output_rows}

    both_row = by_modality["both"]
    both_target = json.loads(both_row["target_json"])
    assert both_row["label_gt"] == "sarcastic"
    assert both_target["label"] == "sarcastic"
    assert both_target["visual_facts"] == [{"id": 1, "fact": "fact one"}]
    assert both_target["evidence_fact_ids"] == [1]

    text_row = by_modality["text"]
    text_target = json.loads(text_row["target_json"])
    assert text_row["label_gt"] == "unknown"
    assert text_target["visual_facts"] == []
    assert text_target["evidence_fact_ids"] == []
    assert text_target["label"] == "unknown"
    assert text_target["incongruity"] == ""
    assert text_target["explanation"] == TEXT_MISSING_EXPLANATION
    assert text_target["missing_modalities"] == ["image"]
    assert text_row["teacher"] == '{"x":1}'
    assert text_row["quality_flags"] == ["keep_this"]

    image_row = by_modality["image"]
    image_target = json.loads(image_row["target_json"])
    assert image_row["label_gt"] == "unknown"
    assert image_target["label"] == "unknown"
    assert image_target["incongruity"] == ""
    assert image_target["text_literal"] == ""
    assert image_target["explanation"] == IMAGE_MISSING_EXPLANATION
    assert image_target["missing_modalities"] == ["text"]
    assert image_row["teacher"] == '{"y":2}'
    assert image_row["quality_flags"] == []


def test_make_text_modality_target_keeps_original_file_when_target_json_is_malformed(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "ood.jsonl"
    rows = [
        {
            "id": "ok",
            "image": "/tmp/ok.jpg",
            "caption": "ok",
            "label_gt": "non_sarcastic",
            "modality": "text",
            "target_json": json.dumps(_base_target()),
        },
        {
            "id": "bad",
            "image": "/tmp/bad.jpg",
            "caption": "bad",
            "label_gt": "sarcastic",
            "modality": "image",
            "target_json": "{not valid json",
        },
    ]
    _write_jsonl(input_path, rows)
    before = input_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed target_json"):
        make_text_modality_target(input_path=input_path)

    after = input_path.read_text(encoding="utf-8")
    assert after == before
