import json
from pathlib import Path

import pytest

from src.data_generation.redeval_oai_gen import (
    build_final_record,
    build_target_json,
    process_split,
    validate_teacher_json,
)
from src.data_generation.sarcasm_curation_prompt import SarcasmOODGen


def _teacher_payload(label: str = "sarcastic") -> dict:
    return {
        "schema_version": "sarcasm_ood_gen_v1",
        "language": "en",
        "label": label,
        "visual_facts": ["fact 1", "fact 2", "fact 3", "fact 4"],
        "text_literal": "literal text",
        "incongruity": "some mismatch",
        "explanation": "reasoned explanation",
        "quality_flags": ["ambiguous_sarcasm"],
    }


def _read_lines(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_sarcasm_oodgen_v1_is_instantiable_and_schema_complete() -> None:
    spec = SarcasmOODGen.v1()
    assert hasattr(spec.prompts, "system")
    assert callable(spec.prompts.build_user_caption)
    assert "CAPTION:" in spec.prompts.build_user_caption("hello")
    required = spec.json_schema["schema"]["required"]
    assert "quality_flags" in required
    assert spec.json_schema["schema"]["properties"]["label"]["enum"] == [
        "sarcastic",
        "non_sarcastic",
    ]


def test_validate_teacher_json_strict_checks() -> None:
    valid = _teacher_payload()
    validate_teacher_json(valid)

    missing_quality = dict(valid)
    missing_quality.pop("quality_flags")
    with pytest.raises(ValueError):
        validate_teacher_json(missing_quality)

    bad_label = dict(valid)
    bad_label["label"] = "unknown"
    with pytest.raises(ValueError):
        validate_teacher_json(bad_label)


def test_build_target_json_order_and_mapping() -> None:
    teacher = _teacher_payload()
    target = build_target_json(label_gt="non_sarcastic", modality="image", teacher=teacher)
    assert list(target.keys()) == [
        "need_explanation",
        "visual_facts",
        "evidence_fact_ids",
        "text_literal",
        "incongruity",
        "label",
        "explanation",
        "missing_modalities",
    ]
    assert target["label"] == "non_sarcastic"
    assert target["missing_modalities"] == ["text"]
    assert target["evidence_fact_ids"] == [1, 2, 3]
    assert target["visual_facts"][0] == {"id": 1, "fact": "fact 1"}


def test_build_final_record_contains_required_fields() -> None:
    row = {
        "id": "abc",
        "image": "/tmp/x.jpg",
        "caption": "caption",
        "label_gt": "sarcastic",
        "modality": "both",
    }
    teacher = _teacher_payload()
    target = build_target_json(label_gt="sarcastic", modality="both", teacher=teacher)
    out = build_final_record(row=row, teacher=teacher, target_json=target, query="QUERY")
    assert out["source"] == "redeval"
    assert out["language"] == "en"
    assert out["query"] == "QUERY"
    assert out["quality_flags"] == teacher["quality_flags"]
    assert out["visual_facts"] == teacher["visual_facts"]
    assert json.loads(out["teacher"])["label"] == "sarcastic"
    assert json.loads(out["target_json"])["label"] == "sarcastic"


def test_process_split_success_and_failure_logs(tmp_path: Path) -> None:
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.write_bytes(b"fakejpeg-a")
    image_b.write_bytes(b"fakejpeg-b")

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    failed_path = tmp_path / "failed.jsonl"

    rows = [
        {
            "id": "1",
            "image": str(image_a),
            "caption": "caption one",
            "label_gt": "sarcastic",
            "modality": "both",
        },
        {
            "id": "2",
            "image": str(image_b),
            "caption": "caption two",
            "label_gt": "non_sarcastic",
            "modality": "text",
        },
    ]
    with input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    def fake_teacher(image_url: str, caption: str) -> dict:
        assert image_url.startswith("data:image/")
        if "two" in caption:
            raise RuntimeError("forced failure")
        return _teacher_payload(label="non_sarcastic")

    stats = process_split(
        split_name="ood",
        input_path=input_path,
        output_path=output_path,
        failed_path=failed_path,
        teacher_generator=fake_teacher,
        query="DPO_QUERY",
    )

    assert stats["total"] == 2
    assert stats["success"] == 1
    assert stats["failed"] == 1
    assert stats["both_label_mismatch"] == 1

    output_rows = _read_lines(output_path)
    failed_rows = _read_lines(failed_path)
    assert len(output_rows) == 1
    assert len(failed_rows) == 1
    assert failed_rows[0]["id"] == "2"

    generated = output_rows[0]
    target = json.loads(generated["target_json"])
    assert generated["source"] == "redeval"
    assert generated["query"] == "DPO_QUERY"
    assert target["label"] == "sarcastic"
