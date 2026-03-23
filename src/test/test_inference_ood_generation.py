import json
from pathlib import Path

from PIL import Image

from src.inference import run_inference as ri


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_load_redeval_ood_rows_converts_image_modalities(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)

    input_path = tmp_path / "ood.jsonl"
    rows = [
        {
            "id": "a",
            "modality": "both",
            "label_gt": "sarcastic",
            "image": str(image_path),
            "caption": "caption",
        },
        {
            "id": "b",
            "modality": "text",
            "label_gt": "non_sarcastic",
            "caption": "text only",
        },
    ]
    with input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    loaded, stats = ri._load_redeval_ood_rows(str(input_path))
    assert len(loaded) == 2
    assert isinstance(loaded[0]["image"], str)
    assert loaded[0]["image"] == str(image_path)
    assert loaded[1]["modality"] == "text"
    assert stats["data_load_failures"] == 0
    assert stats["image_load_failures"] == 0


def test_load_redeval_ood_rows_invalid_json_is_skipped(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.jsonl"
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image_path)
    bad_path.write_text(
        '\n'.join(
            [
                '{"id":"x","modality":"both","label_gt":"sarcastic","image":"%s"}' % str(image_path),
                '{bad json}',
            ]
        )
        + '\n',
        encoding="utf-8",
    )

    loaded, stats = ri._load_redeval_ood_rows(str(bad_path))
    assert len(loaded) == 1
    assert stats["data_load_failures"] == 1


def _patch_ood_generation_dependencies(
    monkeypatch,
    tmp_path: Path,
    ood_rows: list[dict],
    ood_perturb_rows: list[dict],
) -> None:
    class DummyTokenizer:
        pad_token = None
        eos_token = "</s>"
        padding_side = "right"

    class DummyProcessor:
        def __init__(self):
            self.tokenizer = DummyTokenizer()

    class DummyModel:
        def eval(self):
            return self

    class DummyLogistics:
        def __init__(self):
            self.project_root_dir = str(tmp_path)
            self.processed_data_dir = "data/processed"
            self.infer_bathc_size = 2
            self.gen_max_token = 32

    def fake_load_rows(input_path: str):
        if input_path.endswith("ood_perturb.jsonl"):
            return [dict(row) for row in ood_perturb_rows], {
                "data_load_failures": 0,
                "image_load_failures": 0,
            }
        if input_path.endswith("ood.jsonl"):
            return [dict(row) for row in ood_rows], {
                "data_load_failures": 0,
                "image_load_failures": 0,
            }
        raise AssertionError(f"Unexpected input path: {input_path}")

    def fake_prepare_results_dir(results_dir: str) -> str:
        task_dir = Path(results_dir) / "20260101_000001"
        task_dir.mkdir(parents=True, exist_ok=True)
        return str(task_dir)

    monkeypatch.setattr(ri, "Logistics", DummyLogistics)
    monkeypatch.setattr(ri, "_load_redeval_ood_rows", fake_load_rows)
    monkeypatch.setattr(ri, "_load_processor_for_inference", lambda **kwargs: DummyProcessor())
    monkeypatch.setattr(ri, "_load_model_for_inference", lambda **kwargs: DummyModel())
    monkeypatch.setattr(ri, "_build_model_inputs", lambda **kwargs: kwargs["batch_data"])
    monkeypatch.setattr(
        ri,
        "generate_batch",
        lambda model, model_inputs, processor, max_tokens=256: ['{"label":"sarcastic"}']
        * len(model_inputs),
    )
    monkeypatch.setattr(
        ri,
        "get_sft_result_dir",
        lambda model_key: str(tmp_path / "outputs" / "results" / model_key),
    )
    monkeypatch.setattr(ri, "_prepare_results_dir", fake_prepare_results_dir)


def test_ood_generation_writes_two_all_jsonl_outputs(tmp_path: Path, monkeypatch) -> None:
    ood_rows = [
        {
            "id": "ood-1",
            "modality": "text",
            "label_gt": "sarcastic",
            "query": "Q",
            "target_json": '{"label":"sarcastic"}',
            "quality_flags": [],
            "source": "redeval",
        }
    ]
    ood_perturb_rows = [
        {
            "id": "pert-1",
            "modality": "text",
            "label_gt": "non_sarcastic",
            "query": "Q",
            "target_json": '{"label":"non_sarcastic"}',
            "quality_flags": [],
            "source": "redeval",
            "perturbation_tpye": "image",
        }
    ]
    _patch_ood_generation_dependencies(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        ood_rows=ood_rows,
        ood_perturb_rows=ood_perturb_rows,
    )

    outputs = ri.ood_generation(
        model_key="aya",
        checkpoint_path="/tmp/checkpoint",
        model_type="sft",
        max_tokens=16,
        merge_adapter=False,
    )

    assert set(outputs.keys()) == {"ood", "ood_perturb"}
    ood_path = Path(outputs["ood"]["all"])
    perturb_path = Path(outputs["ood_perturb"]["all"])
    assert ood_path.name == "all.jsonl"
    assert perturb_path.name == "all_perturb.jsonl"
    assert ood_path.parent == perturb_path.parent
    assert ood_path.exists()
    assert perturb_path.exists()


def test_ood_generation_propagates_perturbation_and_schema(
    tmp_path: Path, monkeypatch
) -> None:
    ood_rows = [
        {
            "id": "ood-2",
            "modality": "text",
            "label_gt": "sarcastic",
            "query": "Q",
            "target_json": '{"label":"sarcastic"}',
            "quality_flags": [],
            "source": "redeval",
        }
    ]
    ood_perturb_rows = [
        {
            "id": "pert-2",
            "modality": "text",
            "label_gt": "non_sarcastic",
            "query": "Q",
            "target_json": '{"label":"non_sarcastic"}',
            "quality_flags": [],
            "source": "redeval",
            "perturbation_tpye": "text",
        }
    ]
    _patch_ood_generation_dependencies(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        ood_rows=ood_rows,
        ood_perturb_rows=ood_perturb_rows,
    )

    outputs = ri.ood_generation(
        model_key="aya",
        checkpoint_path="/tmp/checkpoint",
        model_type="sft",
        max_tokens=16,
        merge_adapter=False,
    )
    ood_records = _read_jsonl(Path(outputs["ood"]["all"]))
    perturb_records = _read_jsonl(Path(outputs["ood_perturb"]["all"]))
    assert "perturbation_tpye" not in ood_records[0]
    assert perturb_records[0]["perturbation_tpye"] == "text"
    for record in ood_records + perturb_records:
        assert "mode" in record
        assert "target_json" in record
        assert "output" in record
