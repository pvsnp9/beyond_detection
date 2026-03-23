from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from openai import APIStatusError, OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

try:
    from config.logistics import Logistics
    from config.queries import Queries
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from config.logistics import Logistics
    from config.queries import Queries

try:
    from .sarcasm_curation_prompt import SarcasmOODGen
except ImportError:
    from src.data_generation.sarcasm_curation_prompt import SarcasmOODGen


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_TEACHER_KEYS = (
    "schema_version",
    "language",
    "label",
    "visual_facts",
    "text_literal",
    "incongruity",
    "explanation",
    "quality_flags",
)
VALID_LABELS = {"sarcastic", "non_sarcastic"}
VALID_MODALITIES = {"both", "image", "text"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def image_path_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10), reraise=True)
def completion_with_backoff(client: OpenAI, **kwargs: Any) -> Any:
    return client.responses.create(**kwargs)


def request_teacher_annotation(
    client: OpenAI,
    spec: SarcasmOODGen,
    model: str,
    image_url: str,
    caption: str,
) -> Dict[str, Any]:
    try:
        response = completion_with_backoff(
            client=client,
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": spec.prompts.system}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_url},
                        {"type": "input_text", "text": spec.prompts.build_user_caption(caption)},
                    ],
                },
            ],
            text={"format": spec.json_schema},
            temperature=0,
        )
        payload = json.loads(response.output_text)
        if not isinstance(payload, dict):
            raise ValueError("Teacher output is not a JSON object")
        return payload
    except APIStatusError as exc:
        status = getattr(exc, "status_code", None)
        body = None
        if hasattr(exc, "response") and exc.response is not None:
            try:
                body = exc.response.text
            except Exception:
                body = str(exc)
        logger.error("OpenAI APIStatusError status=%s body=%s", status, body)
        raise RuntimeError("OpenAI request failed") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Teacher output is not valid JSON") from exc


def _normalize_modality(modality: Any) -> str:
    value = str(modality or "").strip().lower()
    if value not in VALID_MODALITIES:
        raise ValueError(f"Unsupported modality: {modality}")
    return value


def validate_teacher_json(teacher: Dict[str, Any]) -> None:
    if not isinstance(teacher, dict):
        raise ValueError("Teacher output must be a dict")

    missing = [key for key in REQUIRED_TEACHER_KEYS if key not in teacher]
    if missing:
        raise ValueError(f"Missing required teacher keys: {missing}")

    if not isinstance(teacher["schema_version"], str) or not teacher["schema_version"].strip():
        raise ValueError("schema_version must be a non-empty string")
    if not isinstance(teacher["language"], str):
        raise ValueError("language must be a string")
    if teacher["label"] not in VALID_LABELS:
        raise ValueError(f"label must be one of {sorted(VALID_LABELS)}")
    if not isinstance(teacher["visual_facts"], list) or not teacher["visual_facts"]:
        raise ValueError("visual_facts must be a non-empty list[str]")
    if not all(isinstance(x, str) and x.strip() for x in teacher["visual_facts"]):
        raise ValueError("visual_facts must contain non-empty strings")
    if not isinstance(teacher["text_literal"], str):
        raise ValueError("text_literal must be a string")
    if not isinstance(teacher["incongruity"], str):
        raise ValueError("incongruity must be a string")
    if not isinstance(teacher["explanation"], str):
        raise ValueError("explanation must be a string")
    if not isinstance(teacher["quality_flags"], list):
        raise ValueError("quality_flags must be a list[str]")
    if not all(isinstance(flag, str) for flag in teacher["quality_flags"]):
        raise ValueError("quality_flags must contain only strings")


def _validate_row(row: Dict[str, Any]) -> tuple[str, str, str, str, str]:
    if not isinstance(row, dict):
        raise ValueError("Each row must be a dict")

    row_id = str(row.get("id", "")).strip()
    image_path = str(row.get("image", "")).strip()
    caption = str(row.get("caption", "")).strip()
    label_gt = str(row.get("label_gt", "")).strip()
    modality = _normalize_modality(row.get("modality", "both"))

    if not row_id:
        raise ValueError("Missing id")
    if not image_path:
        raise ValueError("Missing image path")
    if not caption:
        raise ValueError("Missing caption")
    if label_gt not in VALID_LABELS:
        raise ValueError(f"Invalid label_gt: {label_gt}")

    return row_id, image_path, caption, label_gt, modality


def make_visual_fact_objects(teacher_visual_facts: List[str]) -> List[Dict[str, Any]]:
    return [{"id": idx, "fact": fact} for idx, fact in enumerate(teacher_visual_facts, start=1)]


def _missing_modalities(modality: str) -> List[str]:
    if modality == "image":
        return ["text"]
    if modality == "text":
        return ["image"]
    return []


def build_target_json(label_gt: str, modality: str, teacher: Dict[str, Any]) -> Dict[str, Any]:
    visual_facts = make_visual_fact_objects(teacher["visual_facts"])
    evidence_fact_ids = [item["id"] for item in visual_facts[: min(3, len(visual_facts))]]

    return {
        "need_explanation": True,
        "visual_facts": visual_facts,
        "evidence_fact_ids": evidence_fact_ids,
        "text_literal": teacher["text_literal"],
        "incongruity": teacher["incongruity"],
        "label": label_gt,
        "explanation": teacher["explanation"],
        "missing_modalities": _missing_modalities(modality),
    }


def build_final_record(
    row: Dict[str, Any],
    teacher: Dict[str, Any],
    target_json: Dict[str, Any],
    query: str,
) -> Dict[str, Any]:
    record = dict(row)
    record["target_json"] = json.dumps(target_json, ensure_ascii=False)
    record["visual_facts"] = teacher["visual_facts"]
    record["source"] = "redeval"
    record["quality_flags"] = teacher["quality_flags"]
    record["teacher"] = json.dumps(teacher, ensure_ascii=False)
    record["language"] = "en"
    record["query"] = query
    return record


def process_split(
    split_name: str,
    input_path: Path,
    output_path: Path,
    failed_path: Path,
    teacher_generator: Callable[[str, str], Dict[str, Any]],
    query: str,
) -> Dict[str, Any]:
    rows = read_jsonl(input_path)
    # Deterministic regeneration: clear outputs at split start.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    failed_path.write_text("", encoding="utf-8")

    stats: Dict[str, Any] = {
        "split": split_name,
        "total": len(rows),
        "success": 0,
        "failed": 0,
        "both_label_mismatch": 0,
        "modality": Counter(),
        "label_gt": Counter(),
    }

    for row in tqdm(rows, desc=f"Generating {split_name}"):
        row_id = str(row.get("id", "<missing_id>")) if isinstance(row, dict) else "<invalid_row>"
        try:
            row_id, image_path, caption, label_gt, modality = _validate_row(row)
            teacher = teacher_generator(image_path_to_data_url(image_path), caption)
            validate_teacher_json(teacher)

            if modality == "both" and teacher["label"] != label_gt:
                stats["both_label_mismatch"] += 1
                logger.warning(
                    "Label mismatch id=%s modality=both teacher=%s label_gt=%s",
                    row_id,
                    teacher["label"],
                    label_gt,
                )

            target_json = build_target_json(label_gt=label_gt, modality=modality, teacher=teacher)
            append_jsonl(
                output_path,
                build_final_record(row=row, teacher=teacher, target_json=target_json, query=query),
            )

            stats["success"] += 1
            stats["modality"][modality] += 1
            stats["label_gt"][label_gt] += 1
        except Exception as exc:
            stats["failed"] += 1
            append_jsonl(
                failed_path,
                {
                    "id": row_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "modality": row.get("modality") if isinstance(row, dict) else "",
                    "label_gt": row.get("label_gt") if isinstance(row, dict) else "",
                },
            )
            logger.exception("Failed processing split=%s id=%s", split_name, row_id)

    return stats


def print_split_stats(stats: Dict[str, Any], output_path: Path, failed_path: Path) -> None:
    print(f"\n=== {stats['split']} stats ===")
    print(f"total={stats['total']}")
    print(f"success={stats['success']}")
    print(f"failed={stats['failed']}")
    print(f"modality={dict(stats['modality'])}")
    print(f"label_gt={dict(stats['label_gt'])}")
    print(f"both_label_mismatch={stats['both_label_mismatch']}")
    print(f"output={output_path}")
    print(f"failed_log={failed_path}")


def run_redeval_generation() -> Dict[str, Dict[str, Any]]:
    logistics = Logistics()
    queries = Queries()

    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    client = OpenAI(api_key=api_key)
    spec = SarcasmOODGen.v1()

    root = Path(logistics.project_root_dir)
    input_dir = root / logistics.data_generation_dir / "redeval"
    output_dir = root / logistics.processed_data_dir / "redeval"
    output_dir.mkdir(parents=True, exist_ok=True)

    def teacher_generator(image_url: str, caption: str) -> Dict[str, Any]:
        return request_teacher_annotation(
            client=client,
            spec=spec,
            model=model,
            image_url=image_url,
            caption=caption,
        )

    split_inputs = {
        "ood": input_dir / "ood.jsonl",
        "ood_perturb": input_dir / "ood_perturb.jsonl",
    }

    summary: Dict[str, Dict[str, Any]] = {}
    for split_name, input_path in split_inputs.items():
        output_path = output_dir / f"{split_name}.jsonl"
        failed_path = output_dir / f"{split_name}_failed.jsonl"
        stats = process_split(
            split_name=split_name,
            input_path=input_path,
            output_path=output_path,
            failed_path=failed_path,
            teacher_generator=teacher_generator,
            query=queries.DPO_QUERY,
        )
        print_split_stats(stats, output_path=output_path, failed_path=failed_path)
        summary[split_name] = stats

    total_rows = sum(stats["total"] for stats in summary.values())
    total_success = sum(stats["success"] for stats in summary.values())
    total_failed = sum(stats["failed"] for stats in summary.values())
    total_mismatches = sum(stats["both_label_mismatch"] for stats in summary.values())

    print("\n=== combined stats ===")
    print(f"total={total_rows}")
    print(f"success={total_success}")
    print(f"failed={total_failed}")
    print(f"both_label_mismatch={total_mismatches}")
    return summary


def main() -> None:
    run_redeval_generation()


if __name__ == "__main__":
    main()
