import json
from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime

def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    from pathlib import Path
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_store_record(
    sample_id: str,
    caption: str,
    model_json: Dict[str, Any],
    model_name: str = "gpt-5.2",
) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "caption": caption,
        "teacher_model": model_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "teacher": model_json,   # keep nested to preserve original structure
    }


def serialize_chosen(teacher: Dict[str, Any]) -> str:
    vf = "\n".join([f"- {x}" for x in teacher["visual_facts"]])
    return (
        "[VISUAL_FACTS]\n"
        f"{vf}\n"
        "[TEXT_LITERAL]\n"
        f"{teacher['text_literal'].strip()}\n"
        "[INCONGRUITY]\n"
        f"{teacher['incongruity'].strip()}\n"
        "[EXPLANATION]\n"
        f"{teacher['explanation'].strip()}\n"
    )


def make_rejected_omit_visual(teacher: Dict[str, Any]) -> str:
    # Intentionally ignores image facts; plausible but wrong
    return (
        "[VISUAL_FACTS]\n"
        "- (not used)\n"
        "[TEXT_LITERAL]\n"
        f"{teacher['text_literal'].strip()}\n"
        "[INCONGRUITY]\n"
        "The caption is sarcastic based on tone and common usage, without needing visual context.\n"
        "[EXPLANATION]\n"
        "The speaker likely means the opposite of the literal caption in a generic ironic way.\n"
    )



def make_dpo_record(sample_id: str, caption: str, teacher: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"Explain why the caption is sarcastic given the image.\nCAPTION: {caption.strip()}\n"
    chosen = serialize_chosen(teacher)
    rejected = make_rejected_omit_visual(teacher)

    return {
        "id": sample_id,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "language": teacher["language"],
        "quality_flags": teacher["quality_flags"],
    }



def validate_minimal(obj: Dict[str, Any]) -> None:
    """
    Minimal sanity checks (fast, no extra deps).
    """
    required = [
        "schema_version","language","keep","keep_reason",
        "politics_detected","politics_signals",
        "visual_facts","text_literal","incongruity","explanation",
        "quality_flags"
    ]
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    if not isinstance(obj["keep"], bool):
        raise ValueError("keep must be boolean")

    if not isinstance(obj["visual_facts"], list) or not all(isinstance(x, str) for x in obj["visual_facts"]):
        raise ValueError("visual_facts must be a list[str]")

    # If kept, require non-empty reasoning fields
    if obj["keep"]:
        if len(obj["visual_facts"]) < 3 and "low_image_clarity" not in obj["quality_flags"]:
            raise ValueError("keep=true but visual_facts < 3 (and not low_image_clarity)")
        if not obj["incongruity"].strip():
            raise ValueError("keep=true but incongruity is empty")
        if not obj["explanation"].strip():
            raise ValueError("keep=true but explanation is empty")


def get_completed_ids(
    train_path: str = "data/generated/mmsd2/raw/en/train.jsonl",
    validation_path: str = "data/generated/mmsd2/raw/en/validation.jsonl",
) -> Dict[str, List[str]]:
    def read_ids(path: Path) -> List[str]:
        if not path.exists():
            return []

        ids: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_num} in {path}") from exc
                if "id" not in record:
                    raise ValueError(f"Missing id on line {line_num} in {path}")
                ids.append(str(record["id"]))
        return ids

    return {
        "train": read_ids(Path(train_path)),
        "validation": read_ids(Path(validation_path)),
    }



def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
