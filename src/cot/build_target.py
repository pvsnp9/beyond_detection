import json
from typing import Any, Dict, List
import random
from config.queries import Queries
from datasets import Features, Value, Sequence, Image, ClassLabel

def label_to_str(label: int) -> str:
    return "sarcastic" if int(label) == 1 else "non_sarcastic"

def get_label_int(label_str: str) -> int:
    return 1 if label_str == "sarcastic" else 0

def make_visual_fact_objects(teacher_visual_facts: List[str]) -> List[Dict[str, Any]]:
    return [{"id": i + 1, "fact": s} for i, s in enumerate(teacher_visual_facts)]

def default_evidence_ids(visual_facts_objs: List[Dict[str, Any]], k: int = 3) -> List[int]:
    return [vf["id"] for vf in visual_facts_objs[: min(k, len(visual_facts_objs))]]

# Build target for label-only (detection) and explanation tasks
def build_target_label_only(
    *,
    label_str: str,
    need_explanation: bool,
    teacher: Dict[str, Any]
) -> Dict[str, Any]:
    is_sarc = (label_str == "sarcastic")

    # Detection-only response (no explanation requested / needed)
    if not need_explanation:
        return {
            "label": label_str,
            "need_explanation": False,
            "visual_facts": [],
            "evidence_fact_ids": [],
            "incongruity": "",
            "explanation": ""
        }

    # Explanation requested/needed
    if is_sarc:
        vfacts = make_visual_fact_objects(teacher.get("visual_facts", []))
        return {
            "label": "sarcastic",
            "need_explanation": True,
            "visual_facts": vfacts,
            "evidence_fact_ids": default_evidence_ids(vfacts, k=3),
            "incongruity": teacher.get("incongruity", "").strip(),
            "explanation": teacher.get("explanation", "").strip()
        }

    # Non-sarcastic but explanation requested (e.g., "Explain why" queries on negatives)
    vfacts = make_visual_fact_objects(teacher.get("visual_facts", [])[:3])
    return {
        "label": "non_sarcastic",
        "need_explanation": True,
        "visual_facts": vfacts,
        "evidence_fact_ids": default_evidence_ids(vfacts, k=2),
        "incongruity": "",
        "explanation": "Not sarcastic: the caption aligns with what is shown in the image."
    }

quries = Queries()

def build_sft_rows_from_raw(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    teacher = raw["teacher"]
    label_gt = label_to_str(raw["label"])
    caption = raw["caption"]
    img_path = raw["local_image_path"]
    source = raw.get("source", "")
    language = teacher.get("language", "")
    quality_flags = teacher.get("quality_flags", [])

    # Base dictionary for all instances of this image
    base = {
        "id": raw["id"],
        "image": img_path, 
        "caption": caption,
        "label_gt": label_gt,
        "language": language,
        "source": source,
        "quality_flags": quality_flags,
        "teacher": json.dumps(teacher, ensure_ascii=False)
    }

    rows = []

    # --- TASK 1: Standard Detection (Always exists) ---
    q_det = random.choice(quries.DETECTION_QUERIES)
    target_det = build_target_label_only(label_str=label_gt, need_explanation=False, teacher=teacher)
    rows.append({
        **base, 
        "task": "detection", 
        "query": q_det, 
        "target_json": json.dumps(target_det, ensure_ascii=False)
    })

    # --- TASK 2: Enhanced Grounding (Detection vs Explanation) ---
    if label_gt == "sarcastic":
        # Sarcastic: Use the Explanation Query to teach "Why"
        q_exp = random.choice(quries.EXPLANATION_QUERIES)
        target_exp = build_target_label_only(label_str=label_gt, need_explanation=True, teacher=teacher)
        rows.append({
            **base, 
            "task": "explanation", 
            "query": q_exp, 
            "target_json": json.dumps(target_exp, ensure_ascii=False)
        })
    else:
        # Non-Sarcastic: Avoid "Why is this sarcastic?" queries.
        # Use a different Detection Query but set need_explanation=True 
        # to teach the model to PROVE it is non-sarcastic via visual facts.
        remaining_det_queries = [q for q in quries.DETECTION_QUERIES if q != q_det]
        q_det_alt = random.choice(remaining_det_queries) if remaining_det_queries else q_det
        
        target_non_sarc_exp = build_target_label_only(label_str=label_gt, need_explanation=True, teacher=teacher)
        rows.append({
            **base, 
            "task": "detection_explanation", 
            "query": q_det_alt, 
            "target_json": json.dumps(target_non_sarc_exp, ensure_ascii=False)
        })

    return rows


def get_hf_sft_features() -> Features:
    return Features({
        "id": Value("string"),
        "image": Image(),
        "caption": Value("string"),
        "query": Value("string"),
        "task": Value("string"),
        "target_json": Value("string"),
        "label_gt": ClassLabel(names=["non_sarcastic", "sarcastic"]),
        "language": Value("string"),
        "source": Value("string"),
        "quality_flags": Sequence(Value("string")),
        "teacher": Value("string"),
    })


