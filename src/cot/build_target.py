import json
import re
from typing import Any, Dict, List
import random
from config.queries import Queries, NON_SARC_EXPLANATION_TEMPLATES
from config.logistics import REPLACEMENT_WORDS
from datasets import Features, Value, Sequence, Image, ClassLabel
from src.data_generation.dpo_creation import (
    MENTION_RE,
    WORD_RE,
    TAG_RE,
    SARCASTIC_TAGS,
    OTHER_TAGS,
    PLAIN_TAG_TOKENS,
)

quries = Queries()

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
            "text_literal": "",
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
            "text_literal": teacher.get("text_literal", "").strip(),
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
        "text_literal": teacher.get("text_literal", "").strip(),
        "incongruity": "",
        "explanation": random.choice(NON_SARC_EXPLANATION_TEMPLATES)
    }



def build_sft_rows_from_raw(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = clean_sft_record(raw)
    if not raw:
        return [] 
    
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
        "visual_facts": teacher.get("visual_facts", []),
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
    # for each example 
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
        "visual_facts": Sequence(Value("string")),
        "teacher": Value("string"),
    })


def get_hf_dpo_features()-> Features:
    return Features({
        "id": Value("string"),
        "image": Image(), 
        "caption": Value("string"),
        "prompt": Value("string"),
        "chosen": Value("string"),
        "rejected": Value("string"),
        "label_gt": ClassLabel(names=["non_sarcastic", "sarcastic"]),
        "language": Value("string"),
        "source": Value("string"),
        "quality_flags": Sequence(Value("string")),
        "visual_facts": Sequence(Value("string")),
        "rejected_meta": {
            "error_type": Value("string"),
            "confidence": Value("string"),
            "notes": Value("string"),
            "hallucination_span": Value("string"),
        },
    })



def clean_sft_record(reord: Dict[str, Any]):
    try:
        record = reord
        if not isinstance(record, dict):
            return None

        teacher = record.get("teacher", {})
        if not isinstance(teacher, dict):
            teacher = {}
        
        lang = teacher.get("language", "")
        if lang == "" or lang == "zh": return record

        def _normalize_quotes(text: str) -> str:
            return (
                text.replace("\u2018", "'")
                .replace("\u2019", "'")
                .replace("\u201C", '"')
                .replace("\u201D", '"')
            )

        disallowed_tags = set(SARCASTIC_TAGS) | set(OTHER_TAGS)
        hashtag_re = re.compile(r"#([A-Za-z0-9_-]+)")

        def _normalize_tag(tag: str) -> str:
            return re.sub(r"[^a-z0-9]", "", tag.lower())

        def _looks_like_tag_token(token: str) -> bool:
            if "_" in token or "-" in token:
                return True
            return any(ch.isupper() for ch in token) and any(ch.islower() for ch in token)

        def _extract_tag_tokens(text: str) -> List[str]:
            tokens = set(TAG_RE.findall(text))
            for word in WORD_RE.findall(text):
                if _looks_like_tag_token(word):
                    tokens.add(word)
                else:
                    normalized = _normalize_tag(word)
                    if normalized in PLAIN_TAG_TOKENS:
                        tokens.add(word)
            return list(tokens)

        def _mask_mentions(text: str) -> str:
            def _mask(match: re.Match[str]) -> str:
                token = match.group(1)
                if _normalize_tag(token) in disallowed_tags:
                    return ""
                return "@HANDLE"

            return MENTION_RE.sub(_mask, text)

        def _remove_disallowed_tokens(text: str) -> str:
            hashtag_tokens = set(TAG_RE.findall(text))
            tokens = set(WORD_RE.findall(text)) | set(_extract_tag_tokens(text))
            for token in set(tokens):
                normalized = _normalize_tag(token)
                contains_tag = any(tag in normalized for tag in disallowed_tags | PLAIN_TAG_TOKENS)
                if (
                    normalized in disallowed_tags
                    or normalized in PLAIN_TAG_TOKENS
                    or token in hashtag_tokens
                    or contains_tag
                ):
                    text = re.sub(rf"(?i)(?<!\\w){re.escape(token)}(?!\\w)", "", text)
            return text

        def _replace_disallowed_tokens(text: str) -> str:
            if not text:
                return text

            def _replace(match: re.Match[str]) -> str:
                token = match.group(0)
                normalized = _normalize_tag(token)
                contains_tag = any(tag in normalized for tag in disallowed_tags | PLAIN_TAG_TOKENS)
                if (
                    normalized in disallowed_tags
                    or normalized in PLAIN_TAG_TOKENS
                    or contains_tag
                ):
                    return random.choice(list(REPLACEMENT_WORDS))
                return token

            return WORD_RE.sub(_replace, text)

        def _contains_disallowed_or_handles(text: str) -> bool:
            if MENTION_RE.search(text):
                return True
            for token in WORD_RE.findall(text):
                if _normalize_tag(token) in disallowed_tags:
                    return True
            return False

        caption = record.get("caption", "")
        if not isinstance(caption, str):
            caption = str(caption)

        caption = _normalize_quotes(_mask_mentions(caption))
        caption = hashtag_re.sub(
            lambda match: "" if _normalize_tag(match.group(1)) in disallowed_tags else match.group(1),
            caption,
        )
        caption = _remove_disallowed_tokens(caption)
        caption = caption.replace("#", "")
        caption = re.sub(r"\s+", " ", caption).strip()
        if len(caption) < 5:
            return None
        record["caption"] = caption


        visual_facts = teacher.get("visual_facts", [])
        if isinstance(visual_facts, list):
            visual_facts_text = " ".join(str(fact) for fact in visual_facts)
        else:
            visual_facts_text = str(visual_facts)
        if visual_facts_text and _contains_disallowed_or_handles(visual_facts_text):
            return None
        
        cleaned_visual_facts: List[str] = []
        if isinstance(visual_facts, list):
            for fact in visual_facts:
                fact_text = str(fact)
                fact_text = _normalize_quotes(_mask_mentions(fact_text))
                fact_text = hashtag_re.sub(
                    lambda match: "" if _normalize_tag(match.group(1)) in disallowed_tags else match.group(1),
                    fact_text,
                )
                fact_text = _remove_disallowed_tokens(fact_text)
                fact_text = fact_text.replace("#", "")
                fact_text = re.sub(r"\s+", " ", fact_text).strip()
                if fact_text:
                    cleaned_visual_facts.append(fact_text)
        teacher["visual_facts"] = cleaned_visual_facts

        for key in ("text_literal", "incongruity", "explanation"):
            value = teacher.get(key)
            if isinstance(value, str):
                cleaned_value = _normalize_quotes(_mask_mentions(value))
                cleaned_value = _replace_disallowed_tokens(cleaned_value)
                cleaned_value = cleaned_value.replace("#", "")
                cleaned_value = re.sub(r"\s+", " ", cleaned_value).strip()
                teacher[key] = cleaned_value

        record["teacher"] = teacher
        return record
    except Exception:
        return None
