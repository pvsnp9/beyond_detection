
import json
from typing import Any

from config.logistics import ParsedRecord


JsonDict = dict[str, Any]


def safe_json_loads(x: Any) -> tuple[JsonDict | None, str | None]:
    if isinstance(x, dict):
        return x, None
    if isinstance(x, str):
        text = x.strip()
        if not text:
            return {}, None
        try:
            parsed = json.loads(text)
        except Exception as exc:  # noqa: BLE001
            return None, f"json parse error: {exc}"
        if not isinstance(parsed, dict):
            return None, "json parse error: expected object"
        return parsed, None
    if x is None:
        return {}, None
    return None, f"json parse error: unsupported type {type(x).__name__}"


def _coerce_str_or_none(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, str):
        value = x.strip()
        return value if value else None
    return str(x)


def _extract_facts(payload: JsonDict | None) -> list[str]:
    if not isinstance(payload, dict):
        return []
    facts = payload.get("visual_facts", [])
    if not isinstance(facts, list):
        return []

    out: list[str] = []
    for item in facts:
        if isinstance(item, dict):
            fact = item.get("fact")
            if isinstance(fact, str) and fact.strip():
                out.append(fact)
        elif isinstance(item, str) and item.strip():
            out.append(item)
    return out


def _extract_missing_modalities(payload: JsonDict | None) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    value = payload.get("missing_modalities", [])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    for v in value:
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
        elif v is not None:
            out.append(str(v))
    return tuple(out)


def _extract_quality_flags(raw: dict[str, Any]) -> tuple[str, ...]:
    value = raw.get("quality_flags", [])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    for v in value:
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
        elif v is not None:
            out.append(str(v))
    return tuple(out)


def _extract_bool_or_none(payload: JsonDict | None, key: str) -> bool | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    return None


def parse_record(raw: dict[str, Any]) -> ParsedRecord:
    rec_id = str(raw.get("id", ""))
    model_key = _coerce_str_or_none(raw.get("model_key"))
    source = _coerce_str_or_none(raw.get("source"))
    modality = _coerce_str_or_none(raw.get("modality"))
    gt = _coerce_str_or_none(raw.get("gt"))
    query = _coerce_str_or_none(raw.get("query"))
    quality_flags = _extract_quality_flags(raw)

    target_obj, target_err = safe_json_loads(raw.get("target_json"))
    output_obj, output_err = safe_json_loads(raw.get("output"))

    errors: list[str] = []
    if target_err:
        errors.append(f"target_json {target_err}")
    if output_err:
        errors.append(f"output {output_err}")

    ref_facts = _extract_facts(target_obj)
    pred_facts = _extract_facts(output_obj)
    missing_modalities_ref = _extract_missing_modalities(target_obj)
    missing_modalities_pred = _extract_missing_modalities(output_obj)
    need_explanation = _extract_bool_or_none(target_obj, "need_explanation")

    ref_explanation = ""
    ref_incongruity = ""
    ref_text_literal = ""
    if isinstance(target_obj, dict):
        v = target_obj.get("explanation", "")
        if isinstance(v, str):
            ref_explanation = v
        elif v is not None:
            ref_explanation = str(v)

        v = target_obj.get("incongruity", "")
        if isinstance(v, str):
            ref_incongruity = v
        elif v is not None:
            ref_incongruity = str(v)

        v = target_obj.get("text_literal", "")
        if isinstance(v, str):
            ref_text_literal = v
        elif v is not None:
            ref_text_literal = str(v)

    explanation = ""
    if isinstance(output_obj, dict):
        expl = output_obj.get("explanation", "")
        if isinstance(expl, str):
            explanation = expl
        elif expl is not None:
            explanation = str(expl)

    is_valid = len(errors) == 0
    error = "; ".join(errors) if errors else None

    return ParsedRecord(
        id=rec_id,
        model_key=model_key,
        source=source,
        modality=modality,
        quality_flags=quality_flags,
        gt=gt,
        query=query,
        ref_facts=ref_facts,
        pred_facts=pred_facts,
        explanation=explanation,
        ref_explanation=ref_explanation,
        ref_incongruity=ref_incongruity,
        ref_text_literal=ref_text_literal,
        need_explanation=need_explanation,
        missing_modalities_pred=missing_modalities_pred,
        missing_modalities_ref=missing_modalities_ref,
        is_valid=is_valid,
        error=error,
    )
