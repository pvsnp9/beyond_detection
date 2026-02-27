import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from config.logistics import Logistics, Stats
from config.queries import (
    ALLOWED_ERROR_TYPES,
    REJECTED_SYSTEM_PROMPT,
    REJECT_SCHEMA,
    Queries,
    get_allowed_error_types_for_modality,
)


LABELS = {"sarcastic", "non_sarcastic", "unknown"}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _load_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # Ignore malformed historical rows so resume can still proceed.
                continue
            rid = str(rec.get("id", "")).strip()
            if rid:
                ids.add(rid)
    return ids


def load_jsonl(path: Path, exclude_ids: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSON on line {i} in {path}: {e}") from e
            rid = str(rec.get("id", "")).strip()
            if exclude_ids and rid and rid in exclude_ids:
                continue
            rows.append(rec)
    return rows


def dump_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


def _normalize_modality(value: Any) -> str:
    modality = str(value or "").strip().lower()
    if modality in {"both", "image", "text"}:
        return modality
    raise ValueError(f"Unsupported modality: {value!r}")


def _expected_missing_modalities(modality: str) -> List[str]:
    if modality == "text":
        return ["image"]
    if modality == "image":
        return ["text"]
    return []


def _normalize_missing_modalities(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if item in ("image", "text") and item not in out:
            out.append(item)
    return out


def _normalize_quality_flags(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        flag = item.strip()
        if not flag or flag == "NONE" or flag in seen:
            continue
        seen.add(flag)
        out.append(flag)
    return out


def get_caption(record: Dict[str, Any]) -> str:
    cap = (record.get("caption") or "").strip()
    if cap:
        return cap
    p = (record.get("query") or "").strip()
    if "CAPTION:" in p:
        return p.split("CAPTION:", 1)[1].strip()
    return ""


def parse_chosen_object(record: Dict[str, Any]) -> Dict[str, Any]:
    raw = record.get("target_json")
    if raw is None:
        raw = record.get("chosen")
    if isinstance(raw, dict):
        chosen = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            chosen = json.loads(raw)
        except Exception as e:
            raise ValueError(f"Invalid target_json/chosen JSON: {e}") from e
    else:
        raise ValueError("Missing target_json/chosen source for chosen object")

    if not isinstance(chosen, dict):
        raise ValueError("Chosen target must be a JSON object")
    return chosen


def chosen_modality_matches_row_modality(chosen: Dict[str, Any], modality: str) -> bool:
    missing = set(_normalize_missing_modalities(chosen.get("missing_modalities", [])))
    expected = set(_expected_missing_modalities(modality))
    return missing == expected


def extract_visual_facts_for_prompt(record: Dict[str, Any], chosen: Dict[str, Any]) -> List[str]:
    top_level = record.get("visual_facts")
    if isinstance(top_level, list):
        out = [str(x).strip() for x in top_level if str(x).strip()]
        if out:
            return out

    vf = chosen.get("visual_facts", [])
    out: List[str] = []
    if isinstance(vf, list):
        for item in vf:
            if isinstance(item, dict):
                fact = str(item.get("fact", "")).strip()
            else:
                fact = str(item).strip()
            if fact:
                out.append(fact)
    return out


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _auto_sanitize_rejected_for_modality(payload: Dict[str, Any], modality: str) -> None:
    """Apply low-risk modality-specific repairs before validation."""
    if not isinstance(payload, dict):
        return
    rejected = payload.get("rejected")
    if not isinstance(rejected, dict):
        return

    if modality == "image":
        # Image-only examples should not populate text-derived fields.
        if isinstance(rejected.get("text_literal"), str):
            rejected["text_literal"] = ""
        if isinstance(rejected.get("incongruity"), str):
            rejected["incongruity"] = ""


def _validate_rejected_object_shape(obj: Dict[str, Any]) -> Tuple[bool, str]:
    required = {
        "need_explanation",
        "visual_facts",
        "evidence_fact_ids",
        "text_literal",
        "incongruity",
        "label",
        "explanation",
        "missing_modalities",
    }
    if not isinstance(obj, dict):
        return False, "rejected_not_object"
    if set(obj.keys()) != required:
        return False, f"bad_rejected_keys:{sorted(obj.keys())}"
    if not isinstance(obj["need_explanation"], bool):
        return False, "bad_need_explanation"
    if obj.get("label") not in LABELS:
        return False, "bad_label"
    for key in ("text_literal", "incongruity", "explanation"):
        if not isinstance(obj.get(key), str):
            return False, f"bad_{key}"
    if not isinstance(obj.get("evidence_fact_ids"), list) or not all(
        isinstance(x, int) for x in obj["evidence_fact_ids"]
    ):
        return False, "bad_evidence_fact_ids"
    if not isinstance(obj.get("missing_modalities"), list) or not all(
        isinstance(x, str) for x in obj["missing_modalities"]
    ):
        return False, "bad_missing_modalities"
    vf = obj.get("visual_facts")
    if not isinstance(vf, list):
        return False, "bad_visual_facts"
    for i, item in enumerate(vf):
        if not isinstance(item, dict):
            return False, f"bad_visual_fact_item:{i}"
        if set(item.keys()) != {"id", "fact"}:
            return False, f"bad_visual_fact_keys:{i}"
        if not isinstance(item.get("id"), int) or not isinstance(item.get("fact"), str):
            return False, f"bad_visual_fact_types:{i}"
    return True, "ok"


def sanity_check_rejected_payload(
    payload: Dict[str, Any],
    *,
    rec: Dict[str, Any],
    chosen_obj: Dict[str, Any],
    modality: str,
    requested_error_type: str,
    allowed_error_types: List[str],
) -> Tuple[bool, str]:
    try:
        if not isinstance(payload, dict):
            return False, "payload_not_object"
        if set(payload.keys()) != {"id", "rejected", "meta"}:
            return False, f"bad_top_level_keys:{sorted(payload.keys())}"
        if not isinstance(payload.get("id"), str) or not payload["id"].strip():
            return False, "bad_id"

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            return False, "meta_not_object"
        meta_required = {"error_type", "error_field", "hallucination_span", "confidence", "notes"}
        if not meta_required.issubset(meta.keys()):
            return False, "meta_missing_keys"
        error_type = meta.get("error_type")
        if error_type not in ALLOWED_ERROR_TYPES:
            return False, "bad_error_type_global"
        if error_type not in allowed_error_types:
            return False, "bad_error_type_for_modality"
        if error_type != requested_error_type:
            return False, "error_type_not_requested"
        if not isinstance(meta.get("error_field"), str) or not meta["error_field"].strip():
            return False, "empty_error_field"
        if meta.get("confidence") not in ("low", "medium", "high"):
            return False, "bad_confidence"
        if not isinstance(meta.get("notes"), str) or not meta["notes"].strip():
            return False, "empty_notes"
        if not isinstance(meta.get("hallucination_span"), str):
            return False, "bad_hallucination_span"

        rejected = payload.get("rejected")
        ok_shape, reason = _validate_rejected_object_shape(rejected)
        if not ok_shape:
            return False, reason

        expected_missing = _expected_missing_modalities(modality)
        got_missing = _normalize_missing_modalities(rejected.get("missing_modalities", []))
        if sorted(got_missing) != sorted(expected_missing):
            return False, "modality_missing_modalities_mismatch"

        visual_facts = rejected.get("visual_facts", [])
        evidence_ids = rejected.get("evidence_fact_ids", [])
        visual_ids = {item["id"] for item in visual_facts}
        if not set(evidence_ids).issubset(visual_ids):
            return False, "evidence_ids_not_subset_of_visual_fact_ids"

        if modality == "text":
            if visual_facts or evidence_ids:
                return False, "text_only_should_not_have_visual_facts"
            if error_type != "label_mismatch":
                return False, "text_only_error_type_must_be_label_mismatch"
        elif modality == "image":
            if str(rejected.get("text_literal", "")).strip():
                return False, "image_only_text_literal_must_be_empty"
            if str(rejected.get("incongruity", "")).strip():
                return False, "image_only_incongruity_must_be_empty"
        elif modality == "both":
            if got_missing:
                return False, "both_modality_should_have_no_missing_modalities"

        if error_type == "label_mismatch":
            if rejected.get("label") == chosen_obj.get("label"):
                return False, "label_mismatch_but_label_same_as_chosen"

        halluc_span = str(meta.get("hallucination_span", ""))
        rejected_text = _canonical_json(rejected)
        if error_type == "hallucinated_visual_detail":
            if not halluc_span.strip():
                return False, "hallucinated_visual_detail_missing_span"
            # Non-fatal: exact span matching is brittle (quotes/spacing/punctuation differences).
            # Keep the sample if the hallucination error type and other constraints are valid.
        elif halluc_span.strip():
            return False, "non_hallucination_error_has_hallucination_span"

        if _canonical_json(rejected) == _canonical_json(chosen_obj):
            return False, "rejected_identical_to_chosen"

        weak_phrases = [
            "tone and common usage",
            "without visual context",
        ]
        low_text = rejected_text.lower()
        if any(p in low_text for p in weak_phrases):
            return False, "contains_weak_generic_phrase"

        return True, "ok"
    except Exception as e:
        return False, f"sanity_check_exception:{e}"


def build_user_input(
    *,
    rec: Dict[str, Any],
    chosen_obj: Dict[str, Any],
    modality: str,
    requested_error_type: str,
    allowed_error_types: List[str],
) -> str:
    rid = str(rec.get("id", "")).strip()
    caption = get_caption(rec)
    visual_facts = extract_visual_facts_for_prompt(rec, chosen_obj)
    expected_missing = _expected_missing_modalities(modality)

    payload = {
        "id": rid,
        "modality": modality,
        "missing_modalities_from_modality": expected_missing,
        "caption": caption,
        "visual_facts": visual_facts,
        "chosen_json": chosen_obj,
        "allowed_error_types_for_this_record": allowed_error_types,
        "requested_error_type": requested_error_type,
    }
    return json.dumps(payload, ensure_ascii=False)


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((RateLimitError, APIError, APITimeoutError, APIConnectionError)),
    reraise=True,
)
def completion_with_backoff(client: OpenAI, **call_kwargs):
    return client.responses.create(**call_kwargs)


def call_gpt_for_rejected(
    client: OpenAI,
    model: str,
    user_input: str,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if timeout_s is not None:
        kwargs["timeout"] = timeout_s

    resp = completion_with_backoff(
        client=client,
        model=model,
        input=[
            {"role": "system", "content": REJECTED_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": REJECT_SCHEMA["name"],
                "schema": REJECT_SCHEMA["schema"],
                "strict": REJECT_SCHEMA["strict"],
            }
        },
        temperature=0,
        **kwargs,
    )

    out_text = getattr(resp, "output_text", None)
    if not isinstance(out_text, str) or not out_text.strip():
        raise RuntimeError("Empty output_text from model response.")
    try:
        return json.loads(out_text)
    except Exception as e:
        raise RuntimeError(f"Model output is not valid JSON despite schema: {e}") from e


def _build_final_dpo_row(
    rec: Dict[str, Any],
    chosen_obj: Dict[str, Any],
    rejected_obj: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    visual_facts = rec.get("visual_facts")
    if not isinstance(visual_facts, list):
        visual_facts = extract_visual_facts_for_prompt(rec, chosen_obj)

    return {
        "id": str(rec.get("id", "")),
        "image": rec.get("image") or rec.get("local_image_path"),
        "caption": rec.get("caption", ""),
        "modality": _normalize_modality(rec.get("modality", "both")),
        "query": Queries().DPO_QUERY,
        "chosen": json.dumps(chosen_obj, ensure_ascii=False),
        "rejected": json.dumps(rejected_obj, ensure_ascii=False),
        "label_gt": rec.get("label_gt", "unknown"),
        "language": rec.get("language", "en"),
        "source": rec.get("source", ""),
        "quality_flags": _normalize_quality_flags(rec.get("quality_flags", [])),
        "visual_facts": visual_facts if isinstance(visual_facts, list) else [],
        "rejected_meta": {
            "error_type": meta.get("error_type", ""),
            "error_field": meta.get("error_field", ""),
            "confidence": meta.get("confidence", ""),
            "notes": meta.get("notes", ""),
            "hallucination_span": meta.get("hallucination_span", ""),
        },
        "tier": rec.get("tier"),
    }


def main() -> None:
    logistics = Logistics()
    timeout = 60.0
    print_every = 50

    input_file = os.path.join(
        logistics.project_root_dir, logistics.processed_data_dir, "dpo", "en", "train.jsonl"
    )
    out_file = "gen_train.jsonl"
    failed_file = "gen_failed.jsonl"

    try:
        in_path = Path(input_file).expanduser().resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

        out_path = in_path.parent / out_file
        failed_path = in_path.parent / failed_file

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        completed_ids = _load_existing_ids(out_path)

        random.seed(logistics.seed)
        rows = load_jsonl(in_path, exclude_ids=completed_ids)
        stats = Stats(total=len(rows))
        client = OpenAI(api_key=api_key)

        print(f"[{_now()}] Found {len(completed_ids)} existing generated rows in {out_path}", flush=True)
        print(f"[{_now()}] Loaded {stats.total} remaining rows from {in_path}", flush=True)
        print(f"[{_now()}] Writing output to {out_path} (append)", flush=True)
        print(f"[{_now()}] Writing failures to {failed_path} (overwrite)", flush=True)
        print(f"[{_now()}] Model: {logistics.teacher_model}", flush=True)

        seen_success_ids = set(completed_ids)
        with out_path.open("a", encoding="utf-8") as fout, failed_path.open("w", encoding="utf-8") as ffail:
            for idx, rec in enumerate(rows, start=1):
                rid = str(rec.get("id", "")).strip() or f"line_{idx}"
                if rid in seen_success_ids:
                    # Defensive duplicate guard if input contains repeated IDs.
                    continue

                try:
                    modality = _normalize_modality(rec.get("modality", "both"))
                except Exception as e:
                    stats.failed_schema += 1
                    dump_jsonl_line(
                        ffail,
                        {
                            "id": rid,
                            "reason": f"bad_modality:{e}",
                            "error_stage": "precheck",
                        },
                    )
                    continue

                try:
                    chosen_obj = parse_chosen_object(rec)
                except Exception as e:
                    stats.failed_schema += 1
                    dump_jsonl_line(
                        ffail,
                        {
                            "id": rid,
                            "reason": f"bad_chosen_source:{e}",
                            "error_stage": "precheck",
                            "modality": modality,
                        },
                    )
                    continue

                if not chosen_modality_matches_row_modality(chosen_obj, modality):
                    stats.failed_schema += 1
                    dump_jsonl_line(
                        ffail,
                        {
                            "id": rid,
                            "reason": "chosen_modality_mismatch",
                            "error_stage": "precheck",
                            "modality": modality,
                            "chosen_missing_modalities": _normalize_missing_modalities(
                                chosen_obj.get("missing_modalities", [])
                            ),
                        },
                    )
                    continue

                allowed_error_types = get_allowed_error_types_for_modality(modality)
                if not allowed_error_types:
                    stats.failed_schema += 1
                    dump_jsonl_line(
                        ffail,
                        {
                            "id": rid,
                            "reason": "no_allowed_error_types_for_modality",
                            "error_stage": "precheck",
                            "modality": modality,
                        },
                    )
                    continue

                requested_error_type = random.choice(allowed_error_types)
                user_input = build_user_input(
                    rec=rec,
                    chosen_obj=chosen_obj,
                    modality=modality,
                    requested_error_type=requested_error_type,
                    allowed_error_types=allowed_error_types,
                )

                try:
                    payload = call_gpt_for_rejected(
                        client=client,
                        model=logistics.teacher_model,
                        user_input=user_input,
                        timeout_s=timeout,
                    )
                except Exception as e:
                    stats.api_errors += 1
                    print(f"[{_now()}] API_ERROR id={rid} line={idx}: {e}", file=sys.stderr, flush=True)
                    dump_jsonl_line(
                        ffail,
                        {
                            "id": rid,
                            "reason": f"api_error:{e}",
                            "error_stage": "api",
                            "modality": modality,
                            "requested_error_type": requested_error_type,
                        },
                    )
                    continue

                _auto_sanitize_rejected_for_modality(payload, modality=modality)

                ok, reason = sanity_check_rejected_payload(
                    payload,
                    rec=rec,
                    chosen_obj=chosen_obj,
                    modality=modality,
                    requested_error_type=requested_error_type,
                    allowed_error_types=allowed_error_types,
                )
                if not ok:
                    stats.failed_sanity += 1
                    print(
                        f"[{_now()}] SANITY_FAIL id={rid} line={idx} reason={reason} "
                        f"modality={modality} requested_error_type={requested_error_type}",
                        file=sys.stderr,
                        flush=True,
                    )
                    dump_jsonl_line(
                        ffail,
                        {
                            "id": rid,
                            "reason": reason,
                            "error_stage": "validation",
                            "modality": modality,
                            "requested_error_type": requested_error_type,
                            "payload_meta": payload.get("meta", {}),
                        },
                    )
                    continue

                rec_out = _build_final_dpo_row(
                    rec=rec,
                    chosen_obj=chosen_obj,
                    rejected_obj=payload["rejected"],
                    meta=payload.get("meta", {}),
                )
                dump_jsonl_line(fout, rec_out)
                stats.ok += 1
                seen_success_ids.add(rid)

                if idx % print_every == 0 or idx == stats.total:
                    pct = (idx / stats.total) * 100.0 if stats.total else 0.0
                    print(
                        f"[{_now()}] Progress {idx}/{stats.total} ({pct:.1f}%) | "
                        f"ok={stats.ok} api_err={stats.api_errors} "
                        f"schema_fail={stats.failed_schema} sanity_fail={stats.failed_sanity}",
                        flush=True,
                    )

        print("\n=== DONE ===", flush=True)
        print(f"Input:  {in_path}", flush=True)
        print(f"Output: {out_path}", flush=True)
        print(f"Failed: {failed_path}", flush=True)
        print(f"Total: {stats.total}", flush=True)
        print(f"OK:    {stats.ok}", flush=True)
        print(f"API errors:      {stats.api_errors}", flush=True)
        print(f"Schema failures: {stats.failed_schema}", flush=True)
        print(f"Sanity failures: {stats.failed_sanity}", flush=True)

    except Exception as e:
        print(f"[{_now()}] FATAL: {e}", file=sys.stderr, flush=True)
        return


# if __name__ == "__main__":
#     main()
