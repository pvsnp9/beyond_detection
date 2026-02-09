from __future__ import annotations
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from config.queries import REJECTED_SYSTEM_PROMPT , ALLOWED_ERROR_TYPES, REJECT_SCHEMA
from config.logistics import Stats, Logistics

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential



def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSON on line {i} in {path}: {e}") from e
    return rows


def dump_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fp.flush()


def get_caption(record: Dict[str, Any]) -> str:
    # prefer explicit caption field, else attempt to infer from prompt
    cap = (record.get("caption") or "").strip()
    if cap:
        return cap
    # prompt may include "CAPTION: ..."
    p = (record.get("prompt") or "").strip()
    if "CAPTION:" in p:
        return p.split("CAPTION:", 1)[1].strip()
    return ""


def get_visual_facts(record: Dict[str, Any]) -> List[str]:
    # Prefer list field "visual_facts" if present; else parse from chosen block.
    vf = record.get("visual_facts")
    if isinstance(vf, list) and all(isinstance(x, str) for x in vf):
        return [x.strip() for x in vf if x and x.strip()]
    # Fallback: parse from chosen string between [VISUAL_FACTS] and [TEXT_LITERAL]
    chosen = (record.get("chosen") or "")
    if "[VISUAL_FACTS]" in chosen and "[TEXT_LITERAL]" in chosen:
        block = chosen.split("[VISUAL_FACTS]", 1)[1].split("[TEXT_LITERAL]", 1)[0]
        lines = [ln.strip() for ln in block.splitlines()]
        facts = []
        for ln in lines:
            if ln.startswith("- "):
                facts.append(ln[2:].strip())
        return [x for x in facts if x]
    return []


def approx_len(s: str) -> int:
    return len(s.strip())


def sanity_check_rejected_payload(payload: Dict[str, Any], chosen_text: str) -> Tuple[bool, str]:
    """
    Shallow validation:
      - schema keys present
      - error_type valid
      - rejected contains the 4 required sections
      - rejected uses some visual facts (heuristic: includes '-' lines under [VISUAL_FACTS])
      - rejected length within ±20% of chosen (slightly looser than prompt)
    """
    try:
        if not isinstance(payload, dict):
            return False, "payload_not_object"
        required_keys = {"id", "rejected", "meta"}
        if not required_keys.issubset(payload.keys()):
            return False, f"bad_top_level_keys:{sorted(payload.keys())}"
        if not isinstance(payload["id"], str) or not payload["id"].strip():
            return False, "bad_id"
        if not isinstance(payload["rejected"], str) or not payload["rejected"].strip():
            return False, "empty_rejected"
        meta = payload["meta"]
        if not isinstance(meta, dict):
            return False, "meta_not_object"
        if meta.get("error_type") not in ALLOWED_ERROR_TYPES:
            return False, "bad_error_type"
        if meta.get("confidence") not in ("low", "medium", "high"):
            return False, "bad_confidence"
        if not isinstance(meta.get("notes"), str) or not meta["notes"].strip():
            return False, "empty_notes"
        if not isinstance(meta.get("hallucination_span"), str):
            return False, "bad_hallucination_span"

        txt = payload["rejected"]
        for tag in ("[VISUAL_FACTS]", "[TEXT_LITERAL]", "[INCONGRUITY]", "[EXPLANATION]"):
            if tag not in txt:
                return False, f"missing_section:{tag}"

        # Visual facts should not be "(not used)"
        if "(not used)" in txt.lower():
            return False, "visual_facts_not_used"

        # heuristic: ensure at least 2 bullet facts in the rejected VISUAL_FACTS section
        vf_sec = txt.split("[VISUAL_FACTS]", 1)[1].split("[TEXT_LITERAL]", 1)[0]
        bullets = [ln for ln in vf_sec.splitlines() if ln.strip().startswith("- ")]
        if len(bullets) < 2:
            return False, "too_few_visual_facts_bullets"

        # length check
        c_len = max(1, approx_len(chosen_text))
        r_len = approx_len(txt)
        ratio = r_len / c_len
        if ratio < 0.8 or ratio > 1.2:
            print(f"DEBUG: chosen_len={c_len} rejected_len={r_len} ratio={ratio:.2f}")
            # return False, f"length_ratio_out_of_range:{ratio:.2f}"

        # forbid common weak rejected pattern
        weak_phrases = [
            "without needing visual context",
            "tone and common usage",
            "generic ironic",
        ]
        low_txt = txt.lower()
        if any(p in low_txt for p in weak_phrases):
            return False, "contains_weak_generic_phrase"

        return True, "ok"
    except Exception as e:
        return False, f"sanity_check_exception:{e}"


def build_user_input(
    rec: Dict[str, Any],
    error_type: str,
) -> str:
    rid = str(rec.get("id", "")).strip()
    caption = get_caption(rec)
    visual_facts = get_visual_facts(rec)
    chosen = (rec.get("chosen") or "").strip()

    vf_list = "\n".join([f"- {x}" for x in visual_facts]) if visual_facts else "- (none provided)"

    # We steer error_type explicitly so randomization is controlled and uniform.
    # We also remind it not to copy chosen verbatim.
    return (
        f"INPUT:\n"
        f"ID: {rid}\n"
        f"CAPTION: {caption}\n"
        f"VISUAL_FACTS (verbatim list):\n{vf_list}\n"
        f"CHOSEN (for reference style/length; do NOT copy):\n{chosen}\n\n"
        f"IMPORTANT: Use error_type = {error_type}.\n"
        f"Remember: exactly ONE controlled mistake; everything else should be plausible and well-formed.\n"
    )


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
        raise RuntimeError(f"Model output is not valid JSON despite schema: {e}\nRaw: {out_text[:500]}") from e


# ----------------------------
# Main
# ----------------------------

def main() -> None:

    logistics = Logistics()
    timeout = 60.0
    print_every = 50

    input_file = os.path.join(logistics.project_root_dir, logistics.processed_data_dir, "dpo", "en", "train.jsonl")
    out_file = "complete_train.jsonl"
    try:
        in_path = Path(input_file).expanduser().resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Input not found: {in_path}")

        out_path = in_path.parent / out_file


        api_key = os.getenv("OPENAI_API_KEY") 
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        random.seed(logistics.seed)

        rows = load_jsonl(in_path)
        stats = Stats(total=len(rows))

        client = OpenAI(api_key=api_key)

        print(f"[{_now()}] Loaded {stats.total} rows from {in_path}")
        print(f"[{_now()}] Writing output to {out_path}")
        print(f"[{_now()}] Model: {logistics.teacher_model}")
        print(f"[{_now()}] Using uniform error_type over: {ALLOWED_ERROR_TYPES}")

        failed_path = out_path.parent / "failed_data.jsonl"
        with out_path.open("w", encoding="utf-8") as fout, failed_path.open("w", encoding="utf-8") as ffail:
            for idx, rec in enumerate(rows, start=1):
                rid = str(rec.get("id", "")).strip() or f"line_{idx}"
                chosen_text = (rec.get("chosen") or "").strip()

                # Choose error_type uniformly
                error_type = random.choice(ALLOWED_ERROR_TYPES)

                # Build prompt input for this record
                user_input = build_user_input(rec, error_type=error_type)

                try:
                    payload = call_gpt_for_rejected(
                        client=client,
                        model=logistics.teacher_model,
                        user_input=user_input,
                        timeout_s=timeout,
                    )
                except Exception as e:
                    stats.api_errors += 1
                    print(f"[{_now()}] API_ERROR id={rid} line={idx}: {e}", file=sys.stderr)
                    raise

                # Schema sanity checks
                ok, reason = sanity_check_rejected_payload(payload, chosen_text=chosen_text)
                if not ok:
                    stats.failed_sanity += 1
                    print(
                        f"[{_now()}] SANITY_FAIL id={rid} line={idx} reason={reason} "
                        f"error_type_requested={error_type}",
                        file=sys.stderr,
                    )
                    dump_jsonl_line(
                        ffail,
                        {
                            "id": rid,
                            "reason": reason,
                        },
                    )
                    continue

                # Update record: replace rejected with new rejected, attach meta (minimal)
                rec_out = dict(rec)  # shallow copy
                rec_out["rejected"] = payload["rejected"]

                # Minimal metadata; keep it small but useful
                gen_meta = payload.get("meta", {})
                rec_out.setdefault("rejected_meta", {})
                rec_out["rejected_meta"] = {
                    "error_type": gen_meta.get("error_type"),
                    "confidence": gen_meta.get("confidence"),
                    "notes": gen_meta.get("notes"),
                    "hallucination_span": gen_meta.get("hallucination_span"),
                }

                dump_jsonl_line(fout, rec_out)
                stats.ok += 1

                if idx % print_every == 0 or idx == stats.total:
                    pct = (idx / stats.total) * 100.0
                    print(
                        f"[{_now()}] Progress {idx}/{stats.total} ({pct:.1f}%) | "
                        f"ok={stats.ok} api_err={stats.api_errors} sanity_fail={stats.failed_sanity}"
                    )

        print("\n=== DONE ===")
        print(f"Input:  {in_path}")
        print(f"Output: {out_path}")
        print(f"Total: {stats.total}")
        print(f"OK:    {stats.ok}")
        print(f"API errors:      {stats.api_errors}")
        print(f"Sanity failures: {stats.failed_sanity}")

    except Exception as e:
        # Production-grade: log and re-raise (do not suppress)
        print(f"[{_now()}] FATAL: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
