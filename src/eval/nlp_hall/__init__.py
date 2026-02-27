import json
import logging
from pathlib import Path
from typing import Any, Iterable

from config.logistics import HallEvalConfig
from src.eval.nlp_hall.io import parse_record

logger = logging.getLogger(__name__)


def _serialize_flags(flags: tuple[str, ...]) -> str:
    return "|".join(flags)


def _base_row(parsed: Any) -> dict[str, Any]:
    mismatch = set(parsed.missing_modalities_pred) != set(parsed.missing_modalities_ref)
    return {
        "id": parsed.id,
        "model_key": parsed.model_key,
        "source": parsed.source,
        "modality": parsed.modality,
        "gt": parsed.gt,
        "quality_flags": _serialize_flags(parsed.quality_flags),
        "missing_modalities_pred": _serialize_flags(parsed.missing_modalities_pred),
        "missing_modalities_ref": _serialize_flags(parsed.missing_modalities_ref),
        "missing_modality_mismatch": int(mismatch),
        "is_valid": int(bool(parsed.is_valid)),
        "error": parsed.error,
    }


def _fill_null_metrics(row: dict[str, Any]) -> dict[str, Any]:
    row.update(
        {
            "tp": None,
            "fp": None,
            "fn": None,
            "n_ref": None,
            "n_pred": None,
            "mean_match_sim": None,
            "add": None,
            "omit": None,
            "p": None,
            "r": None,
            "f1": None,
            "eg": None,
            "eh": None,
            "ec": None,
            "eqs": None,
            "k_claims": None,
            "len_tokens": None,
        }
    )
    return row


def evaluate_records(records: Iterable[dict[str, Any]], cfg: HallEvalConfig) -> list[dict[str, Any]]:
    from src.eval.nlp_hall.expl_metrics import compute_expl_metrics_routed
    from src.eval.nlp_hall.fact_metrics import compute_fact_counts, compute_fact_metrics
    from src.eval.nlp_hall.similarity import get_similarity_backend

    rows: list[dict[str, Any]] = []
    sim_backend = get_similarity_backend(cfg)

    for raw in records:
        try:
            parsed = parse_record(raw)
        except Exception as exc:  # noqa: BLE001
            logger.exception("parse_record failed")
            fallback = {
                "id": str(raw.get("id", "")),
                "model_key": raw.get("model_key"),
                "source": raw.get("source"),
                "modality": raw.get("modality"),
                "gt": raw.get("gt"),
                "quality_flags": "",
                "missing_modalities_pred": "",
                "missing_modalities_ref": "",
                "missing_modality_mismatch": 0,
                "is_valid": 0,
                "error": f"parse_record exception: {exc}",
            }
            if not cfg.skip_invalid:
                rows.append(_fill_null_metrics(fallback))
            continue

        base = _base_row(parsed)
        if not parsed.is_valid:
            if not cfg.skip_invalid:
                rows.append(_fill_null_metrics(base))
            continue

        try:
            counts = compute_fact_counts(
                ref_facts=parsed.ref_facts,
                pred_facts=parsed.pred_facts,
                sim_backend=sim_backend,
                cfg=cfg,
            )
            fact = compute_fact_metrics(counts)
            expl = compute_expl_metrics_routed(
                parsed=parsed,
                sim_backend=sim_backend,
                cfg=cfg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("evaluation failed for id=%s", parsed.id)
            base["is_valid"] = 0
            base["error"] = f"evaluation error: {exc}"
            if not cfg.skip_invalid:
                rows.append(_fill_null_metrics(base))
            continue

        row = {
            **base,
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
            "n_ref": counts.n_ref,
            "n_pred": counts.n_pred,
            "mean_match_sim": counts.mean_match_sim,
            "add": fact.add,
            "omit": fact.omit,
            "p": fact.p,
            "r": fact.r,
            "f1": fact.f1,
            "eg": expl.eg,
            "eh": expl.eh,
            "ec": expl.ec,
            "eqs": expl.eqs,
            "k_claims": expl.k_claims,
            "len_tokens": expl.len_tokens,
        }
        rows.append(row)

    return rows


def evaluate_jsonl(path: str | Path, cfg: HallEvalConfig):
    from src.eval.nlp_hall.aggregate import records_to_dataframe

    in_path = Path(path)
    records: list[dict[str, Any]] = []

    with in_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except Exception:  # noqa: BLE001
                item = {
                    "id": str(lineno),
                    "target_json": "{",
                    "output": "{",
                }
            if not isinstance(item, dict):
                item = {
                    "id": str(lineno),
                    "target_json": "{",
                    "output": "{",
                }
            records.append(item)

    rows = evaluate_records(records=records, cfg=cfg)
    return records_to_dataframe(rows)


def aggregate_all(df):
    from src.eval.nlp_hall.aggregate import aggregate_all as _aggregate_all

    return _aggregate_all(df)


__all__ = ["evaluate_records", "evaluate_jsonl", "aggregate_all"]
