import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Union

from config.logistics import Logistics
from src.eval.mmsd_eval.io import load_jsonl, parse_path_triple, parse_row


METRIC_KEYS = {"classification", "reliability", "bert", "farge"}


def _resolve_out_dir(input_jsonl: Path, out_dir: Optional[Union[str, Path]]) -> tuple[Path, Dict[str, Optional[str]]]:
    triple = parse_path_triple(input_jsonl) or {}
    if out_dir is not None:
        return Path(out_dir), triple

    logistics = Logistics()
    base = Path(logistics.project_root_dir) / logistics.reports_dir / "mmsd"
    model = triple.get("model") or "unknown_model"
    type_ = triple.get("type") or "unknown_type"
    run_mode = triple.get("run_mode") or "default"
    return base / model / type_ / run_mode, triple


def _write_summary_txt(out_path: Path, metrics: Dict[str, Any]) -> None:
    meta = metrics["meta"]
    cls = metrics.get("classification", {})
    rel = metrics.get("reliability", {})
    expl = metrics.get("explainability", {})
    farge = metrics.get("farge", {})

    lines = []
    lines.append(f"mmsd_eval report  model={meta['model']}  type={meta['type']}  run_mode={meta['run_mode']}")
    lines.append(f"source: {meta['source_jsonl']}")
    lines.append(f"n_total={meta['n_total']}  n_parseable={meta['n_parseable']}")
    lines.append("")

    if cls:
        overall = cls.get("overall", {})
        lines.append("Classification (three-way: sarcastic / non_sarcastic / unknown)")
        lines.append(
            f"  accuracy={overall.get('accuracy', 0):.4f}  macro_f1={overall.get('macro_f1', 0):.4f}  weighted_f1={overall.get('weighted_f1', 0):.4f}"
        )
        for label, stats in cls.get("three_way", {}).get("per_class", {}).items():
            lines.append(
                f"    {label}: P={stats['precision']:.4f}  R={stats['recall']:.4f}  F1={stats['f1']:.4f}  n={stats['support']}"
            )
        for mode, stats in cls.get("by_mode", {}).items():
            lines.append(
                f"  mode={mode}: acc={stats['accuracy']:.4f}  macro_f1={stats['macro_f1']:.4f}  n={stats['n']}"
            )
        lines.append("")

    if rel:
        psr = rel.get("psr", {})
        mar = rel.get("mar", {})
        lines.append("Reliability")
        lines.append(
            f"  PSR (parsing success rate) = {psr.get('rate', 0):.4f}  ({psr.get('parseable', 0)}/{psr.get('total', 0)})"
        )
        if psr.get("failure_breakdown"):
            for bucket, count in psr["failure_breakdown"].items():
                lines.append(f"    {bucket}: {count}")
        lines.append(
            f"  MAR (modality aware rate) = {mar.get('exact_set_match_rate', 0):.4f}  set-F1={mar.get('f1', 0):.4f}  n={mar.get('n_evaluated', 0)}"
        )
        for mode, stats in mar.get("by_mode", {}).items():
            lines.append(f"    mode={mode}: exact={stats['exact_set_match_rate']:.4f}  n={stats['n']}")
        lines.append("")

    if expl:
        bert = expl.get("bert_score", {})
        lines.append("Explainability (BERTScore)")
        lines.append(
            f"  P={bert.get('precision', 0):.4f}  R={bert.get('recall', 0):.4f}  F1={bert.get('f1', 0):.4f}  n_pairs={bert.get('n_pairs', 0)}"
        )
        for mode, stats in bert.get("by_mode", {}).items():
            lines.append(f"    mode={mode}: F1={stats['f1']:.4f}")
        lines.append("")

    if farge:
        fact = farge.get("fact", {})
        e = farge.get("expl", {})
        lines.append("FaRGE")
        lines.append(
            f"  fact P={fact.get('p', 0):.4f}  R={fact.get('r', 0):.4f}  F1={fact.get('f1', 0):.4f}  micro_F1={fact.get('micro_f1', 0):.4f}"
        )
        lines.append(
            f"  expl eg={e.get('eg', 0):.4f}  eh={e.get('eh', 0):.4f}  ec={e.get('ec', 0):.4f}  eqs={e.get('eqs', 0):.4f}"
        )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_per_example(
    out_path: Path,
    rows,
    mar_per_example,
    farge_df,
) -> None:
    mar_by_id = {item["id"]: item for item in mar_per_example} if mar_per_example else {}
    farge_by_id: Dict[Any, Dict[str, Any]] = {}
    if farge_df is not None and "id" in getattr(farge_df, "columns", []):
        for _, fr in farge_df.iterrows():
            farge_by_id[fr["id"]] = fr.to_dict()

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            record_id = row.raw.get("id")
            gt = row.raw.get("gt")
            pred_label = None
            if isinstance(row.output, dict):
                pred_label = row.output.get("label")
            payload = {
                "id": record_id,
                "mode": row.raw.get("mode"),
                "modality": row.raw.get("modality"),
                "gt": gt,
                "pred_label": pred_label,
                "parse_status": row.parse_status,
                "label_correct": (pred_label == gt) if (pred_label and gt) else None,
                "mar_match": (mar_by_id.get(record_id, {}) or {}).get("match"),
            }
            farge_row = farge_by_id.get(record_id)
            if farge_row is not None:
                for col in ("p", "r", "f1", "eg", "eh", "ec", "eqs"):
                    payload[f"farge_{col}"] = _coerce(farge_row.get(col))
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _coerce(value: Any) -> Any:
    try:
        import math

        if value is None:
            return None
        f = float(value)
        if math.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return value


def _strip_internal(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(metrics)
    farge = out.get("farge")
    if isinstance(farge, dict) and "_df" in farge:
        farge = {k: v for k, v in farge.items() if k != "_df"}
        out["farge"] = farge
    rel = out.get("reliability")
    if isinstance(rel, dict):
        mar = rel.get("mar")
        if isinstance(mar, dict) and "_per_example" in mar:
            rel = {**rel, "mar": {k: v for k, v in mar.items() if k != "_per_example"}}
            out["reliability"] = rel
    return out


def run(
    input_jsonl: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    *,
    skip: Optional[Iterable[str]] = None,
    similarity_backend: str = "sbert",
) -> Dict[str, Any]:
    from src.eval.mmsd_eval.adapters import (
        compute_classification,
        compute_explainability,
        compute_farge,
    )
    from src.eval.mmsd_eval.reliability import compute_mar, compute_psr

    skip_set: Set[str] = set(skip or [])
    unknown = skip_set - METRIC_KEYS
    if unknown:
        raise ValueError(f"Unknown skip keys: {unknown}. Valid: {sorted(METRIC_KEYS)}")

    input_path = Path(input_jsonl).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    raw_records = load_jsonl(input_path)
    rows = [parse_row(rec) for rec in raw_records]

    out_path, triple = _resolve_out_dir(input_path, out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    psr = compute_psr(rows)
    metrics: Dict[str, Any] = {
        "meta": {
            "model": triple.get("model"),
            "type": triple.get("type"),
            "run_mode": triple.get("run_mode"),
            "run_timestamp": triple.get("timestamp"),
            "source_jsonl": str(input_path),
            "n_total": len(rows),
            "n_parseable": psr["parseable"],
            "eval_timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    }

    if "classification" not in skip_set:
        print("[mmsd_eval] computing classification metrics...")
        metrics["classification"] = compute_classification(rows)

    if "reliability" not in skip_set:
        print("[mmsd_eval] computing reliability metrics (PSR, MAR)...")
        mar = compute_mar(rows)
        metrics["reliability"] = {"psr": psr, "mar": mar}

    if "bert" not in skip_set:
        print("[mmsd_eval] computing BERTScore...")
        metrics["explainability"] = compute_explainability(rows)

    farge_df = None
    if "farge" not in skip_set:
        print("[mmsd_eval] computing FaRGE metrics...")
        farge = compute_farge(rows, similarity_backend=similarity_backend)
        farge_df = farge.get("_df")
        metrics["farge"] = farge

    serializable = _strip_internal(metrics)
    metrics_path = out_path / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    mar_per_example = (
        metrics.get("reliability", {}).get("mar", {}).get("_per_example")
        if "reliability" not in skip_set
        else None
    )
    _write_per_example(out_path / "per_example.jsonl", rows, mar_per_example, farge_df)
    _write_summary_txt(out_path / "summary.txt", serializable)

    print(f"[mmsd_eval] wrote report to {out_path}")
    return serializable
