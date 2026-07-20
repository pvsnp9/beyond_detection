import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.eval.parsing import (
    discover_latest_results,
    iter_records,
    parse_freeform_output,
    parse_rich_freeform_output,
    parse_structured_output,
    reports_root,
    results_root,
)

BINARY_LABELS = ("sarcastic", "non_sarcastic")
MISSING_TARGET = {"both": frozenset(), "text": frozenset({"image"}), "image": frozenset({"text"})}


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    # paper convention: 0.0 whenever a denominator is zero
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": p, "r": r, "f1": f1}


def evaluate_file(path: Path, gen_type: str) -> Dict[str, Any]:
    total = 0
    modality_count = {"both": 0, "image": 0, "text": 0}
    parsing_failed = 0
    bad_format = 0
    pred_unknown = 0
    both_pairs: list[tuple[str, str]] = []  # (gt, pred) over parsed both-modality rows
    uni_n = {"text": 0, "image": 0}
    uni_correct = {"text": 0, "image": 0}
    mar_hits = 0
    mar_n = 0

    for rec in iter_records(path):
        total += 1
        modality = rec["modality"]
        modality_count[modality] += 1

        # freeform has no payload (label-only); rich and structured recover the full schema
        if gen_type == "freeform_sft":
            pred, _, status = parse_freeform_output(rec["output"])
            payload = None
        elif gen_type == "rich_freeform_sft":
            payload, status = parse_rich_freeform_output(rec["output"])
            pred = payload["label"] if status == "ok" else None
        else:
            payload, status = parse_structured_output(rec["output"])
            pred = payload["label"] if status == "ok" else None
        if status == "parsing_failed":
            parsing_failed += 1
            continue
        if status == "bad_format":
            bad_format += 1
            continue

        if payload is not None:
            mar_n += 1
            predicted_missing = {str(m).strip().lower() for m in payload["missing_modalities"]}
            mar_hits += predicted_missing == MISSING_TARGET[modality]

        if modality == "both":
            both_pairs.append((rec["gt"], pred))
            pred_unknown += pred == "unknown"
        else:
            uni_n[modality] += 1
            uni_correct[modality] += pred == "unknown"

    n_ok = total - parsing_failed - bad_format
    n_both = len(both_pairs)

    per_class = {}
    for cls in BINARY_LABELS:
        tp = sum(1 for gt, pred in both_pairs if gt == cls and pred == cls)
        fp = sum(1 for gt, pred in both_pairs if gt != cls and pred == cls)
        fn = sum(1 for gt, pred in both_pairs if gt == cls and pred != cls)
        per_class[cls] = _prf(tp, fp, fn)
    macro = {k: sum(per_class[c][k] for c in BINARY_LABELS) / len(BINARY_LABELS) for k in ("p", "r", "f1")}

    n_uni = uni_n["text"] + uni_n["image"]
    metrics = {
        "both": {
            "n_evaluated": n_both,
            "pred_unknown_count": pred_unknown,
            "accuracy": sum(1 for gt, pred in both_pairs if gt == pred) / n_both if n_both else None,
            "per_class": per_class,
            "macro": macro,
        },
        "unimodal": {
            "n_text": uni_n["text"],
            "n_image": uni_n["image"],
            "text_accuracy": uni_correct["text"] / uni_n["text"] if uni_n["text"] else None,
            "image_accuracy": uni_correct["image"] / uni_n["image"] if uni_n["image"] else None,
            "overall_accuracy": (uni_correct["text"] + uni_correct["image"]) / n_uni if n_uni else None,
        },
        "psr": n_ok / total if total else None,
        "mar": mar_hits / mar_n if mar_n else None,
    }
    return {
        "total_samples": total,
        "modality_count": modality_count,
        "parsing_failed_count": parsing_failed,
        "bad_format_count": bad_format,
        "n_ok": n_ok,
        "metrics": metrics,
    }


def main() -> None:
    files = discover_latest_results()
    if not files:
        raise FileNotFoundError(f"no result files discovered under {results_root()}")

    rows = []
    for (model, gen_type, split), path in files.items():
        print(f"classification: {model}/{gen_type}/{split} <- {path}")
        body = evaluate_file(path, gen_type)
        rows.append({"model": model, "gen_type": gen_type, "split": split, "source_file": str(path), **body})

    report_dir = reports_root()
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = report_dir / f"CLS_{ts}.json"
    report = {"generated_at": ts, "results_root": str(results_root()), "rows": rows}
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
