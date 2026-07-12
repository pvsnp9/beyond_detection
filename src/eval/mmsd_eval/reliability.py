from collections import defaultdict
from typing import Any, Dict, Iterable, List

from src.eval.mmsd_eval.io import ParsedRow


MODALITY_LABELS = ("text", "image")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def compute_psr(rows: List[ParsedRow]) -> Dict[str, Any]:
    total = len(rows)
    failure_breakdown: Dict[str, int] = defaultdict(int)
    parseable = 0
    for row in rows:
        if row.parse_status == "ok":
            parseable += 1
        else:
            failure_breakdown[row.parse_status] += 1

    return {
        "parseable": parseable,
        "total": total,
        "rate": _safe_div(parseable, total),
        "failure_breakdown": dict(failure_breakdown),
    }


def _modality_set(value: Any) -> set:
    if not isinstance(value, list):
        return set()
    out: set = set()
    for item in value:
        if isinstance(item, str) and item in MODALITY_LABELS:
            out.add(item)
    return out


def compute_mar(rows: List[ParsedRow]) -> Dict[str, Any]:
    exact_match = 0
    evaluated = 0
    by_mode_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"match": 0, "total": 0})

    tp = fp = fn = 0  # micro over modality labels

    per_example: List[Dict[str, Any]] = []

    for row in rows:
        target = row.target
        output = row.output
        if not isinstance(target, dict) or not isinstance(output, dict):
            per_example.append({"id": row.raw.get("id"), "evaluated": False, "match": None})
            continue

        ref = _modality_set(target.get("missing_modalities"))
        pred = _modality_set(output.get("missing_modalities"))
        evaluated += 1
        is_match = ref == pred
        if is_match:
            exact_match += 1
        tp += len(ref & pred)
        fp += len(pred - ref)
        fn += len(ref - pred)

        mode = row.raw.get("mode", "unknown")
        bucket = by_mode_counts[mode]
        bucket["total"] += 1
        if is_match:
            bucket["match"] += 1

        per_example.append({"id": row.raw.get("id"), "evaluated": True, "match": is_match})

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    by_mode = {
        mode: {
            "exact_set_match_rate": _safe_div(counts["match"], counts["total"]),
            "n": counts["total"],
        }
        for mode, counts in by_mode_counts.items()
    }

    return {
        "exact_set_match_rate": _safe_div(exact_match, evaluated),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_evaluated": evaluated,
        "by_mode": by_mode,
        "_per_example": per_example,
    }
