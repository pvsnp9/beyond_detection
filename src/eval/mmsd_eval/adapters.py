from collections import defaultdict
from typing import Any, Dict, List

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.eval.mmsd_eval.io import ParsedRow
from src.utils.metrics import audit_records


THREE_WAY_LABELS = ["sarcastic", "non_sarcastic", "unknown"]


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def compute_classification(rows: List[ParsedRow]) -> Dict[str, Any]:
    raw_records = [row.raw for row in rows]
    audit = audit_records(iter(raw_records))

    y_true = audit["y_true"]
    y_pred = audit["y_pred"]
    modes = audit["modes"]

    if not y_true:
        return {
            "overall": {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0},
            "three_way": {"labels": THREE_WAY_LABELS, "per_class": {}, "confusion_matrix": []},
            "by_mode": {},
            "audit": _trim_audit(audit),
        }

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=THREE_WAY_LABELS, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": int(supports[i]),
        }
        for i, label in enumerate(THREE_WAY_LABELS)
    }
    cm = confusion_matrix(y_true, y_pred, labels=THREE_WAY_LABELS).tolist()

    by_mode_buckets: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for mode, gt, pred in zip(modes, y_true, y_pred):
        bucket = by_mode_buckets[mode]
        bucket["y_true"].append(gt)
        bucket["y_pred"].append(pred)

    by_mode: Dict[str, Dict[str, float]] = {}
    for mode, bucket in by_mode_buckets.items():
        if not bucket["y_true"]:
            continue
        by_mode[mode] = {
            "accuracy": float(accuracy_score(bucket["y_true"], bucket["y_pred"])),
            "macro_f1": float(
                f1_score(bucket["y_true"], bucket["y_pred"], average="macro", zero_division=0)
            ),
            "n": len(bucket["y_true"]),
        }

    return {
        "overall": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        },
        "three_way": {
            "labels": THREE_WAY_LABELS,
            "per_class": per_class,
            "confusion_matrix": cm,
        },
        "by_mode": by_mode,
        "audit": _trim_audit(audit),
    }


def _trim_audit(audit: Dict[str, Any]) -> Dict[str, Any]:
    drop = {"y_true", "y_pred", "modes", "labels"}
    return {k: v for k, v in audit.items() if k not in drop}


def compute_explainability(rows: List[ParsedRow]) -> Dict[str, Any]:
    from src.eval.nlp_metrics import score_pairs

    overall_refs: List[str] = []
    overall_cands: List[str] = []
    by_mode_pairs: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: {"refs": [], "cands": []}
    )

    for row in rows:
        if not isinstance(row.target, dict) or not isinstance(row.output, dict):
            continue
        ref = row.target.get("explanation")
        cand = row.output.get("explanation")
        if not isinstance(ref, str) or not isinstance(cand, str):
            continue
        if not ref.strip() or not cand.strip():
            continue
        overall_refs.append(ref)
        overall_cands.append(cand)
        mode = row.raw.get("mode", "unknown")
        by_mode_pairs[mode]["refs"].append(ref)
        by_mode_pairs[mode]["cands"].append(cand)

    overall = score_pairs(overall_refs, overall_cands)
    by_mode = {
        mode: score_pairs(pairs["refs"], pairs["cands"])
        for mode, pairs in by_mode_pairs.items()
    }
    return {
        "bert_score": {
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
            "n_pairs": len(overall_refs),
            "by_mode": by_mode,
        }
    }


def compute_farge(rows: List[ParsedRow], similarity_backend: str = "sbert") -> Dict[str, Any]:
    from config.logistics import HallEvalConfig
    from src.eval.nlp_hall import evaluate_records
    from src.eval.nlp_hall.aggregate import (
        aggregate_macro,
        aggregate_micro_facts,
        records_to_dataframe,
    )

    cfg = HallEvalConfig(backend=similarity_backend, skip_invalid=False)
    raw_records = [row.raw for row in rows]
    result_rows = evaluate_records(records=iter(raw_records), cfg=cfg)
    df = records_to_dataframe(result_rows)

    metrics_cols = ["add", "omit", "p", "r", "f1", "eg", "eh", "ec", "eqs", "k_claims", "len_tokens"]
    macro = aggregate_macro(df, metrics=metrics_cols, group_cols=[])
    micro = aggregate_micro_facts(df, group_cols=[])

    macro_row = macro.iloc[0].to_dict() if not macro.empty else {}
    micro_row = micro.iloc[0].to_dict() if not micro.empty else {}

    fact = {
        "p": _to_float(macro_row.get("p")),
        "r": _to_float(macro_row.get("r")),
        "f1": _to_float(macro_row.get("f1")),
        "add": _to_float(macro_row.get("add")),
        "omit": _to_float(macro_row.get("omit")),
        "micro_p": _to_float(micro_row.get("micro_p")),
        "micro_r": _to_float(micro_row.get("micro_r")),
        "micro_f1": _to_float(micro_row.get("micro_f1")),
        "tp": _to_int(micro_row.get("tp")),
        "fp": _to_int(micro_row.get("fp")),
        "fn": _to_int(micro_row.get("fn")),
    }
    expl = {
        "eg": _to_float(macro_row.get("eg")),
        "eh": _to_float(macro_row.get("eh")),
        "ec": _to_float(macro_row.get("ec")),
        "eqs": _to_float(macro_row.get("eqs")),
    }
    shape = {
        "k_claims": _to_float(macro_row.get("k_claims")),
        "len_tokens": _to_float(macro_row.get("len_tokens")),
    }

    by_modality_macro = aggregate_macro(df, metrics=metrics_cols, group_cols=["modality"])
    by_modality_micro = aggregate_micro_facts(df, group_cols=["modality"])
    by_modality = _merge_per_group(by_modality_macro, by_modality_micro, "modality")

    return {
        "fact": fact,
        "expl": expl,
        "shape": shape,
        "by_modality": by_modality,
        "n_valid": int(df["is_valid"].astype(bool).sum()) if "is_valid" in df.columns else len(df),
        "_df": df,
    }


def _merge_per_group(macro_df, micro_df, key: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if macro_df is None or macro_df.empty:
        return out
    macro_indexed = macro_df.set_index(key)
    micro_indexed = micro_df.set_index(key) if (micro_df is not None and not micro_df.empty) else None
    for group_value, row in macro_indexed.iterrows():
        entry = {
            "p": _to_float(row.get("p")),
            "r": _to_float(row.get("r")),
            "f1": _to_float(row.get("f1")),
            "eg": _to_float(row.get("eg")),
            "eh": _to_float(row.get("eh")),
            "ec": _to_float(row.get("ec")),
            "eqs": _to_float(row.get("eqs")),
            "n_examples": int(row.get("n_examples", 0)),
        }
        if micro_indexed is not None and group_value in micro_indexed.index:
            mrow = micro_indexed.loc[group_value]
            entry["micro_f1"] = _to_float(mrow.get("micro_f1"))
        out[str(group_value)] = entry
    return out


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f != f:  # NaN
        return 0.0
    return f


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
