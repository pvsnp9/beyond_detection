import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


REQUIRED_SCHEMA = {
    "need_explanation": bool,
    "visual_facts": list,
    "evidence_fact_ids": list,
    "text_literal": str,
    "incongruity": str,
    "label": str,
    "explanation": str,
    "missing_modalities": list,
}


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def _to_float_map(labels, values):
    return {label: float(value) for label, value in zip(labels, values)}


def _is_schema_valid(output):
    if not isinstance(output, dict):
        return False

    for key, expected_type in REQUIRED_SCHEMA.items():
        if key not in output or not isinstance(output[key], expected_type):
            return False
    return True


def _has_constraint_violation(output):
    visual_facts = output.get("visual_facts")
    evidence_fact_ids = output.get("evidence_fact_ids")

    if not isinstance(visual_facts, list) or not isinstance(evidence_fact_ids, list):
        return True

    fact_ids = []
    for fact in visual_facts:
        if not isinstance(fact, dict):
            return True
        fact_ids.append(fact.get("id"))

    is_subset = all(eid in fact_ids for eid in evidence_fact_ids)
    is_consecutive = fact_ids == list(range(1, len(fact_ids) + 1))
    return not (is_subset and is_consecutive)


def _is_modality_aware_success(modality, missing_modalities):
    if not isinstance(missing_modalities, list):
        return False

    missing_set = set(missing_modalities)
    if modality == "text":
        return missing_set == {"image"}
    if modality == "image":
        return missing_set == {"text"}
    if modality == "both":
        return len(missing_modalities) == 0
    return False


def run_sarcasm_audit(jsonl_path):
    metrics = {
        "total_sample": 0,
        "total_image_only_sample": 0,
        "total_text_only_sample": 0,
        "parse_success": 0,
        "schema_valid": 0,
        "constraint_violation": 0,
        "modality_aware_success": 0,
        "unimodal_correct": 0,
        "unimodal_total": 0,
        "y_true": [],
        "y_pred": [],
    }

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            metrics["total_sample"] += 1
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue

            gt_label = item.get("gt")
            modality = item.get("modality")
            if modality == "image":
                metrics["total_image_only_sample"] += 1
            elif modality == "text":
                metrics["total_text_only_sample"] += 1

            try:
                output_raw = item.get("output")
                output = json.loads(output_raw)
                if not isinstance(output, dict):
                    continue
                metrics["parse_success"] += 1
            except (json.JSONDecodeError, TypeError):
                continue

            if _is_schema_valid(output):
                metrics["schema_valid"] += 1

            if _has_constraint_violation(output):
                metrics["constraint_violation"] += 1

            if _is_modality_aware_success(modality, output.get("missing_modalities")):
                metrics["modality_aware_success"] += 1

            pred_label = output.get("label")
            gt_label_str = str(gt_label) if gt_label is not None else "__missing_gt__"
            pred_label_str = pred_label if isinstance(pred_label, str) else "__invalid_label__"
            metrics["y_true"].append(gt_label_str)
            metrics["y_pred"].append(pred_label_str)

            if modality in ["text", "image"]:
                metrics["unimodal_total"] += 1
                if pred_label_str == gt_label_str:
                    metrics["unimodal_correct"] += 1

    parsed_count = metrics["parse_success"]
    unique_labels = sorted(set(metrics["y_true"]) | set(metrics["y_pred"]))
    if unique_labels:
        f1_scores = f1_score(
            metrics["y_true"],
            metrics["y_pred"],
            average=None,
            labels=unique_labels,
            zero_division=0,
        )
        precision_scores = precision_score(
            metrics["y_true"],
            metrics["y_pred"],
            average=None,
            labels=unique_labels,
            zero_division=0,
        )
        recall_scores = recall_score(
            metrics["y_true"],
            metrics["y_pred"],
            average=None,
            labels=unique_labels,
            zero_division=0,
        )
        macro_f1 = f1_score(metrics["y_true"], metrics["y_pred"], average="macro", zero_division=0)
        accuracy = accuracy_score(metrics["y_true"], metrics["y_pred"])
    else:
        f1_scores = []
        precision_scores = []
        recall_scores = []
        macro_f1 = 0.0
        accuracy = 0.0

    return {
        "total_sample": metrics["total_sample"],
        "total_image_only_sample": metrics["total_image_only_sample"],
        "total_text_only_sample": metrics["total_text_only_sample"],
        "parse_success_rate": float(
            _safe_divide(metrics["parse_success"], metrics["total_sample"])
        ),
        "schema_validation_rate": float(_safe_divide(metrics["schema_valid"], parsed_count)),
        "constraint_violation_rate": float(
            _safe_divide(metrics["constraint_violation"], parsed_count)
        ),
        "modality_aware_rate": float(
            _safe_divide(metrics["modality_aware_success"], parsed_count)
        ),
        "macro_f1": float(macro_f1),
        "f1_by_class": _to_float_map(unique_labels, f1_scores),
        "precision_by_class": _to_float_map(unique_labels, precision_scores),
        "recall_by_class": _to_float_map(unique_labels, recall_scores),
        "accuracy": float(accuracy),
        "unimodal_accuracy": float(
            _safe_divide(metrics["unimodal_correct"], metrics["unimodal_total"])
        ),
    }
