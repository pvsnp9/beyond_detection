import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config.logistics import Logistics
from src.eval.classification import get_latest_result_files


def compute_bert_metrics():
    try:
        files = get_latest_result_files()
        report_dir = Path(Logistics().reports_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        out_path = report_dir / f"nlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        results: Dict[str, Dict[str, Any]] = {}
        with out_path.open("w", encoding="utf-8") as out_handle:
            for model_name, model_files in files.items():
                explanation_files = [
                    path for path in model_files if Path(path).name == "explanation.jsonl"
                ]
                if not explanation_files:
                    continue

                total_sample = 0
                parsing_failed_count = 0
                bad_pred_format_count = 0
                per_mode_pairs: Dict[str, Dict[str, List[str]]] = {}

                for file_path in explanation_files:
                    with open(file_path, "r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            total_sample += 1
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                parsing_failed_count += 1
                                continue

                            mode = record.get("mode", "unknown")
                            target_json = record.get("target_json")
                            output = record.get("output")

                            try:
                                if isinstance(target_json, str):
                                    target_json = json.loads(target_json)
                                if isinstance(output, str):
                                    output = json.loads(output)
                            except json.JSONDecodeError:
                                parsing_failed_count += 1
                                continue

                            if not isinstance(target_json, dict) or not isinstance(output, dict):
                                parsing_failed_count += 1
                                continue

                            reference = target_json.get("explanation")
                            candidate = output.get("explanation")

                            if not isinstance(reference, str):
                                parsing_failed_count += 1
                                continue
                            if not isinstance(candidate, str):
                                bad_pred_format_count += 1
                                continue

                            mode_bucket = per_mode_pairs.setdefault(
                                mode, {"references": [], "candidates": []}
                            )
                            mode_bucket["references"].append(reference)
                            mode_bucket["candidates"].append(candidate)

                bert_score_by_mode: Dict[str, Dict[str, float]] = {}
                if per_mode_pairs:
                    from bert_score import score

                    for mode, pairs in per_mode_pairs.items():
                        candidates = pairs["candidates"]
                        references = pairs["references"]
                        if not candidates:
                            continue
                        precision, recall, f1 = score(
                            candidates, references, lang="en", verbose=True
                        )
                        bert_score_by_mode[mode] = {
                            "preciosion": float(precision.mean().item()),
                            "recall": float(recall.mean().item()),
                            "f1": float(f1.mean().item()),
                        }

                payload = {
                    "model": model_name,
                    "total_sample": total_sample,
                    "parsing_failed_count": parsing_failed_count,
                    "bad_pred_fromat_count": bad_pred_format_count,
                    "bert_score_by_mode": bert_score_by_mode,
                }
                out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                results[model_name] = payload

        print(results)
        return results
    except Exception as e:
        print(f"failed to compute nlp metrics: {e}")    
        raise e


if __name__ == "__main__":
    compute_bert_metrics()
