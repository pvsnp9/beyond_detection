import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
from config.logistics import Logistics, ModelCards
from src.eval.classification import get_latest_result_files
from transformers import AutoProcessor


def compute_bert_metrics():
    try:
        files = get_latest_result_files()
        report_dir = Path(os.path.join(Logistics().reports_dir, "expl"))
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
                    print(f"Processing BERT score for model: {model_name}, file: {file_path}")
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


def compute_malformed_outputs(files: Optional[Dict[str, List[str]]] = None):
    try:
        files = files or get_latest_result_files()
        model_cards = ModelCards()
        base_model_map = {
            "llama": model_cards.llama3_2vl,
            "aya": model_cards.aya_model_name,
            "gemma": model_cards.gemm3_12b,
            "qwen": model_cards.qwen3_vl_8b_instruct,
        }

        for model_name, model_files in files.items():
            base_model_id = base_model_map.get(model_name)
            if base_model_id is None:
                raise ValueError(f"Unknown model name: {model_name}")

            try:
                processor = AutoProcessor.from_pretrained(
                    base_model_id, trust_remote_code=True, use_fast=True
                )
            except Exception as exc:
                print(f"AutoProcessor(use_fast=True) failed, retrying use_fast=False: {exc}")
                processor = AutoProcessor.from_pretrained(
                    base_model_id, trust_remote_code=True, use_fast=False
                )

            tokenizer = getattr(processor, "tokenizer", processor)
            result_dir = Path(Logistics().results_dir) / f"{model_name}/malformed_outputs"
            result_dir.mkdir(parents=True, exist_ok=True)

            for file_path in model_files:
                out_path = result_dir / Path(file_path).name
                with open(file_path, "r", encoding="utf-8") as handle, out_path.open(
                    "w", encoding="utf-8"
                ) as out_handle:
                    print(f"Processing malformed outputs for model: {model_name}, file: {file_path}")
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as exc:
                            print(f"malformed_line: {exc}")
                            continue

                        mode = record.get("mode", "unknown")
                        record_id = record.get("id", "unknown")
                        output = record.get("output")
                        if not isinstance(output, str):
                            continue

                        try:
                            json.loads(output)
                            continue
                        except json.JSONDecodeError as exc:
                            error = f"invalid_output_json: {exc}"

                        token_length: Optional[int]
                        try:
                            encoded = tokenizer(output, add_special_tokens=False)
                            input_ids = (
                                encoded.get("input_ids")
                                if isinstance(encoded, dict)
                                else getattr(encoded, "input_ids", None)
                            )
                            if input_ids is None:
                                token_length = 0
                            elif hasattr(input_ids, "shape"):
                                token_length = int(input_ids.shape[-1])
                            elif isinstance(input_ids, list) and input_ids and isinstance(
                                input_ids[0], list
                            ):
                                token_length = len(input_ids[0])
                            else:
                                token_length = len(input_ids)
                        except Exception as exc:
                            token_length = None
                            error = f"{error}; tokenization_failed: {exc}"

                        payload = {
                            "model_name": model_name,
                            "mode": mode,
                            "id": record_id,
                            "token_length": token_length,
                            "output": output,
                            "error": error,
                        }
                        out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    print(f"Saved malformed output report to: {out_path}")
    except Exception as e:
        print(f"failed to compute malformed output metrics: {e}")    
        raise e

# if __name__ == "__main__":
#     # compute_bert_metrics()
#     compute_malformed_outputs()
