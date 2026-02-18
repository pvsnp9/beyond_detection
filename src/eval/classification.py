from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os
import json
from config.logistics import Logistics


def _parse_timestamp_dir(name: str) -> Optional[float]:
    try:
        return datetime.strptime(name, "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return None


def _latest_timestamp_dir(model_dir: Path) -> Optional[Path]:
    subdirs = [path for path in model_dir.iterdir() if path.is_dir()]
    if not subdirs:
        return None

    parsed = [(path, _parse_timestamp_dir(path.name)) for path in subdirs]
    parsed_dirs = [item for item in parsed if item[1] is not None]
    if parsed_dirs:
        return max(parsed_dirs, key=lambda item: item[1])[0]

    return max(subdirs, key=lambda path: path.stat().st_mtime)


def get_latest_result_files(
    results_dir: Optional[Union[str, Path]] = None
) -> Dict[str, List[str]]:
    root = Path(results_dir or Logistics().results_dir)
    if not root.exists():
        return {}

    result: Dict[str, List[str]] = {}
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        latest_dir = _latest_timestamp_dir(model_dir)
        if latest_dir is None:
            continue
        files = sorted(str(path) for path in latest_dir.glob("*.jsonl"))
        result[model_dir.name] = files

    return result

def merge_files(input_files: List[str]) -> None:
    try:
        if not input_files:
            raise ValueError("input_files must contain at least one file path")
        output_dir = Path(input_files[0]).parent
        output_file = str(output_dir / "merged.jsonl")
        print(f"Starting merge from {len(input_files)} files into {output_file}")
        total_lines = 0
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as out_handle:
            for input_path in input_files:
                with open(input_path, "r", encoding="utf-8") as in_handle:
                    for line in in_handle:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        out_handle.write(line + "\n")
                        total_lines += 1
        print(
            f"Completed merging into {output_file}. Total lines merged: {total_lines}"
        )
    except Exception as e:
        print(f"Exception occured while merging files: {e}")
        raise e
    

def compute_f1(files: Optional[Dict[str, List[str]]] = None) -> Dict[str, Dict[str, Any]]:
    try:
        files = files or {}
        report_dir = Path(os.path.join(Logistics().reports_dir, "cls"))
        report_dir.mkdir(parents=True, exist_ok=True)
        out_path = report_dir / f"f1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        label_map = {"sarcastic": 1, "non_sarcastic": 0}
        results: Dict[str, Dict[str, Any]] = {}

        with out_path.open("w", encoding="utf-8") as out_handle:
            for model_name, model_files in files.items():
                per_task_payload: Dict[str, Dict[str, Any]] = {}

                for file_path in model_files:
                    # only process merged files,
                    if file_path.endswith("explanation.jsonl"): #or file_path.endswith("/merged.jsonl"):
                        continue
                    print(f"Processing file: {file_path} for model: {model_name}\n")
                    task = Path(file_path).stem
                    total_sample = 0
                    parsing_failed_count = 0
                    bad_pred_format_count = 0
                    per_mode_counts: Dict[str, Dict[str, int]] = {}

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

                            ground_truth = target_json.get("label")
                            pred_label = output.get("lable", output.get("label"))
                            if pred_label is None:
                                bad_pred_format_count += 1
                                continue

                            if ground_truth not in label_map:
                                parsing_failed_count += 1
                                continue
                            if pred_label not in label_map:
                                bad_pred_format_count += 1
                                continue

                            gt = label_map[ground_truth]
                            pred = label_map[pred_label]
                            counts = per_mode_counts.setdefault(
                                mode, {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                            )
                            if pred == 1 and gt == 1:
                                counts["tp"] += 1
                            elif pred == 1 and gt == 0:
                                counts["fp"] += 1
                            elif pred == 0 and gt == 1:
                                counts["fn"] += 1
                            elif pred == 0 and gt == 0:
                                counts["tn"] += 1

                    f1_by_mode: Dict[str, float] = {}
                    precision_by_mode: Dict[str, float] = {}
                    reacall_by_mode: Dict[str, float] = {}
                    accuracy_by_mode: Dict[str, float] = {}
                    for mode, counts in per_mode_counts.items():
                        tp = counts["tp"]
                        fp = counts["fp"]
                        fn = counts["fn"]
                        tn = counts["tn"]
                        denom = (2 * tp) + fp + fn
                        f1_by_mode[mode] = (2 * tp / denom) if denom else 0.0
                        precision_denom = tp + fp
                        recall_denom = tp + fn
                        accuracy_denom = tp + fp + fn + tn
                        precision_by_mode[mode] = tp / precision_denom if precision_denom else 0.0
                        reacall_by_mode[mode] = tp / recall_denom if recall_denom else 0.0
                        accuracy_by_mode[mode] = (tp + tn) / accuracy_denom if accuracy_denom else 0.0

                    per_task_payload[task] = {
                        "total_sample": total_sample,
                        "parsing_failed_count": parsing_failed_count,
                        "bad_pred_fromat_count": bad_pred_format_count,
                        "f1_by_mode": f1_by_mode,
                        "precision_by_mode": precision_by_mode,
                        "reacall_by_mode": reacall_by_mode,
                        "accuracy_by_mode": accuracy_by_mode,
                    }

                payload = {model_name: per_task_payload}
                out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                results[model_name] = per_task_payload
        print(results)
        return results
    except Exception as e:
        print(f"Exception occured while computing F1-score: {e}")
        raise e



if __name__ == "__main__":
    gen_files = get_latest_result_files(os.path.join(Logistics().project_root_dir, Logistics().results_dir))
    # #code for merging explanation and detection_explanation files into one detection_explanation file
    for model_name, files in gen_files.items():
        print(f"Processing model: {model_name} with files: {files}")
        if len(files) >= 2:
            merge_files(files)
        else: print(f"Not enough files to merge for model: {model_name}")
        print("-----"*20)
    
    updated_files= get_latest_result_files(os.path.join(Logistics().project_root_dir, Logistics().results_dir))
    compute_f1(updated_files)
