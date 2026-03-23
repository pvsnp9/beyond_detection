import json
from datetime import datetime
import torch
from transformers import (
    AutoProcessor, 
    MllamaForConditionalGeneration,      # Llama 3.2
    AyaVisionForConditionalGeneration,   # Aya
    Gemma3ForConditionalGeneration,      # Gemma 3
    AutoModelForImageTextToText           # Generic Fallback
)
import os
import time
from PIL import Image
from peft import PeftModel
from config.logistics import Logistics
from config.queries import Queries
from src.collators.inference_collator import (aya_inference_collator,
                                            qwen_inference_collator,
                                            gemma_inference_collator,
                                            llama_inference_collator )
from src.datasets.loader import load_hf_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.sft_utils import get_sft_result_dir


VALID_MODALITIES = {"both", "text", "image"}
VALID_MODEL_TYPES = {"sft", "mdpo"}
MDPO_MODEL_KEYS = {"llama", "qwen"}

MODEL_DIR_CANDIDATES = {
    "sft": {
        "aya": ["aya_vision_8b"],
        "llama": ["llama32"],
        "gemma": ["gemma3"],
        "qwen": ["qwen3_vl_8b", "qwen3"],
    },
    "mdpo": {
        "llama": ["llama32"],
        "qwen": ["qwen3", "qwen3_vl_8b"],
    },
}


class BaseModels:
    def __init__(self):
        self.models = {
            "llama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "aya": "CohereLabs/aya-vision-8b",
            "gemma": "google/gemma-3-12b-it",
            "qwen": "Qwen/Qwen3-VL-8B-Instruct"
        }

    
    def get_class(self, model_key):
        qwen_class = AutoModelForImageTextToText
        if model_key == "qwen":
            from transformers import Qwen3VLForConditionalGeneration
            qwen_class = Qwen3VLForConditionalGeneration
        mapping = {
            "llama": MllamaForConditionalGeneration,
            "aya": AyaVisionForConditionalGeneration,
            "gemma": Gemma3ForConditionalGeneration,
            "qwen": qwen_class,
        }
        return mapping.get(model_key, AutoModelForImageTextToText)


def get_test_data(config_name: str = "en"):
    dataset_id = None
    try:
        dataset_id = Logistics().hf_sft_ds_id
        ds = load_hf_dataset(
            dataset_id,
            split="test",
            config_name=config_name,
            # download_mode="force_redownload",
        )
        return ds
    except Exception as e:
        ds_name = dataset_id or "<unknown>"
        raise Exception(f"failed to load the test dataset {ds_name}: {e}")


def _load_redeval_ood_rows(input_path: str) -> tuple[list[dict], dict]:
    rows = []
    stats = {"data_load_failures": 0, "image_load_failures": 0}
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    required_fields = ("id", "modality", "label_gt")
    with open(input_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["data_load_failures"] += 1
                continue

            if not isinstance(row, dict):
                stats["data_load_failures"] += 1
                continue

            missing_required = False
            for field in required_fields:
                if row.get(field) is None:
                    stats["data_load_failures"] += 1
                    missing_required = True
                    break
            if missing_required:
                continue

            try:
                modality = _normalize_modality(row.get("modality"))
            except Exception:
                stats["data_load_failures"] += 1
                continue
            row["modality"] = modality
            if modality in {"both", "image"}:
                image_path = row.get("image")
                if not isinstance(image_path, str) or not image_path.strip():
                    stats["image_load_failures"] += 1
                    continue
                if not os.path.isfile(image_path):
                    stats["image_load_failures"] += 1
                    continue

            rows.append(row)
    return rows, stats


def _resolve_model_dir(model_key: str, model_type: str) -> str:
    logistics = Logistics()
    root_dir = os.path.join(
        logistics.project_root_dir,
        logistics.models_output_dir,
        model_type,
    )
    candidates = MODEL_DIR_CANDIDATES[model_type].get(model_key, [])
    for subdir in candidates:
        full_path = os.path.join(root_dir, subdir)
        if os.path.isdir(full_path):
            return full_path
    fallback = candidates[0] if candidates else model_key
    return os.path.join(root_dir, fallback)


def _get_best_model(model_dir: str) -> dict:
    result = {"best_metric": None, "best_model_path": None}
    try:
        if not os.path.isdir(model_dir):
            return result

        checkpoints = []
        for name in os.listdir(model_dir):
            if not name.startswith("checkpoint-"):
                continue
            step = name.split("-")[-1]
            if step.isdigit():
                checkpoints.append((int(step), os.path.join(model_dir, name)))

        if not checkpoints:
            return result

        latest_checkpoint = max(checkpoints, key=lambda item: item[0])[1]
        state_path = os.path.join(latest_checkpoint, "train_state.json")
        if not os.path.isfile(state_path):
            state_path = os.path.join(latest_checkpoint, "trainer_state.json")

        if not os.path.isfile(state_path):
            result["best_model_path"] = latest_checkpoint
            return result

        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)

        result["best_metric"] = state.get("best_metric")
        best_ckpt = state.get("best_model_checkpoint", latest_checkpoint)
        if isinstance(best_ckpt, str) and best_ckpt and not os.path.isabs(best_ckpt):
            best_ckpt = os.path.join(model_dir, best_ckpt)
        result["best_model_path"] = best_ckpt
        return result
    except Exception:
        return result


def _load_processor_for_inference(base_model_id: str, checkpoint_path: str = None):
    candidates = []
    if checkpoint_path:
        candidates.append(checkpoint_path)
        parent_dir = os.path.dirname(checkpoint_path)
        if parent_dir and parent_dir not in candidates:
            candidates.append(parent_dir)
    else:
        candidates.append(base_model_id)

    last_error = None
    for source in candidates:
        try:
            print(f"Loading processor from: {source}")
            processor = AutoProcessor.from_pretrained(
                source,
                trust_remote_code=True,
                use_fast=True,
            )
            if getattr(processor, "tokenizer", None) is not None:
                if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
                    processor.tokenizer.pad_token = processor.tokenizer.eos_token
            return processor
        except Exception as exc:
            last_error = exc
            print(f"Failed to load processor from {source}: {exc}")

    raise RuntimeError(
        f"Failed to load processor from candidates={candidates}; last_error={last_error}"
    )


def _load_model_for_inference(
    model_class,
    base_model_id: str,
    checkpoint_path: str = None,
    merge_adapter: bool = True,
):
    if checkpoint_path:
        try:
            model = model_class.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load base model for checkpoint mode (base={base_model_id}): {exc}"
            ) from exc

        try:
            print(f"Loading Adapter: {checkpoint_path}")
            model = PeftModel.from_pretrained(model, checkpoint_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load adapter checkpoint: {checkpoint_path}: {exc}"
            ) from exc

        if merge_adapter and hasattr(model, "merge_and_unload"):
            try:
                print("Merging adapter into base model for inference...")
                model = model.merge_and_unload()
                print("Adapter merge completed.")
            except Exception as exc:
                print(f"Adapter merge failed; continuing with unmerged adapter: {exc}")
        return model

    model = model_class.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def generate_batch(
    model,
    model_inputs,
    processor,
    max_tokens=256,
):
    try:
        inputs = model_inputs.to(model.device)
        print("Generating response (deterministic structured decode)...")
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "repetition_penalty": 1.05,
        }
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **generation_kwargs,
            )
        generated_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)]
        outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)

        invalid_indices = []
        for idx, text in enumerate(outputs):
            try:
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    invalid_indices.append(idx)
            except Exception:
                invalid_indices.append(idx)

        if invalid_indices:
            print(
                "Retrying deterministic decode for malformed JSON outputs: "
                f"{len(invalid_indices)}/{len(outputs)}"
            )
            deterministic_kwargs = {"max_new_tokens": max_tokens, "do_sample": False}
            with torch.no_grad():
                deterministic_ids = model.generate(
                    **inputs,
                    **deterministic_kwargs,
                )
            deterministic_generated = [
                out[len(ins):] for ins, out in zip(inputs.input_ids, deterministic_ids)
            ]
            deterministic_outputs = processor.batch_decode(
                deterministic_generated, skip_special_tokens=True
            )
            for idx in invalid_indices:
                outputs[idx] = deterministic_outputs[idx]

        still_invalid = []
        for idx, text in enumerate(outputs):
            try:
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    still_invalid.append(idx)
            except Exception:
                still_invalid.append(idx)

        if still_invalid:
            print(
                "Forcing fallback JSON object for outputs still malformed after retry: "
                f"{len(still_invalid)}"
            )
            for idx in still_invalid:
                outputs[idx] = "{}"

        return outputs

    except Exception as e:
        print(f"Generate batch failed: {e}")
        return []


def _normalize_modality(value) -> str:
    modality = str(value or "").strip().lower()
    if modality in VALID_MODALITIES:
        return modality
    raise ValueError(
        f"Invalid modality '{value}'. Expected one of: {sorted(VALID_MODALITIES)}"
    )


def _build_model_inputs(model_key: str, batch_data, mode: str, model_type: str, processor):
    if model_key.startswith("aya"):
        return aya_inference_collator(
            batch_data=batch_data,
            mode=mode,
            model_type=model_type,
            processor=processor,
        )
    if model_key.startswith("llama"):
        return llama_inference_collator(
            batch_data=batch_data,
            mode=mode,
            model_type=model_type,
            processor=processor,
        )
    if model_key.startswith("qwen"):
        return qwen_inference_collator(
            batch_data=batch_data,
            mode=mode,
            model_type=model_type,
            processor=processor,
        )
    if model_key.startswith("gemma"):
        return gemma_inference_collator(
            batch_data=batch_data,
            mode=mode,
            model_type=model_type,
            processor=processor,
        )
    raise ValueError(f"Unknown model key:{model_key}")


def _map_gt_label(value) -> str:
    try:
        gt_idx = int(value)
    except (TypeError, ValueError):
        gt_raw = str(value or "").strip().lower()
        if gt_raw in {"sarcastic", "1"}:
            return "sarcastic"
        if gt_raw in {"unknown", "2"}:
            return "unknown"
        return "non_sarcastic"

    if gt_idx == 1:
        return "sarcastic"
    if gt_idx == 2:
        return "unknown"
    return "non_sarcastic"


def _prepare_results_dir(results_dir: str) -> str:
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_dir = os.path.join(results_dir, timestamp)
        try:
            os.makedirs(task_dir)
            return task_dir
        except FileExistsError:
            pass
        time.sleep(1)


def _prepare_results_path(results_dir: str, output_filename: str = "all.jsonl") -> str:
    task_dir = _prepare_results_dir(results_dir)
    return os.path.join(task_dir, output_filename)


def _prepare_batch_examples_for_mode(grouped_batch, mode: str):
    prepared_batch = []
    image_load_failures = 0
    for example in grouped_batch:
        if mode not in {"both", "image"}:
            prepared_batch.append(example)
            continue

        image_value = example.get("image")
        try:
            if image_value is None:
                raise ValueError("image is None")
            if isinstance(image_value, str):
                with Image.open(image_value) as image:
                    prepared_image = image.convert("RGB")
            elif isinstance(image_value, Image.Image):
                prepared_image = image_value if image_value.mode == "RGB" else image_value.convert("RGB")
            else:
                prepared_image = image_value
        except Exception:
            image_load_failures += 1
            continue

        prepared_example = dict(example)
        prepared_example["image"] = prepared_image
        prepared_batch.append(prepared_example)
    return prepared_batch, image_load_failures


def _run_generation_for_dataset(
    ds,
    model,
    processor,
    model_key: str,
    checkpoint_path: str,
    model_type: str,
    max_tokens: int,
    results_dir: str,
    task: str = "all",
    include_perturbation_tpye: bool = False,
    results_path: str = None,
    output_filename: str = "all.jsonl",
):
    loader = DataLoader(
        ds,
        batch_size=Logistics().infer_bathc_size,
        collate_fn=lambda b: b,
    )
    if results_path is None:
        results_path = _prepare_results_path(
            results_dir=results_dir,
            output_filename=output_filename,
        )
    written_count = 0
    failed_gen_count = 0
    image_load_failures = 0

    with open(results_path, "w", encoding="utf-8", buffering=1) as handle:
        for batch in tqdm(loader, desc="Test", leave=False):
            grouped_batches = {}
            for example in batch:
                grouped_mode = _normalize_modality(example.get("modality"))
                grouped_batches.setdefault(grouped_mode, []).append(example)

            for mode, grouped_batch in grouped_batches.items():
                prepared_batch, mode_image_load_failures = _prepare_batch_examples_for_mode(
                    grouped_batch=grouped_batch,
                    mode=mode,
                )
                image_load_failures += mode_image_load_failures
                if not prepared_batch:
                    continue
                print(
                    f"INFERENCE MODE:{mode} BATCH_SIZE:{len(prepared_batch)}"
                )
                model_inputs = _build_model_inputs(
                    model_key=model_key,
                    batch_data=prepared_batch,
                    mode=mode,
                    model_type=model_type,
                    processor=processor,
                )
                if model_inputs is None:
                    raise RuntimeError(f"failed to generate model input for {model_key}")

                outputs = generate_batch(
                    model=model,
                    model_inputs=model_inputs,
                    processor=processor,
                    max_tokens=max_tokens,
                )
                if not outputs:
                    failed_gen_count += len(prepared_batch)
                    print(
                        "Skipping empty outputs for "
                        f"model={model_key}, task={task}, mode={mode}, "
                        f"batch_size={len(prepared_batch)}, checkpoint={checkpoint_path}"
                    )
                    continue
                for example, output in zip(prepared_batch, outputs):
                    record_query = (
                        Queries().DPO_QUERY
                        if model_type == "mdpo"
                        else example.get("query")
                    )
                    record = {
                        "modality": mode,
                        "mode": mode,
                        "model_type": model_type,
                        "gt": _map_gt_label(example.get("label_gt")),
                        "task": task,
                        "query": record_query,
                        "target_json": example.get("target_json"),
                        "output": output,
                        "quality_flags": example.get("quality_flags"),
                        "id": example.get("id"),
                        "source": example.get("source"),
                        "model_key": model_key,
                    }
                    if include_perturbation_tpye and "perturbation_tpye" in example:
                        record["perturbation_tpye"] = example.get("perturbation_tpye")
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    written_count += 1

    return (
        {task: results_path},
        {
            "written_count": written_count,
            "failed_gen_count": failed_gen_count,
            "image_load_failures": image_load_failures,
        },
    )


def run_vlm_generation(
    model_key: str,
    checkpoint_path,
    model_type: str,
    max_tokens=256,
    merge_adapter: bool = True,
):
    try:
        basemodel = BaseModels()
        base_model_id = basemodel.models[model_key]
        model_class = basemodel.get_class(model_key)
        processor = _load_processor_for_inference(
            base_model_id=base_model_id,
            checkpoint_path=checkpoint_path,
        )
        if getattr(processor, "tokenizer", None) is not None:
            if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            # Force generation-time left padding for decoder-only models.
            processor.tokenizer.padding_side = "left"
        model = _load_model_for_inference(
            model_class=model_class,
            base_model_id=base_model_id,
            checkpoint_path=checkpoint_path,
            merge_adapter=merge_adapter,
        )
        model.eval()

        results_dir = get_sft_result_dir(model_key)
        ds = get_test_data()
        task_outputs, run_stats = _run_generation_for_dataset(
            ds=ds,
            model=model,
            processor=processor,
            model_key=model_key,
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            max_tokens=max_tokens,
            results_dir=results_dir,
            task="all",
            include_perturbation_tpye=False,
        )
        print(
            "##########################\n"
            f"Total examples written for task=all: {run_stats['written_count']}\n"
            f"Total generation failures for task=all: {run_stats['failed_gen_count']}"
        )
        print("Complted Generating")
        return task_outputs
    except Exception as e: 
        print(f"failed to run generation: {e}")
        raise


def ood_generation(
    model_key: str,
    checkpoint_path,
    model_type: str,
    max_tokens=256,
    merge_adapter: bool = True,
):
    try:
        basemodel = BaseModels()
        base_model_id = basemodel.models[model_key]
        model_class = basemodel.get_class(model_key)
        processor = _load_processor_for_inference(
            base_model_id=base_model_id,
            checkpoint_path=checkpoint_path,
        )
        if getattr(processor, "tokenizer", None) is not None:
            if processor.tokenizer.pad_token is None and processor.tokenizer.eos_token is not None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            processor.tokenizer.padding_side = "left"
        model = _load_model_for_inference(
            model_class=model_class,
            base_model_id=base_model_id,
            checkpoint_path=checkpoint_path,
            merge_adapter=merge_adapter,
        )
        model.eval()

        logistics = Logistics()
        redeval_root = os.path.join(
            logistics.project_root_dir,
            logistics.processed_data_dir,
            "redeval",
        )
        split_configs = [
            ("ood", "ood.jsonl", False, "all.jsonl"),
            ("ood_perturb", "ood_perturb.jsonl", True, "all_perturb.jsonl"),
        ]

        results_dir = get_sft_result_dir(model_key)
        run_task_dir = _prepare_results_dir(results_dir)
        split_outputs = {}
        aggregate_written_count = 0
        aggregate_failed_gen_count = 0
        aggregate_data_load_failures = 0
        aggregate_image_load_failures = 0

        for split_name, filename, include_perturbation_tpye, output_filename in split_configs:
            input_path = os.path.join(redeval_root, filename)
            ds, load_stats = _load_redeval_ood_rows(input_path)
            data_load_failures = load_stats["data_load_failures"]
            image_load_failures = load_stats["image_load_failures"]
            split_results_path = os.path.join(run_task_dir, output_filename)

            _, run_stats = _run_generation_for_dataset(
                ds=ds,
                model=model,
                processor=processor,
                model_key=model_key,
                checkpoint_path=checkpoint_path,
                model_type=model_type,
                max_tokens=max_tokens,
                results_dir=results_dir,
                task="all",
                include_perturbation_tpye=include_perturbation_tpye,
                results_path=split_results_path,
                output_filename=output_filename,
            )
            split_outputs[split_name] = {"all": split_results_path}
            aggregate_written_count += run_stats["written_count"]
            aggregate_failed_gen_count += run_stats["failed_gen_count"]
            aggregate_data_load_failures += data_load_failures
            aggregate_image_load_failures += image_load_failures + run_stats["image_load_failures"]
            print(
                "##########################\n"
                f"SPLIT={split_name}\n"
                f"Total examples written: {run_stats['written_count']}\n"
                f"Total generation failures: {run_stats['failed_gen_count']}\n"
                f"Total data-load failures: {data_load_failures}\n"
                f"Total image-load failures: {image_load_failures + run_stats['image_load_failures']}"
            )

        print(
            "##########################\n"
            "OOD GENERATION SUMMARY\n"
            f"Total examples written (all splits): {aggregate_written_count}\n"
            f"Total generation failures (all splits): {aggregate_failed_gen_count}\n"
            f"Total data-load failures (all splits): {aggregate_data_load_failures}\n"
            f"Total image-load failures (all splits): {aggregate_image_load_failures}"
        )
        print("Complted OOD Generating")
        return split_outputs
    except Exception as e:
        print(f"failed to run OOD generation: {e}")
        raise




if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m src.inference.run_inference <model_key> [model_type] [run_mode]"
        )

    model_key = sys.argv[1].strip().lower()
    model_type = sys.argv[2].strip().lower() if len(sys.argv) >= 3 else "sft"
    run_mode = sys.argv[3].strip().lower() if len(sys.argv) >= 4 else "default"
    if model_type not in VALID_MODEL_TYPES:
        raise SystemExit(
            f"Unknown model_type: {model_type} (expected: {sorted(VALID_MODEL_TYPES)})"
        )
    if run_mode not in {"default", "ood"}:
        raise SystemExit(
            f"Unknown run_mode: {run_mode} (expected: default|ood)"
        )

    if model_key not in BaseModels().models:
        raise SystemExit(f"Unknown model_key: {model_key}")
    if model_type == "mdpo" and model_key not in MDPO_MODEL_KEYS:
        raise SystemExit(
            f"model_type=mdpo supports only {sorted(MDPO_MODEL_KEYS)}; got {model_key}"
        )

    model_dir = _resolve_model_dir(model_key=model_key, model_type=model_type)
    best_model = _get_best_model(model_dir)
    best_metric = best_model.get("best_metric")
    best_chkpt_path = best_model.get("best_model_path")
    if not best_chkpt_path:
        raise SystemExit(
            f"No checkpoint found for {model_key} under {model_dir} "
            f"(model_type={model_type})"
        )
    print(
        f"Running inference for {model_key} ({model_type}, mode={run_mode}) from {best_chkpt_path} "
        f"[best_metric={best_metric}] merge_adapter=True"
    )
    if run_mode == "ood":
        ood_generation(
            model_key=model_key,
            checkpoint_path=best_chkpt_path,
            model_type=model_type,
            max_tokens=Logistics().gen_max_token,
            merge_adapter=True,
        )
    else:
        run_vlm_generation(
            model_key=model_key,
            checkpoint_path=best_chkpt_path,
            model_type=model_type,
            max_tokens=Logistics().gen_max_token,
            merge_adapter=True,
        )
