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
from peft import PeftModel
from config.logistics import Logistics
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


def generate_batch(
    model,
    model_inputs,
    processor,
    max_tokens=256,
):
    try:
        inputs = model_inputs.to(model.device)
        print("Generating response (sampling pass)...")
        sample_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.05,
        }
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **sample_kwargs,
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


def _build_model_inputs(model_key: str, batch_data, mode: str, processor):
    if model_key.startswith("aya"):
        return aya_inference_collator(batch_data=batch_data, mode=mode, processor=processor)
    if model_key.startswith("llama"):
        return llama_inference_collator(batch_data=batch_data, mode=mode, processor=processor)
    if model_key.startswith("qwen"):
        return qwen_inference_collator(batch_data=batch_data, mode=mode, processor=processor)
    if model_key.startswith("gemma"):
        return gemma_inference_collator(batch_data=batch_data, mode=mode, processor=processor)
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


def run_vlm_generation(model_key: str, checkpoint_path, model_type: str, max_tokens=256):
    try:
        basemodel = BaseModels()
        base_model_id = basemodel.models[model_key]
        model_class = basemodel.get_class(model_key)
        processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True, use_fast=True)
        if getattr(processor, "tokenizer", None) is not None:
            processor.tokenizer.padding_side = "left"

        model = model_class.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        if checkpoint_path:
            print(f"Loading Adapter: {checkpoint_path}")
            model = PeftModel.from_pretrained(model, checkpoint_path)
        model.eval()

        results_dir = get_sft_result_dir(model_key)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_outputs = {}
        task = "all"
        ds = get_test_data()
        loader = DataLoader(
            ds,
            batch_size=Logistics().infer_bathc_size,
            collate_fn=lambda b: b,
        )
        task_dir = os.path.join(results_dir, timestamp)
        os.makedirs(task_dir, exist_ok=True)
        results_path = os.path.join(task_dir, f"{task}.jsonl")
        task_outputs[task] = results_path
        written_count = 0
        failed_gen_count = 0
        with open(results_path, "w", encoding="utf-8") as handle:
            for batch in tqdm(loader, desc="Test", leave=False):
                grouped_batches = {}
                for example in batch:
                    grouped_mode = _normalize_modality(example.get("modality"))
                    grouped_batches.setdefault(grouped_mode, []).append(example)

                for mode, grouped_batch in grouped_batches.items():
                    print(
                        f"INFERENCE MODE:{mode} BATCH_SIZE:{len(grouped_batch)}"
                    )
                    model_inputs = _build_model_inputs(
                        model_key=model_key,
                        batch_data=grouped_batch,
                        mode=mode,
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
                        failed_gen_count += len(grouped_batch)
                        print(
                            "Skipping empty outputs for "
                            f"model={model_key}, task={task}, mode={mode}, "
                            f"batch_size={len(grouped_batch)}, checkpoint={checkpoint_path}"
                        )
                        continue
                    for example, output in zip(grouped_batch, outputs):
                        record = {
                            "modality": mode,
                            "mode": mode,
                            "model_type": model_type,
                            "gt": _map_gt_label(example.get("label_gt")),
                            "task": task,
                            "query": example.get("query"),
                            "target_json": example.get("target_json"),
                            "output": output,
                            "quality_flags": example.get("quality_flags"),
                            "id": example.get("id"),
                            "source": example.get("source"),
                            "model_key": model_key,
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written_count += 1
        print(
            "##########################\n"
            f"Total examples written for task={task}: {written_count}\n"
            f"Total generation failures for task={task}: {failed_gen_count}"
        )
        print("Complted Generating")
        return task_outputs
    except Exception as e: 
        print(f"failed to run generation: {e}")
        raise



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m src.inference.run_inference <model_key> [model_type]"
        )

    model_key = sys.argv[1].strip().lower()
    model_type = sys.argv[2].strip().lower() if len(sys.argv) >= 3 else "sft"
    if model_type not in VALID_MODEL_TYPES:
        raise SystemExit(
            f"Unknown model_type: {model_type} (expected: {sorted(VALID_MODEL_TYPES)})"
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
        f"Running inference for {model_key} ({model_type}) from {best_chkpt_path} "
        f"[best_metric={best_metric}]"
    )
    run_vlm_generation(
        model_key=model_key,
        checkpoint_path=best_chkpt_path,
        model_type=model_type,
        max_tokens=Logistics().gen_max_token,
    )
