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
from pathlib import Path
from peft import PeftModel
from config.logistics import Logistics
from src.collators.inference_collator import (aya_inference_collator,
                                            qwen_inference_collator,
                                            gemma_inference_collator,
                                            llama_inference_collator )
from src.datasets.loader import load_hf_dataset
from src.hf.publish_lora_models import get_sft_best_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.sft_utils import get_sft_result_dir


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


def get_test_data(config_name: str = "en", task:str = "all"):
    dataset_id = None
    try:
        dataset_id = Logistics().hf_sft_ds_id
        ds = load_hf_dataset(
            dataset_id,
            split="test",
            config_name=config_name,
            # download_mode="force_redownload",
        )
        if task.lower() == "detection": 
            ds_detect = ds.filter(lambda x: x["task"] == "detection")
            return ds_detect
        elif task.lower() == "explanation":
            ds_exp = ds.filter(lambda x: x["task"] == "explanation")
            return ds_exp
        elif task.lower() == "detection_explanation":
            ds_dex = ds.filter(lambda x: x["task"] == "detection_explanation")
            return ds_dex
        else: return ds
    except Exception as e:
        ds_name = dataset_id or "<unknown>"
        raise Exception(f"failed to load the test dataset {ds_name}: {e}")


def generate_batch(model, model_inputs, processor, task: str = None, max_tokens=256):
    try:
        inputs = model_inputs.to(model.device)
        # Generation
        print("Generating response...")
        task_name = (task or "").strip().lower()
        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": False}
        if task_name in ["explanation", "detection_explanation"]:
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 50,
            }
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode while trimming the prompt tokens
        generated_ids = [
            out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)
        ]
        return processor.batch_decode(generated_ids, skip_special_tokens=True)

    except Exception as e:
        task_label = (task or "").strip().lower() or "<unknown>"
        print(f"Generate batch failed for task={task_label}: {e}")
        return []



def run_vlm_generation(model_key: str, checkpoint_path, max_tokens=256, modes=None):
    try:
        if modes is None:
            modes = ["both", "text", "image"]
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
        for task in ["detection_explanation", "explanation"]:  #, "detection"
            ds = get_test_data(task=task)
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
                    for mode in modes:
                        print(f"INFERENCE MODE:{mode}")
                        if model_key.startswith("aya"):
                            model_inputs = aya_inference_collator(
                                batch_data=batch,
                                mode=mode,
                                processor=processor)
                        elif model_key.startswith('llama'):
                            model_inputs = llama_inference_collator(
                                batch_data=batch,
                                mode=mode,
                                processor=processor
                            )
                        elif model_key.startswith('qwen'):
                            model_inputs = qwen_inference_collator(
                                batch_data=batch,
                                mode=mode,
                                processor=processor)
                        elif model_key.startswith('gemma'):
                            model_inputs = gemma_inference_collator(
                                batch_data=batch,
                                mode=mode,
                                processor=processor
                            )
                        else:
                            raise ValueError(f"Unknown model key:{model_key}")
                        # generate
                        if model_inputs is not None:
                            outputs = generate_batch(
                                model=model,
                                model_inputs=model_inputs,
                                processor=processor,
                                max_tokens=max_tokens,
                                task=task
                            )
                            if not outputs:
                                failed_gen_count += len(batch)
                                print(
                                    "Skipping empty outputs for "
                                    f"model={model_key}, task={task}, mode={mode}, "
                                    f"batch_size={len(batch)}, checkpoint={checkpoint_path}"
                                )
                                continue
                            for example, output in zip(batch, outputs):
                                record = {
                                    "mode": mode,
                                    "gt":"sarcastic" if int(example.get("label_gt")) == 1 else "non_sarcastic",
                                    "task": example.get("task"),
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
                                # print(f"Example meta: {record}\nOutput:\n{output}")
                        else:
                            raise RuntimeError(f"failed to generate model input for {model_key}")
                    # for testing purpose
                    # break
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
        raise SystemExit("Usage: python -m src.inference.run_inference <model_key>")

    model_key = sys.argv[1].strip().lower()
    saved_model_path = Path(
        os.path.join(Logistics().project_root_dir, Logistics().models_output_dir, "sft")
    )
    models_dirs = {
        "aya": "aya_vision_8b",
        "llama": "llama32",
        "gemma": "gemma3",
        "qwen": "qwen3_vl_8b",
    }

    if model_key not in models_dirs:
        raise SystemExit(f"Unknown model_key: {model_key}")

    model_dir = os.path.join(saved_model_path, models_dirs[model_key])
    best_model = get_sft_best_model(model_dir)
    best_metric = best_model.get("best_metric")
    best_chkpt_path = best_model.get("best_model_path")
    if not best_chkpt_path:
        raise SystemExit(f"No checkpoint found for {model_key} under {model_dir}")
    print(f"Running inference for {model_key} from {best_chkpt_path}")
    run_vlm_generation(
        model_key=model_key,
        checkpoint_path=best_chkpt_path,
        max_tokens=Logistics().gen_max_token,
    )
