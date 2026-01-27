import json
import os

from config.logistics import Logistics, ModelCards

from pathlib import Path
from transformers import (
    MllamaForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    AyaVisionForConditionalGeneration, AutoProcessor
)
from peft import PeftModel
import torch


def get_sft_best_model(model_name: str) -> dict:
    result = {"best_metric": None, "best_model_path": None}
    try:
        logistics = Logistics()
        base_dir = os.path.join(
            logistics.project_root_dir,
            logistics.models_output_dir,
            "sft",
            model_name,
        )
        if not os.path.isdir(base_dir):
            return result

        checkpoints = []
        for name in os.listdir(base_dir):
            if not name.startswith("checkpoint-"):
                continue
            step = name.split("-")[-1]
            if step.isdigit():
                checkpoints.append((int(step), os.path.join(base_dir, name)))

        if not checkpoints:
            return result

        latest_checkpoint = max(checkpoints, key=lambda item: item[0])[1]
        state_path = os.path.join(latest_checkpoint, "train_state.json")
        if not os.path.isfile(state_path):
            state_path = os.path.join(latest_checkpoint, "trainer_state.json")

        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)

        result["best_metric"] = state.get("best_metric")
        result["best_model_path"] = state.get("best_model_checkpoint", latest_checkpoint)
        return result
    except Exception:
        return result


def publish_aya_vision(checkpoint_path):
    try:
        repo_id = ModelCards().hf_model_ids['aya']
        print(f"🌍 Publishing Aya-Vision to {repo_id}")

        processor = AutoProcessor.from_pretrained(ModelCards().aya_model_name)

        # Using the specific class avoids the AutoModel mapping error
        model = AyaVisionForConditionalGeneration.from_pretrained(
            ModelCards().aya_model_name,
            dtype=torch.bfloat16, 
            device_map="auto",
        )
        
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()

        model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)
        print("Done!")
    except Exception as exc:
        print(f"Failed to publish Aya-Vision: {exc}")

def publish_llama_vision(checkpoint_path):
    try:
        repo_id = ModelCards().hf_model_ids['llama']
        print(f"🦙 Publishing Llama-Vision to {repo_id}")

        processor = AutoProcessor.from_pretrained(ModelCards().llama3_2vl)

        model = MllamaForConditionalGeneration.from_pretrained(
            ModelCards().llama3_2vl,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()

        model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)
        print("Done!")
    except Exception as exc:
        print(f"Failed to publish Llama-Vision: {exc}")

def publish_qwen3_vl(checkpoint_path):
    try:
        try:
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        except Exception as e: raise e

        repo_id = ModelCards().hf_model_ids['qwen']
        print(f"🚀 Publishing Qwen3-VL to {repo_id}")

        # Use the specific Qwen3 processor
        processor = Qwen3VLProcessor.from_pretrained(
            ModelCards().qwen3_vl_8b_instruct, 
            trust_remote_code=True
        )

        # Use the specific Qwen3 model class
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            ModelCards().qwen3_vl_8b_instruct,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load and merge your sarcasm SFT adapter
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()

        model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)
        print("Done!")
    except Exception as exc:
        print(f"Failed to publish Qwen3-VL: {exc}")

def publish_gemma3_vl(checkpoint_path):
    try:
        repo_id = ModelCards().hf_model_ids['gemma']
        print(f"💎 Publishing Gemma 3 VL to {repo_id}")

        # Gemma 3 often uses <image_soft_token> as a placeholder
        processor = AutoProcessor.from_pretrained(ModelCards().gemm3_12b)

        model = Gemma3ForConditionalGeneration.from_pretrained(
            ModelCards().gemm3_12b,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()

        model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)
        print("Done!")
    except Exception as exc:
        print(f"Failed to publish Gemma 3 VL: {exc}")


if __name__ == "__main__":
    saved_model_path = Path(os.path.join(Logistics().project_root_dir, Logistics().models_output_dir, "sft"))
    models_dir = [p.name for p in saved_model_path.iterdir() if p.is_dir()]
    print(models_dir)
    # for model_dir in models_dir:
    #     best_metric, best_chkpt_path = get_sft_best_model(model_dir).values()
    #     print(f"Running: {model_dir}\n[Best score:{best_metric} || Checkpoint:{best_chkpt_path}]")
    #     if model_dir.startswith("aya"): publish_aya_vision(best_chkpt_path)
    #     elif model_dir.startswith("gemma3"): publish_gemma3_vl(best_chkpt_path)
    #     elif model_dir.startswith("qwen3"):  publish_qwen3_vl(best_chkpt_path)
    #     elif model_dir.startswith("llama3"): publish_llama_vision(best_chkpt_path)
    #     else: print(f"UNKNOWN model dir:{model_dir}")

