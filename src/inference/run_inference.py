import json
import os
import sys
from datetime import datetime

import torch
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AyaVisionForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    MllamaForConditionalGeneration,
)
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from config.logistics import Logistics
from config.queries import Queries
from src.collators.inference_collator import (
    aya_inference_collator,
    gemma_inference_collator,
    llama_inference_collator,
    qwen_inference_collator,
)
from src.datasets.loader import load_hf_dataset
from src.utils.sft_utils import get_sft_result_dir

VALID_MODALITIES = {"both", "text", "image"}

BASE_MODELS = {
    "llama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "aya": "CohereLabs/aya-vision-8b",
    "gemma": "google/gemma-3-12b-it",
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
}

COLLATORS = {
    "llama": llama_inference_collator,
    "aya": aya_inference_collator,
    "gemma": gemma_inference_collator,
    "qwen": qwen_inference_collator,
}

_SFT_DIRS = {"llama": "llama32", "gemma": "gemma3", "aya": "aya_vision_8b", "qwen": "qwen3_vl_8b"}
_DPO_DIRS = {"llama": "llama32", "gemma": "gemma3", "qwen": "qwen3", "aya": "aya"}

# query="dpo" -> record/prompt query is Queries().DPO_QUERY, else the dataset query
GENERATION_TYPES = {
    "sft": {"dirs": _SFT_DIRS, "system_prompt": "SYSTEM_PROMPT", "query": "dataset"},
    "freeform_sft": {"dirs": _SFT_DIRS, "system_prompt": "FREEFORM_SYSTEM_PROMPT", "query": "dataset"},
    "rich_freeform_sft": {"dirs": _SFT_DIRS, "system_prompt": "RICH_FREEFORM_SYSTEM_PROMPT", "query": "dataset"},
    "mdpo": {"dirs": _DPO_DIRS, "system_prompt": "SYSTEM_PROMPT", "query": "dpo"},
    "dpo": {"dirs": _DPO_DIRS, "system_prompt": "SYSTEM_PROMPT", "query": "dpo"},
    "dpo_random": {"dirs": _DPO_DIRS, "system_prompt": "SYSTEM_PROMPT", "query": "dpo"},
}


def _get_model_class(model_key: str):
    if model_key == "qwen":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    return {
        "llama": MllamaForConditionalGeneration,
        "aya": AyaVisionForConditionalGeneration,
        "gemma": Gemma3ForConditionalGeneration,
    }.get(model_key, AutoModelForImageTextToText)


class SafeRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    # Mllama's <|image|> id equals text vocab_size, overflowing the default
    # repetition-penalty gather/scatter; clamp ids into lm_head range first.
    def __init__(self, penalty: float):
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        safe_ids = input_ids.clamp(max=scores.shape[-1] - 1)
        score = torch.gather(scores, 1, safe_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores.scatter_(1, safe_ids, score)
        return scores


def get_test_data(config_name: str = "en"):
    dataset_id = Logistics().hf_sft_ds_id
    try:
        return load_hf_dataset(dataset_id, split="test", config_name=config_name)
    except Exception as e:
        raise RuntimeError(f"failed to load test dataset {dataset_id}: {e}") from e


def get_best_checkpoint(model_dir: str) -> tuple[str, float]:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model dir does not exist (train this type first?): {model_dir}")

    checkpoints = [
        (int(name.split("-")[-1]), os.path.join(model_dir, name))
        for name in os.listdir(model_dir)
        if name.startswith("checkpoint-") and name.split("-")[-1].isdigit()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint-* found under {model_dir}")

    latest = max(checkpoints)[1]
    state_path = next(
        (p for p in (os.path.join(latest, n) for n in ("train_state.json", "trainer_state.json"))
         if os.path.isfile(p)),
        None,
    )
    if state_path is None:
        raise FileNotFoundError(f"no train_state.json/trainer_state.json in {latest}")

    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)

    best_ckpt = state.get("best_model_checkpoint")
    if not best_ckpt:
        raise FileNotFoundError(f"best_model_checkpoint missing in {state_path}")
    if not os.path.isabs(best_ckpt):
        best_ckpt = os.path.join(model_dir, best_ckpt)
    if not os.path.isdir(best_ckpt):
        # stale absolute path from a moved run: retry basename under model_dir
        candidate = os.path.join(model_dir, os.path.basename(best_ckpt))
        if not os.path.isdir(candidate):
            raise FileNotFoundError(f"best checkpoint not found on disk: {best_ckpt}")
        best_ckpt = candidate
    return best_ckpt, state.get("best_metric")


def _resolve_adapter_lineage(checkpoint_path: str) -> dict:
    # mDPO/DPO adapters are fresh LoRAs trained on an SFT-merged base; their
    # training_meta.json (next to or above the checkpoint) records that base.
    for candidate_dir in (checkpoint_path, os.path.dirname(checkpoint_path)):
        meta_path = os.path.join(candidate_dir, "training_meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception as e:
                raise RuntimeError(f"failed to read {meta_path}: {e}") from e
    return {}


def load_processor(base_model_id: str, checkpoint_path: str):
    candidates = [checkpoint_path, os.path.dirname(checkpoint_path), base_model_id]
    last_error = None
    for source in candidates:
        try:
            processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, use_fast=True)
        except Exception as e:
            last_error = e
            continue
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            # dirs without preprocessor_config.json yield a bare tokenizer; skip them
            last_error = ValueError(f"{source} has no multimodal processor config")
            continue
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # decoder-only generation
        print(f"Loaded processor from: {source}")
        return processor
    raise RuntimeError(f"failed to load processor from {candidates}: {last_error}")


def load_merged_model(model_key: str, checkpoint_path: str):
    base_model_id = BASE_MODELS[model_key]
    chain_sft_checkpoint = None

    lineage = _resolve_adapter_lineage(checkpoint_path)
    if lineage.get("adapter_style") == "fresh_lora_on_merged_sft":
        merged_base = lineage.get("merged_base_path")
        if merged_base and os.path.isdir(merged_base):
            base_model_id = merged_base
        elif lineage.get("sft_adapter_checkpoint"):
            chain_sft_checkpoint = lineage["sft_adapter_checkpoint"]
        else:
            raise RuntimeError(
                f"adapter {checkpoint_path} needs its SFT-merged base but neither "
                "merged_base_path nor sft_adapter_checkpoint is available"
            )

    print(f"Loading base model: {base_model_id}")
    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto", "trust_remote_code": True}
    if model_key == "gemma":
        # SDPA hits "attn_bias_ptr is not correctly aligned" on padded batches
        load_kwargs["attn_implementation"] = "eager"
    try:
        model = _get_model_class(model_key).from_pretrained(base_model_id, **load_kwargs)
    except Exception as e:
        raise RuntimeError(f"failed to load base model {base_model_id}: {e}") from e

    try:
        if chain_sft_checkpoint:
            print(f"Merged base missing; chaining SFT adapter first: {chain_sft_checkpoint}")
            model = PeftModel.from_pretrained(model, chain_sft_checkpoint).merge_and_unload()
        print(f"Applying and merging adapter: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path).merge_and_unload()
    except Exception as e:
        raise RuntimeError(f"failed to apply/merge adapter {checkpoint_path}: {e}") from e

    model.eval()
    return model


def generate_batch(model, model_inputs, processor, max_tokens: int) -> list[str]:
    try:
        inputs = model_inputs.to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                logits_processor=LogitsProcessorList([SafeRepetitionPenaltyLogitsProcessor(1.05)]),
            )
        generated = [out[len(ins):] for ins, out in zip(inputs.input_ids, output_ids)]
        return processor.batch_decode(generated, skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError(f"batch generation failed: {e}") from e


def _normalize_modality(value) -> str:
    modality = str(value or "").strip().lower()
    if modality not in VALID_MODALITIES:
        raise ValueError(f"invalid modality '{value}'; expected one of {sorted(VALID_MODALITIES)}")
    return modality


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
    return {1: "sarcastic", 2: "unknown"}.get(gt_idx, "non_sarcastic")


def _prepare_batch_for_mode(grouped_batch, mode: str):
    prepared, image_failures = [], 0
    for example in grouped_batch:
        if mode not in {"both", "image"}:
            prepared.append(example)
            continue
        image_value = example.get("image")
        try:
            if isinstance(image_value, str):
                with Image.open(image_value) as image:
                    image_value = image.convert("RGB")
            elif isinstance(image_value, Image.Image):
                image_value = image_value if image_value.mode == "RGB" else image_value.convert("RGB")
            elif image_value is None:
                raise ValueError("image is None")
        except Exception:
            image_failures += 1
            continue
        prepared.append({**example, "image": image_value})
    return prepared, image_failures


def run_generation(ds, model, processor, model_key: str, gen_type: str, results_path: str):
    spec = GENERATION_TYPES[gen_type]
    system_prompt = getattr(Queries(), spec["system_prompt"])
    dpo_query = Queries().DPO_QUERY if spec["query"] == "dpo" else None
    collate = COLLATORS[model_key]
    loader = DataLoader(ds, batch_size=Logistics().infer_batch_size, collate_fn=lambda b: b)
    max_tokens = Logistics().gen_max_token
    written, image_failures = 0, 0

    with open(results_path, "w", encoding="utf-8", buffering=1) as handle:
        for batch in tqdm(loader, desc=f"{gen_type}:{model_key}", leave=False):
            grouped = {}
            for example in batch:
                grouped.setdefault(_normalize_modality(example.get("modality")), []).append(example)

            for mode, grouped_batch in grouped.items():
                prepared, mode_failures = _prepare_batch_for_mode(grouped_batch, mode)
                image_failures += mode_failures
                if not prepared:
                    continue
                model_inputs = collate(
                    batch_data=prepared,
                    mode=mode,
                    model_type=gen_type,
                    processor=processor,
                    system_prompt=system_prompt,
                )
                outputs = generate_batch(model, model_inputs, processor, max_tokens)
                for example, output in zip(prepared, outputs):
                    record = {
                        "modality": mode,
                        "mode": mode,
                        "model_type": gen_type,
                        "gt": _map_gt_label(example.get("label_gt")),
                        "query": dpo_query or example.get("query"),
                        "target_json": example.get("target_json"),
                        "output": output,
                        "quality_flags": example.get("quality_flags"),
                        "id": example.get("id"),
                        "source": example.get("source"),
                        "model_key": model_key,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
    return written, image_failures


def main(gen_type: str, model_key: str):
    logistics = Logistics()
    model_dir = os.path.join(
        logistics.project_root_dir,
        logistics.models_output_dir,
        gen_type,
        GENERATION_TYPES[gen_type]["dirs"][model_key],
    )
    checkpoint_path, best_metric = get_best_checkpoint(model_dir)
    print(f"Running {gen_type} generation for {model_key}: {checkpoint_path} [best_metric={best_metric}]")

    processor = load_processor(BASE_MODELS[model_key], checkpoint_path)
    model = load_merged_model(model_key, checkpoint_path)

    ds = get_test_data()
    limit = int(os.environ.get("INFER_LIMIT", "0"))  # >0 caps examples for smoke runs
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"INFER_LIMIT={limit}: running on {len(ds)} examples")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(get_sft_result_dir(model_key), f"{gen_type}_iid_{timestamp}.jsonl")
    written, image_failures = run_generation(
        ds=ds,
        model=model,
        processor=processor,
        model_key=model_key,
        gen_type=gen_type,
        results_path=results_path,
    )
    print(
        "##########################\n"
        f"Examples written: {written}\n"
        f"Image load failures: {image_failures}\n"
        f"Results: {results_path}"
    )
    return results_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python -m src.inference.run_inference <sft|freeform_sft|rich_freeform_sft|mdpo|dpo|dpo_random> <llama|gemma|qwen|aya>"
        )
    gen_type = sys.argv[1].strip().lower()
    model_key = sys.argv[2].strip().lower()
    if gen_type not in GENERATION_TYPES:
        raise SystemExit(f"unknown gen_type: {gen_type} (expected: {sorted(GENERATION_TYPES)})")
    if model_key not in GENERATION_TYPES[gen_type]["dirs"]:
        raise SystemExit(
            f"{gen_type} supports {sorted(GENERATION_TYPES[gen_type]['dirs'])}; got {model_key}"
        )
    main(gen_type, model_key)
