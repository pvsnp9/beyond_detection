import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Logistics:
    project_root_dir: str = "/projects/mzampier/tsuyog/beyond_detection"
    output_root_dir: str = "outputs"
    models_output_dir: str = "outputs/models"
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    hf_cache_dir: str = "data/raw/hf_cache"
    cache_dir: str = "cache"
    sft_log_dir: str = "outputs/logs/sft"
    results_dir:str = "outputs/results"
    reports_dir: str = "outputs/reports"

    data_generation_dir: str = "data/generated"
    combined_data_dir: str = f"{data_generation_dir}/combined"
    hf_datatset_ids: dict[str, str] = field(
        default_factory=lambda: {
            "mmsd2": "coderchen01/MMSD2.0",
            "muse": "alita9/muse-sarcasm-explanation",
            "sarcnet": "alita9/sarcnet",
        }
    )

    splits: List[str] = field(
        default_factory=lambda: ["train", "validation", "test"]
    )

    langs: List[str] = field(default_factory=lambda: ["en", "zh"])
    infer_bathc_size:int = 4
    gen_max_token:int = 384

    hf_sft_ds_id: str = "alita9/beyond_sarcasm_detection_sft"
    hf_token:any = None
    wandb_token: any = None
    wandb_project: str = "sarcasm_sft"
    wandb_tags: List[str] = field(default_factory=lambda: ["sarcasm", "sft"])

@dataclass(frozen=True)
class LocalDataDirs:
    data_root_dir: str = f"/projects/mzampier/tsuyog/sarcasm_bench"
    mmsd2: str = f"{data_root_dir}/data/mmsd2.0/dataset_image"
    muse: str = f"{data_root_dir}/data/muse/images"
    sarcnet: str = f"{data_root_dir}/data/sarcnet/images"




#------------------------------
# Model Cards
# ------------------------------

@dataclass(frozen=True)
class ModelCards:
    llama3_2vl: str = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    aya_model_name: str = "CohereLabs/aya-vision-8b"
    gemm3_12b: str = "google/gemma-3-12b-it"
    qwen3_vl_8b_instruct: str = "Qwen/Qwen3-VL-8B-Instruct"

    hf_model_ids: dict[str, str] = field(
        default_factory=lambda: {
            "aya": "alita9/aya-lora-sarcasm-sft",
            "qwen": "alita9/qwen3-lora-sarcasm-sft",
            "gemma": "alita9/gemma3-lora-sarcasm-sft",
            "llama": "alita9/llam32-lora-sarcasm-sft"
        }
    )

    llama_max_length:int = 1024
    aya_max_ength: int = 2048


# ------------------------------
# QLoRA (bitsandbytes) Params
# ------------------------------
@dataclass(frozen=True)
class QLoRAParams:
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"

    # Compute dtype for 4-bit matmuls
    # On A100: bf16 is ideal
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Optional extras (handy to keep)
    llm_int8_threshold: float = 6.0   # unused for pure 4-bit, safe to keep
    llm_int8_has_fp16_weight: bool = False


# ------------------------------
# LoRA Params
# ------------------------------
@dataclass(frozen=True)
class LoRAParams:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"

    # attention + MLP projections
    target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


# ------------------------------
# SFT Parameters (A100 optimized)
# ------------------------------
@dataclass
class SFTParams:
    # Training length
    num_epochs: int = 3                   # start with 1; go to 2 if still improving

    # Dataloader
    num_workers: int = 10                 # 8–12 usually best on A100 nodes

    # Optimization
    lr: float = 1e-4                      # LoRA SFT standard (2e-5 is typically too low)
    weight_decay: float = 0.0             # adapters generally don’t need wd
    warmup_steps: int = 0                 # use warmup_ratio instead
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # Batch sizing
    batch_size: int = 4                   # per-device micro-batch; drop to 2 if images are large
    gradient_accumulation_steps: int = 8  # effective batch = 32 (good default)

    # Sequence
    max_length: int = 1024


    # Precision (prefer bf16 in  SFTConfig; keep fp16 off)
    fp16: bool = False

    # Logging / checkpointing
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 2

    # Model selection
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Repro
    seed: int = 42

    # Output
    model_dir: str = "/projects/mzampier/tsuyog/beyond_detection/outputs/models"

    # Early stopping (only if you actually add the callback)
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01

    # Inference / eval decode (make deterministic for classification)
    do_sample: bool = False
    top_p: float = 1.0
    temperature: float = 0.0


# ------------------------------
# (Optional) SFTConfig - extras you should pass explicitly
# ------------------------------
@dataclass
class SFTConfigExtras:
    # Speed / memory
    bf16: bool = True                     # A100 supports bf16
    tf32: bool = True
    gradient_checkpointing: bool = True
    use_cache: bool = False               # must be False when checkpointing

    # Attention acceleration
    use_flash_attention_2: bool = True    # try; fallback gracefully

    # Optimizer
    optim: str = "paged_adamw_8bit"

    # Dataloader tuning
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 2

    # Trainer display
    disable_tqdm: bool = False

    # Saving
    save_safetensors: bool = True

    # Eval strategy
    evaluation_strategy: str = "steps"


def build_cfg(model_name_or_path: str) -> dict:
    logistics = Logistics()
    logistics.wandb_tags.append(model_name_or_path)
    safe_model_name = model_name_or_path.replace("/", "-")
    run_name = f"{safe_model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return {
        "logistics": logistics,
        "qlora": QLoRAParams(),
        "lora": LoRAParams(),
        "sft": SFTParams(),
        "sft_extras": SFTConfigExtras(),
        "mode": "rationale_sft",
        "model": {
            "base_model_name_or_path": model_name_or_path,
            "use_flash_attention": True,
            "torch_dtype": "bfloat16",
            "gradient_checkpointing": True,
            "use_cache": False,
        },
        "dataset": {
            "dataset_id": logistics.hf_sft_ds_id,
            "lang": "en",
            "train_split": "train",
            "eval_split": "validation",
            "fallback_eval_split": "test",
            "streaming": False,
            "max_train_samples": None,
            "max_eval_samples": 8,
        },
        "output": {
            "adapter_output_dir": None,
            "run_name": run_name,
        },
        "wandb": {
            "project": os.environ.get("WANDB_PROJECT", logistics.wandb_project),
            "entity": None,
            "tags": logistics.wandb_tags,
            "log_model": False,
        },
        "collator": {
            "image_key": "image",
            "query_key": "query",
            "caption_key": "caption",
            "target_key": "target_json",
            "system_prompt": None,
        },
        "eval_decode": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 8,
        },
    }
