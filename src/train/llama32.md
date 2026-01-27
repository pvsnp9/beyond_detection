# Report: Llama32 SFT Pipeline Implementation

## What Was Implemented
- `src/train/llama32sft.py`: Full SFT pipeline for Llama-3.2 Vision using QLoRA + TRL `SFTTrainer`.
- `src/train/sft.py`: Thin dispatcher that calls the Llama32 SFT entrypoint.
- `config/logistics.py`: Shared `build_cfg(model_name_or_path)` for reuse across SFT variants.
- `src/utils/env.py`: Token resolution + safe runtime env setup.
- `src/utils/logging.py`: W&B init (optional) + metadata logging.
- `src/utils/eval_sarcasm.py`: Lightweight generation-based eval with accuracy/macro‑F1.

## How the SFT Script Works (`src/train/llama32sft.py`)
- Builds config via `build_cfg(ModelCards().llama3_2vl)` and sets the seed.
- Resolves HF/W&B tokens from env or `config/logistics.py`.
- Loads datasets using `load_hf_dataset` with validation fallback to test, otherwise uses a train slice.
- Sets TF32 flags and loads processor + model with QLoRA.
- FlashAttention‑2 is attempted in a tight try/except; if unavailable, it reloads without it and logs a clear warning.
- Applies `prepare_model_for_kbit_training`, then LoRA via PEFT.
- Instantiates `Llama32VisionSFTCollator` for train and eval.
- Runs the required label‑sanity check to ensure supervised tokens exist.
- Configures `SFTConfig` and trains with `SFTTrainer`.
- Runs generation-based eval before and after training (and on eval steps via callback).
- Saves adapter‑only weights and processor to `cfg.sft.model_dir/sft/llama32/`, writes `training_meta.json`, and validates adapter artifacts.

## Utilities
- `src/utils/env.py`: `resolve_tokens_and_env` + `set_runtime_env` for safe token handling.
- `src/utils/logging.py`: W&B init that never blocks training; logs run metadata.
- `src/utils/eval_sarcasm.py`: Generates short outputs, normalizes labels, computes accuracy/macro‑F1 when possible.

## Notes
- Default adapter output directory: `cfg.sft.model_dir/sft/llama32/` (overrideable).
- Evaluation uses `max_new_tokens=8` for fast label probing, not full JSON generation.
