# Codex Instruction: Add Qwen3-VL-8B-Instruct SFT (mirror gemma3sft + reuse utilities)

## Goal
Implement a production-grade SFT pipeline for `Qwen/Qwen3-VL-8B-Instruct` that mirrors your `gemma3sft` script structure and reuses the same helper utilities (`_select_subset`, `_load_eval_dataset`, `LogCallback`, `_count_trainable_params`, etc.). Qwen3-VL uses `Qwen3VLForConditionalGeneration` + `AutoProcessor`, supports `attn_implementation` (e.g., `sdpa`, `flash_attention_2`), and the recommended decode trims the prompt tokens from `generated_ids`. :contentReference[oaicite:0]{index=0}

---

## 1) New collator (minimal, production-grade)

### File
Create: `src/collators/qwen3_collator.py`

### Class
Implement `class Qwen3VisionSFTCollator:` with the same constructor signature as `Gemma3VisionSFTCollator`:
- `processor`, `training=True`, `max_length=None`
- keys: `image_key="image"`, `query_key="query"`, `caption_key="caption"`, `target_key="target_json"`
- `system_prompt=None` default to `Queries().SYSTEM_PROMPT`

### Formatting (mirror your existing collators)
- `_user_text(ex)`:
  - query stripped
  - if caption: `"{query}\nCAPTION: {caption}"`
- `_user_message(user_text)`:
  - optional system message with `{"role":"system","content":[{"type":"text","text":system_prompt}]}`
  - user message with **image placeholder + text**:
    - `{"role":"user","content":[{"type":"image"},{"type":"text","text":user_text}]}`
  (Qwen3-VL supports this “content list” image+text message format in Transformers examples. :contentReference[oaicite:1]{index=1})

### Tokenization strategy (keep consistent with your pipeline)
- Build `full_texts` (and `prompt_texts` for training) via:
  - `processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=...)`
- Tokenize once via:
  - `model_inputs = processor(text=full_texts, images=images_per_sample, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)`
- IMPORTANT: Qwen3-VL processors may emit `token_type_ids`; ensure compatibility:
  - `model_inputs.pop("token_type_ids", None)` before returning (recommended in HF docs). :contentReference[oaicite:2]{index=2}

### Labels (training=True)
- `labels = input_ids.clone()`
- mask padding using `pad_id = processor.tokenizer.pad_token_id` (do not hardcode)
- mask prompt prefix:
  - compute `prompt_len` by calling processor on `prompt_text` with the paired image (`images=[[image]]`, no padding)
  - set `labels[i, :prompt_len] = -100`
- set `model_inputs["labels"] = labels`

### Validation + error handling
- Raise `ValueError` if missing `image_key`
- If `training=True`, require non-empty `target_json`
- Wrap `__call__` body in try/except and raise:
  - `RuntimeError(f"Qwen3 collate failed: {exc}") from exc`
- Keep code minimal; no logging frameworks.

---

## 2) New SFT entrypoint (mirror gemma3sft)

### File
Create: `src/train/qwen3_vl_8b_sft.py`

### Structure
Copy the flow of your Gemma3 script and change only what’s model-specific:
- `cfg = build_cfg(ModelCards().qwen3_vl_8b_instruct)` (use your actual ModelCards field name)
- seed/env/login/wandb/output dirs/datasets exactly as Gemma3

### Model + processor (Qwen3-specific)
- Import model class:
  - `from transformers import Qwen3VLForConditionalGeneration, AutoProcessor`
- Load processor:
  - `processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True)`
  - set `padding_side="right"`
  - set `pad_token = eos_token` if missing (same as Gemma3)

### Quant + attention
- Keep BitsAndBytesConfig identical to Gemma3.
- Load model with attention option:
  - Prefer `attn_implementation="flash_attention_2"` when enabled, else default or `sdpa`
  - Use clean fallback on exception, printing one short message.
  (Transformers docs show Qwen3-VL supports `attn_implementation` and provide examples. :contentReference[oaicite:3]{index=3})

### LoRA
- Same QLoRA prep:
  - `prepare_model_for_kbit_training`
  - `LoraConfig(... target_modules from cfg ...)`
  - `get_peft_model`
- Print trainable params summary via `_count_trainable_params`

### Collators
- Use `Qwen3VisionSFTCollator` for both train/infer:
  - train: `training=True`
  - eval: `training=False`
  - pass through cfg keys + system_prompt exactly like Gemma3

### “supervised tokens exist” check
- Same DataLoader one-batch assert:
  - print supervised tokens per row
  - assert any labels != -100

### Trainer
- Same `SFTConfig` fields as Gemma3 (including `dataset_kwargs={"skip_prepare_dataset": True}`, `remove_unused_columns=False`)
- Same `SFTTrainer`, callback, checkpoint resume logic.

### Pre/post eval
- Replace `evaluate_gemma` with `evaluate_qwen3` (added below).
- Log metrics to wandb under `gen_eval/pre_*` and `gen_eval/post_*`

### Save artifacts + metadata
- Same adapter save, processor save, `training_meta.json`, adapter file existence checks, and `log_run_metadata`

### Required try/except
Wrap whole `main()` and raise:
- `RuntimeError(f"Qwen3-VL SFT failed: {exc}") from exc`

---

## 3) Add Qwen3 eval function (uses trimmed decoding)

### File
Update: `src/utils/eval_sarcasm.py`

### Function
Add:
- `def evaluate_qwen3(model, processor, eval_dataset, eval_collator, cfg) -> dict:`

Implementation rules:
- Mirror `evaluate_gemma` structure (dataloader loop, generation, metric computation, return dict keys)
- Qwen3 recommended decode:
  - After `generated_ids = model.generate(**inputs, max_new_tokens=...)`
  - Trim prompt tokens per-row:
    - `generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]`
  - Decode with:
    - `processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)`
  (This exact trimming pattern is shown in HF docs/model card examples for Qwen3-VL. :contentReference[oaicite:4]{index=4})
- Ensure `inputs.pop("token_type_ids", None)` before `generate()` (also recommended). :contentReference[oaicite:5]{index=5}
- Keep eval params configurable via `cfg["eval"]` with sensible fallbacks:
  - `max_new_tokens` default 256
  - sampling defaults off unless cfg enables

Error handling:
- Wrap function in try/except and raise:
  - `RuntimeError(f"evaluate_qwen3 failed: {exc}") from exc`

---

## 4) Write collator test script at test/test_qwen3_collator.py
- Mimic test_aya_collator.py
Keep it short and behind try/except; no extra files.

---

## Notes (don’t over-engineer)
- No fake processor, no mocks, no extra abstractions.
- Reuse your existing dataset loader, subset selection, logging, and trainer utilities.
- Qwen3-VL is natively supported in Transformers and uses `Qwen3VLForConditionalGeneration` + `AutoProcessor`. :contentReference[oaicite:6]{index=6}
