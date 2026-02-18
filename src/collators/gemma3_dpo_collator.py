from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from config.queries import Queries


class Gemma3VisionDPOCollator:
    def __init__(
        self,
        processor: Any,
        max_length: Optional[int] = None,
        image_key: str = "image",
        prompt_key: str = "query",
        caption_key: str = "caption",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        system_prompt: Optional[str] = None,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.caption_key = caption_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.system_prompt = system_prompt or Queries().SYSTEM_PROMPT

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
    def _user_text(self, example: Dict[str, Any]) -> str:
        prompt = (example.get(self.prompt_key) or "").strip()
        caption = (example.get(self.caption_key) or "").strip()
        if prompt and caption:
            return f"{prompt}\nCAPTION: {caption}".strip()
        if caption:
            return f"CAPTION: {caption}".strip()
        return prompt

    def _user_message(self, user_text: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        )
        return messages

    def _normalize_image(self, image: Any) -> Image.Image:
        if image is None:
            raise ValueError("image is None")
        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except Exception as exc:
                shape = getattr(image, "shape", None)
                raise ValueError(
                    f"invalid image type for PIL conversion: {type(image)} shape={shape}"
                ) from exc
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _build_labels(
        self,
        full_texts: List[str],
        prompt_texts: List[str],
        model_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        labels = model_inputs["input_ids"].clone()
        labels[:] = -100
        pad_id = self.processor.tokenizer.pad_token_id

        for i, (full_text, prompt_text) in enumerate(zip(full_texts, prompt_texts)):
            if not full_text.startswith(prompt_text):
                raise ValueError("prompt_text is not a prefix of full_text")
            target_text = full_text[len(prompt_text) :]
            target_ids = self.processor.tokenizer(
                target_text, add_special_tokens=False
            )["input_ids"]
            if not target_ids:
                raise ValueError("target text tokenized to empty sequence")

            full_ids = model_inputs["input_ids"][i].tolist()
            target_len = len(target_ids)
            start_idx = None
            if full_ids[-target_len:] == target_ids:
                start_idx = len(full_ids) - target_len
            else:
                for idx in range(len(full_ids) - target_len + 1):
                    if full_ids[idx : idx + target_len] == target_ids:
                        start_idx = idx
                        break

            if start_idx is None:
                raise ValueError("target tokens not found in input_ids")

            labels[i, start_idx : start_idx + target_len] = model_inputs["input_ids"][
                i, start_idx : start_idx + target_len
            ]

        if pad_id is not None:
            labels[model_inputs["input_ids"] == pad_id] = -100

        return labels

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        try:
            if not batch:
                raise ValueError("empty batch")

            for example in batch:
                if self.image_key not in example:
                    raise ValueError(f"missing `{self.image_key}` in example")
                chosen = (example.get(self.chosen_key) or "").strip()
                rejected = (example.get(self.rejected_key) or "").strip()
                if not chosen or not rejected:
                    raise ValueError("missing chosen/rejected for DPO training")

            images = [self._normalize_image(example[self.image_key]) for example in batch]
            images_per_sample = [[image] for image in images]
            user_texts = [self._user_text(example) for example in batch]

            prompt_texts: List[str] = []
            chosen_texts: List[str] = []
            rejected_texts: List[str] = []

            for example, user_text in zip(batch, user_texts):
                user_messages = self._user_message(user_text)
                prompt_texts.append(
                    self.processor.apply_chat_template(
                        user_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

                chosen = (example.get(self.chosen_key) or "").strip()
                rejected = (example.get(self.rejected_key) or "").strip()

                chosen_messages = user_messages + [
                    {"role": "assistant", "content": [{"type": "text", "text": chosen}]}
                ]
                rejected_messages = user_messages + [
                    {"role": "assistant", "content": [{"type": "text", "text": rejected}]}
                ]

                chosen_texts.append(
                    self.processor.apply_chat_template(
                        chosen_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
                rejected_texts.append(
                    self.processor.apply_chat_template(
                        rejected_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )

            prompt_inputs = self.processor(
                text=prompt_texts,
                images=images_per_sample,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            chosen_inputs = self.processor(
                text=chosen_texts,
                images=images_per_sample,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            rejected_inputs = self.processor(
                text=rejected_texts,
                images=images_per_sample,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            chosen_labels = self._build_labels(chosen_texts, prompt_texts, chosen_inputs)
            rejected_labels = self._build_labels(rejected_texts, prompt_texts, rejected_inputs)

            output: Dict[str, torch.Tensor] = {
                "prompt_input_ids": prompt_inputs["input_ids"],
                "prompt_attention_mask": prompt_inputs["attention_mask"],
                "chosen_input_ids": chosen_inputs["input_ids"],
                "chosen_attention_mask": chosen_inputs["attention_mask"],
                "rejected_input_ids": rejected_inputs["input_ids"],
                "rejected_attention_mask": rejected_inputs["attention_mask"],
                "chosen_labels": chosen_labels,
                "rejected_labels": rejected_labels,
            }

            for key in ("pixel_values", "token_type_ids"):
                if key in chosen_inputs:
                    output[key] = chosen_inputs[key]

            return output
        except Exception as exc:
            print(f"[CRITICAL] Gemma3 DPO collate failed: {exc}")
            raise
