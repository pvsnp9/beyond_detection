from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from config.queries import Queries


class Qwen3VisionDPOCollator:
    """
    Qwen3-VL DPO collator (single-image).
    - Builds prompt + chosen/rejected chat text via processor.apply_chat_template
    - Tokenizes prompt/chosen/rejected inputs separately
    - Masks prompt + padding tokens in labels so loss is only on assistant output
    """

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
        max_image_side: int = 1536,
        max_image_pixels: int = 6_000_000,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.caption_key = caption_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.system_prompt = system_prompt or Queries().SYSTEM_PROMPT
        self.max_image_side = max_image_side
        self.max_image_pixels = max_image_pixels

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

    def _resize_pil(self, pil: Image.Image) -> Image.Image:
        pil = pil.convert("RGB")
        w, h = pil.size
        scale_side = min(1.0, self.max_image_side / float(max(w, h)))
        area = w * h
        scale_area = (
            (self.max_image_pixels / float(area)) ** 0.5
            if area > self.max_image_pixels
            else 1.0
        )
        scale = min(scale_side, scale_area)
        if scale < 1.0:
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            pil = pil.resize((nw, nh), resample=Image.BICUBIC)
        return pil

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
        return self._resize_pil(image)

    def _find_subsequence(self, haystack: List[int], needle: List[int]) -> int:
        n = len(needle)
        if n == 0:
            return -1
        for j in range(0, len(haystack) - n + 1):
            if haystack[j : j + n] == needle:
                return j
        return -1

    def _assistant_marker_ids(self) -> List[List[int]]:
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            raise ValueError("processor.tokenizer is required")
        candidates = [
            "<|im_start|>assistant\n",
            "<|im_start|>assistant",
        ]
        return [tok(c, add_special_tokens=False)["input_ids"] for c in candidates]

    def _build_labels(self, model_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            raise ValueError("processor.tokenizer is required")
        pad_id = tok.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id is None (pad_token not set)")

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()
        labels[labels == pad_id] = -100

        marker_id_variants = self._assistant_marker_ids()
        bs, seqlen = labels.shape
        for i in range(bs):
            ids = input_ids[i].tolist()
            start = -1
            marker_len = 0
            for marker_ids in marker_id_variants:
                j = self._find_subsequence(ids, marker_ids)
                if j != -1:
                    start = j
                    marker_len = len(marker_ids)
                    break
            if start == -1:
                raise RuntimeError(
                    "Could not find '<|im_start|>assistant' marker in input_ids. "
                    "Likely max_length truncation (prompt too long) or a different chat template."
                )
            prompt_len = min(start + marker_len, seqlen)
            labels[i, :prompt_len] = -100

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
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            chosen_inputs = self.processor(
                text=chosen_texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            rejected_inputs = self.processor(
                text=rejected_texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )

            prompt_inputs.pop("token_type_ids", None)
            chosen_inputs.pop("token_type_ids", None)
            rejected_inputs.pop("token_type_ids", None)

            chosen_labels = self._build_labels(chosen_inputs)
            rejected_labels = self._build_labels(rejected_inputs)

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

            for key in ("pixel_values", "image_grid_thw"):
                if key in chosen_inputs:
                    output[key] = chosen_inputs[key]

            return output

        except Exception as exc:
            raise RuntimeError(f"Qwen3 DPO collate failed: {exc}") from exc
