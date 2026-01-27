from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from config.queries import Queries


class Qwen3VisionSFTCollator:
    """
    Qwen3-VL SFT collator (single-image).
    - Builds chat text via processor.apply_chat_template
    - Tokenizes full inputs once (text + images)
    - Masks prompt + padding tokens in labels so loss is only on assistant output
      (locate '<|im_start|>assistant' marker in input_ids; robust to vision tokens)
    """

    def __init__(
        self,
        processor: Any,
        training: bool = True,
        max_length: Optional[int] = None,
        image_key: str = "image",
        query_key: str = "query",
        caption_key: str = "caption",
        target_key: str = "target_json",
        system_prompt: Optional[str] = None,
        max_image_side: int = 1536,
        max_image_pixels: int = 6_000_000,
    ) -> None:
        self.processor = processor
        self.training = training
        self.max_length = max_length
        self.image_key = image_key
        self.query_key = query_key
        self.caption_key = caption_key
        self.target_key = target_key
        self.system_prompt = system_prompt or Queries().SYSTEM_PROMPT

        self.max_image_side = max_image_side
        self.max_image_pixels = max_image_pixels

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _user_text(self, example: Dict[str, Any]) -> str:
        query = (example.get(self.query_key) or "").strip()
        caption = (example.get(self.caption_key) or "").strip()
        if caption:
            return f"{query}\nCAPTION: {caption}".strip()
        return query

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
        # Downscale by max side and total pixel cap to stabilize VRAM usage.
        pil = pil.convert("RGB")
        w, h = pil.size

        scale_side = min(1.0, self.max_image_side / float(max(w, h)))
        area = w * h
        scale_area = (self.max_image_pixels / float(area)) ** 0.5 if area > self.max_image_pixels else 1.0
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
        """Return start index of needle in haystack, or -1."""
        n = len(needle)
        if n == 0:
            return -1
        for j in range(0, len(haystack) - n + 1):
            if haystack[j : j + n] == needle:
                return j
        return -1

    def _assistant_marker_ids(self) -> List[List[int]]:
        """
        Return likely tokenizations for the assistant start marker.
        Different templates/models sometimes include a newline.
        """
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            raise ValueError("processor.tokenizer is required")

        candidates = [
            "<|im_start|>assistant\n",
            "<|im_start|>assistant",
        ]
        return [tok(c, add_special_tokens=False)["input_ids"] for c in candidates]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        try:
            if not batch:
                raise ValueError("empty batch")

            for example in batch:
                if self.image_key not in example:
                    raise ValueError(f"missing `{self.image_key}` in example")
                if self.training:
                    target = (example.get(self.target_key) or "").strip()
                    if not target:
                        raise ValueError("missing target_json for training")

            images = [self._normalize_image(example[self.image_key]) for example in batch]
            user_texts = [self._user_text(example) for example in batch]

            full_texts: List[str] = []
            for example, user_text in zip(batch, user_texts):
                user_messages = self._user_message(user_text)

                if self.training:
                    target = (example.get(self.target_key) or "").strip()
                    full_messages = user_messages + [
                        {"role": "assistant", "content": [{"type": "text", "text": target}]}
                    ]
                    full_texts.append(
                        self.processor.apply_chat_template(
                            full_messages,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                    )
                    # print(f"\n\nfrom collator:{self.processor.apply_chat_template(full_messages[-1:], tokenize=False, add_generation_prompt=False)}\n\n")
                else:
                    full_texts.append(
                        self.processor.apply_chat_template(
                            user_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )

            # Flat list: one image per sample
            model_inputs = self.processor(
                text=full_texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            model_inputs.pop("token_type_ids", None)

            if self.training:
                tok = getattr(self.processor, "tokenizer", None)
                if tok is None:
                    raise ValueError("processor.tokenizer is required")
                pad_id = tok.pad_token_id
                if pad_id is None:
                    raise ValueError("tokenizer.pad_token_id is None (pad_token not set)")

                input_ids = model_inputs["input_ids"]
                labels = input_ids.clone()

                # Mask padding
                labels[labels == pad_id] = -100

                # --- locate assistant marker in *input_ids* (robust to vision tokens) ---
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
                        # means truncation removed the marker or template differs
                        raise RuntimeError(
                            "Could not find '<|im_start|>assistant' marker in input_ids. "
                            "Likely max_length truncation (prompt too long) or a different chat template."
                        )

                    prompt_len = min(start + marker_len, seqlen)
                    labels[i, :prompt_len] = -100

                model_inputs["labels"] = labels

            return model_inputs

        except Exception as exc:
            raise RuntimeError(f"Qwen3 collate failed: {exc}") from exc
