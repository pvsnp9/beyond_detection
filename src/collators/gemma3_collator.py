from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from config.queries import Queries


class Gemma3VisionSFTCollator:
    def __init__(
        self,
        processor: Any,
        training: bool = True,
        max_length: Optional[int] = None,
        image_key: str = "image",
        modality_key: str = "modality",
        query_key: str = "query",
        caption_key: str = "caption",
        target_key: str = "target_json",
        system_prompt: Optional[str] = None,
    ) -> None:
        self.processor = processor
        self.training = training
        self.max_length = max_length
        self.image_key = image_key
        self.modality_key = modality_key
        self.query_key = query_key
        self.caption_key = caption_key
        self.target_key = target_key
        self.system_prompt = system_prompt or Queries().SYSTEM_PROMPT
        
        if getattr(self.processor, "tokenizer", None) is not None:
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def _get_modality(self, example: Dict[str, Any]) -> str:
        modality = (example.get(self.modality_key) or "both").strip().lower()
        if modality not in {"both", "text", "image"}:
            raise ValueError(f"invalid modality `{modality}`")
        return modality

    def _user_text(self, example: Dict[str, Any], modality: str = "both") -> str:
        query = (example.get(self.query_key) or "").strip()
        if modality == "image":
            return query
        caption = (example.get(self.caption_key) or "").strip()
        if caption:
            return f"{query}\nCAPTION: {caption}".strip()
        return query

    def _user_message(self, user_text: str, modality: str = "both") -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        if modality == "text":
            user_content = [{"type": "text", "text": user_text}]
        else:
            user_content = [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ]
        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )
        return messages

    def _prompt_length(self, image: Any, prompt_text: str, modality: str = "both") -> int:
        processor_kwargs: Dict[str, Any] = {
            "text": prompt_text,
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": self.max_length,
        }
        if modality in {"both", "image"}:
            processor_kwargs["images"] = [[image]]
        prompt_inputs = self.processor(**processor_kwargs)
        return prompt_inputs["input_ids"].shape[-1]

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

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        try:
            if not batch:
                raise ValueError("empty batch")

            for example in batch:
                modality = self._get_modality(example)
                if modality in {"both", "image"} and self.image_key not in example:
                    raise ValueError(
                        f"missing `{self.image_key}` in `{modality}` example"
                    )
                if self.training:
                    target = (example.get(self.target_key) or "").strip()
                    if not target:
                        raise ValueError("missing target_json for training")

            images_per_sample: List[List[Image.Image]] = []
            has_any_image = False
            full_texts: List[str] = []
            prompt_texts: List[str] = []

            for example in batch:
                modality = self._get_modality(example)
                user_text = self._user_text(example, modality)
                user_messages = self._user_message(user_text, modality)

                if modality in {"both", "image"}:
                    images_per_sample.append(
                        [self._normalize_image(example[self.image_key])]
                    )
                    has_any_image = True
                else:
                    images_per_sample.append([])

                if self.training:
                    target = (example.get(self.target_key) or "").strip()
                    full_messages = user_messages + [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": target}],
                        }
                    ]
                    full_texts.append(
                        self.processor.apply_chat_template(
                            full_messages,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                    )
                    prompt_texts.append(
                        self.processor.apply_chat_template(
                            user_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )
                else:
                    full_texts.append(
                        self.processor.apply_chat_template(
                            user_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    )

            processor_kwargs: Dict[str, Any] = {
                "text": full_texts,
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
                "max_length": self.max_length,
            }
            if has_any_image:
                processor_kwargs["images"] = images_per_sample

            model_inputs = self.processor(**processor_kwargs)

            if self.training:
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

                model_inputs["labels"] = labels

            return model_inputs
        except Exception as exc:
            raise RuntimeError(f"Gemma3 collate failed: {exc}") from exc
