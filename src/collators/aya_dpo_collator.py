from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from PIL import Image

from config.queries import Queries


class AyaVisionDPOCollator:
    def __init__(
        self,
        processor: Any,
        max_length: Optional[int] = None,
        image_key: str = "image",
        prompt_key: str = "prompt",
        caption_key: str = "caption",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        system_prompt: Optional[str] = None,
        ignore_index: int = -100,
        pad_to_multiple_of: Optional[int] = None,
        mask_vision_tokens: bool = True,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.caption_key = caption_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.system_prompt = system_prompt or Queries().SYSTEM_PROMPT
        self.ignore_index = ignore_index
        self.pad_to_multiple_of = pad_to_multiple_of
        self.mask_vision_tokens = mask_vision_tokens

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._vision_token_ids: Set[int] = self._infer_vision_token_ids()

    def _infer_vision_token_ids(self) -> Set[int]:
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            return set()

        vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
        candidate_tokens = [
            "<image>",
            "<|START_OF_IMG|>",
            "<|END_OF_IMG|>",
            "<|IMG_PATCH|>",
            "<|IMG_LINE_BREAK|>",
            "TILE",
            "TILE_GLOBAL",
        ]

        ids: Set[int] = set()
        unk_id = getattr(tok, "unk_token_id", None)

        for t in candidate_tokens:
            if t in vocab:
                tid = tok.convert_tokens_to_ids(t)
                if tid is not None and tid != unk_id:
                    ids.add(int(tid))

        return ids

    def _normalize_image(self, image: Any) -> Image.Image:
        if image is None:
            raise ValueError("image is None")

        if isinstance(image, Image.Image):
            pil_img = image
        else:
            try:
                arr = np.asarray(image)
                if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    arr = arr[:, :, 0]
                pil_img = Image.fromarray(arr)
            except Exception as exc:
                shape = getattr(image, "shape", None)
                raise ValueError(
                    f"invalid image type for PIL conversion: {type(image)} shape={shape}"
                ) from exc

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img = pil_img.resize((364, 364))
        return pil_img

    def _user_text(self, example: Dict[str, Any]) -> str:
        prompt = (example.get(self.prompt_key) or "").strip()
        caption = (example.get(self.caption_key) or "").strip()
        if prompt and caption:
            return f"{prompt}\nCAPTION: {caption}".strip()
        if caption:
            return f"CAPTION: {caption}".strip()
        return prompt

    def _messages(self, user_text: str, image: Image.Image) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = []
        if self.system_prompt:
            msgs.append(
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            )
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": image},
                ],
            }
        )
        return msgs

    def _apply_template(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        add_generation_prompt: bool,
    ) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, Any] = dict(
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        if self.pad_to_multiple_of is not None:
            kwargs["pad_to_multiple_of"] = self.pad_to_multiple_of
        return self.processor.apply_chat_template(messages_batch, **kwargs)

    def _build_labels(
        self,
        full_texts: List[str],
        prompt_texts: List[str],
        model_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        labels = model_inputs["input_ids"].clone()
        labels[:] = self.ignore_index

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)

        bs = model_inputs["input_ids"].shape[0]
        for i in range(bs):
            full_text = full_texts[i]
            prompt_text = prompt_texts[i]
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
            labels[model_inputs["input_ids"] == pad_id] = self.ignore_index

        if self.mask_vision_tokens and self._vision_token_ids:
            for vid in self._vision_token_ids:
                labels[model_inputs["input_ids"] == vid] = self.ignore_index

        return labels

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        try:
            if not batch:
                raise ValueError("empty batch")

            for ex in batch:
                if self.image_key not in ex:
                    raise ValueError(f"missing `{self.image_key}` in example")
                chosen = (ex.get(self.chosen_key) or "").strip()
                rejected = (ex.get(self.rejected_key) or "").strip()
                if not chosen or not rejected:
                    raise ValueError("missing chosen/rejected for DPO training")

            images: List[Image.Image] = [self._normalize_image(ex[self.image_key]) for ex in batch]
            user_texts: List[str] = [self._user_text(ex) for ex in batch]

            prompt_messages_batch: List[List[Dict[str, Any]]] = []
            chosen_messages_batch: List[List[Dict[str, Any]]] = []
            rejected_messages_batch: List[List[Dict[str, Any]]] = []
            prompt_texts: List[str] = []
            chosen_texts: List[str] = []
            rejected_texts: List[str] = []

            for ex, utext, img in zip(batch, user_texts, images):
                user_msgs = self._messages(utext, img)
                prompt_messages_batch.append(user_msgs)
                prompt_texts.append(
                    self.processor.apply_chat_template(
                        user_msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

                chosen = (ex.get(self.chosen_key) or "").strip()
                rejected = (ex.get(self.rejected_key) or "").strip()
                chosen_msgs = user_msgs + [
                    {"role": "assistant", "content": [{"type": "text", "text": chosen}]}
                ]
                rejected_msgs = user_msgs + [
                    {"role": "assistant", "content": [{"type": "text", "text": rejected}]}
                ]
                chosen_messages_batch.append(chosen_msgs)
                rejected_messages_batch.append(rejected_msgs)
                chosen_texts.append(
                    self.processor.apply_chat_template(
                        chosen_msgs,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
                rejected_texts.append(
                    self.processor.apply_chat_template(
                        rejected_msgs,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )

            prompt_inputs = self._apply_template(
                prompt_messages_batch,
                add_generation_prompt=True,
            )
            chosen_inputs = self._apply_template(
                chosen_messages_batch,
                add_generation_prompt=False,
            )
            rejected_inputs = self._apply_template(
                rejected_messages_batch,
                add_generation_prompt=False,
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

            for key in chosen_inputs.keys():
                if key in ("input_ids", "attention_mask"):
                    continue
                if key in chosen_inputs:
                    output[key] = chosen_inputs[key]

            return output
        except Exception as exc:
            print(f"[CRITICAL] Aya DPO collate failed: {exc}")
            raise
