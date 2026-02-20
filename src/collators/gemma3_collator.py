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

    def _build_labels(
        self,
        model_inputs: Dict[str, torch.Tensor],
        full_texts: List[str],
        prompt_texts: List[str],
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

    def _collate_group(
        self,
        examples: List[Dict[str, Any]],
        with_images: bool,
    ) -> Dict[str, torch.Tensor]:
        full_texts: List[str] = []
        prompt_texts: List[str] = []
        images_per_sample: List[List[Image.Image]] = []

        for example in examples:
            modality = self._get_modality(example)
            effective_modality = "text" if not with_images else modality
            user_text = self._user_text(example, effective_modality)
            user_messages = self._user_message(user_text, effective_modality)

            if with_images:
                images_per_sample.append([self._normalize_image(example[self.image_key])])

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
        if with_images:
            processor_kwargs["images"] = images_per_sample

        model_inputs = self.processor(**processor_kwargs)

        if self.training:
            model_inputs["labels"] = self._build_labels(
                model_inputs=model_inputs,
                full_texts=full_texts,
                prompt_texts=prompt_texts,
            )

        return model_inputs

    def _pad_2d(self, tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
        if tensor.shape[1] >= target_len:
            return tensor
        pad_cols = target_len - tensor.shape[1]
        pad = torch.full(
            (tensor.shape[0], pad_cols),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padding_side = getattr(self.processor.tokenizer, "padding_side", "right")
        if padding_side == "left":
            return torch.cat([pad, tensor], dim=1)
        return torch.cat([tensor, pad], dim=1)

    def _merge_group_outputs(
        self,
        batch_size: int,
        text_idxs: List[int],
        text_inputs: Dict[str, torch.Tensor],
        img_idxs: List[int],
        img_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        target_len = max(
            text_inputs["input_ids"].shape[1],
            img_inputs["input_ids"].shape[1],
        )

        text_input_ids = self._pad_2d(text_inputs["input_ids"], target_len, pad_id)
        img_input_ids = self._pad_2d(img_inputs["input_ids"], target_len, pad_id)
        text_attention = self._pad_2d(text_inputs["attention_mask"], target_len, 0)
        img_attention = self._pad_2d(img_inputs["attention_mask"], target_len, 0)

        merged_input_ids = torch.full(
            (batch_size, target_len),
            pad_id,
            dtype=img_input_ids.dtype,
            device=img_input_ids.device,
        )
        merged_attention = torch.zeros(
            (batch_size, target_len),
            dtype=img_attention.dtype,
            device=img_attention.device,
        )

        for local_idx, batch_idx in enumerate(text_idxs):
            merged_input_ids[batch_idx] = text_input_ids[local_idx]
            merged_attention[batch_idx] = text_attention[local_idx]
        for local_idx, batch_idx in enumerate(img_idxs):
            merged_input_ids[batch_idx] = img_input_ids[local_idx]
            merged_attention[batch_idx] = img_attention[local_idx]

        output: Dict[str, torch.Tensor] = {
            "input_ids": merged_input_ids,
            "attention_mask": merged_attention,
        }
        if "token_type_ids" in text_inputs and "token_type_ids" in img_inputs:
            text_token_type_ids = self._pad_2d(text_inputs["token_type_ids"], target_len, 0)
            img_token_type_ids = self._pad_2d(img_inputs["token_type_ids"], target_len, 0)
            merged_token_type_ids = torch.zeros(
                (batch_size, target_len),
                dtype=img_token_type_ids.dtype,
                device=img_token_type_ids.device,
            )
            for local_idx, batch_idx in enumerate(text_idxs):
                merged_token_type_ids[batch_idx] = text_token_type_ids[local_idx]
            for local_idx, batch_idx in enumerate(img_idxs):
                merged_token_type_ids[batch_idx] = img_token_type_ids[local_idx]
            output["token_type_ids"] = merged_token_type_ids

        if self.training:
            text_labels = self._pad_2d(text_inputs["labels"], target_len, -100)
            img_labels = self._pad_2d(img_inputs["labels"], target_len, -100)
            merged_labels = torch.full(
                (batch_size, target_len),
                -100,
                dtype=img_labels.dtype,
                device=img_labels.device,
            )
            for local_idx, batch_idx in enumerate(text_idxs):
                merged_labels[batch_idx] = text_labels[local_idx]
            for local_idx, batch_idx in enumerate(img_idxs):
                merged_labels[batch_idx] = img_labels[local_idx]
            output["labels"] = merged_labels

        # Keep pixel_values compact (image rows only). Gemma3 matches image embeddings
        # against image tokens across the full batch; padding fake image rows causes
        # token/embedding count mismatches in mixed text+image batches.
        if "pixel_values" in img_inputs:
            output["pixel_values"] = img_inputs["pixel_values"]

        # Other optional multimodal tensors remain batch-aligned if present.
        for key in ("aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask"):
            if key not in img_inputs:
                continue
            img_tensor = img_inputs[key]
            if key == "cross_attention_mask":
                if img_tensor.ndim < 2:
                    raise ValueError(
                        f"invalid cross_attention_mask rank={img_tensor.ndim}, "
                        f"expected at least 2 dimensions"
                    )
                seq_len = img_tensor.shape[1]
                padding_len = target_len - seq_len
                if padding_len > 0:
                    pad_shape = list(img_tensor.shape)
                    pad_shape[1] = padding_len
                    padding = torch.zeros(
                        pad_shape,
                        dtype=img_tensor.dtype,
                        device=img_tensor.device,
                    )
                    padding_side = getattr(self.processor.tokenizer, "padding_side", "right")
                    if padding_side == "left":
                        img_tensor = torch.cat([padding, img_tensor], dim=1)
                    else:
                        img_tensor = torch.cat([img_tensor, padding], dim=1)
                elif padding_len < 0:
                    raise ValueError(
                        f"cross_attention_mask seq_len ({seq_len}) exceeds merged target_len ({target_len})"
                    )
            merged_shape = (batch_size,) + tuple(img_tensor.shape[1:])
            merged_tensor = torch.zeros(
                merged_shape,
                dtype=img_tensor.dtype,
                device=img_tensor.device,
            )
            for local_idx, batch_idx in enumerate(img_idxs):
                merged_tensor[batch_idx] = img_tensor[local_idx]
            output[key] = merged_tensor

        return output

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        try:
            if not batch:
                raise ValueError("empty batch")

            text_idxs: List[int] = []
            img_idxs: List[int] = []
            for i, example in enumerate(batch):
                modality = self._get_modality(example)
                if modality in {"both", "image"}:
                    if self.image_key not in example:
                        raise ValueError(
                            f"missing `{self.image_key}` in `{modality}` example"
                        )
                    img_idxs.append(i)
                else:
                    text_idxs.append(i)

                if self.training:
                    target = (example.get(self.target_key) or "").strip()
                    if not target:
                        raise ValueError("missing target_json for training")

            if not text_idxs:
                img_examples = [batch[i] for i in img_idxs]
                return self._collate_group(img_examples, with_images=True)
            if not img_idxs:
                text_examples = [batch[i] for i in text_idxs]
                return self._collate_group(text_examples, with_images=False)

            text_examples = [batch[i] for i in text_idxs]
            img_examples = [batch[i] for i in img_idxs]
            text_inputs = self._collate_group(text_examples, with_images=False)
            img_inputs = self._collate_group(img_examples, with_images=True)
            return self._merge_group_outputs(
                batch_size=len(batch),
                text_idxs=text_idxs,
                text_inputs=text_inputs,
                img_idxs=img_idxs,
                img_inputs=img_inputs,
            )
        except Exception as exc:
            raise RuntimeError(f"Gemma3 collate failed: {exc}") from exc
