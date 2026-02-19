from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from config.queries import Queries


class Llama32VisionSFTCollator:
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

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        labels = input_ids.clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        assistant_header_ids = self.processor.tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )

        for i in range(labels.shape[0]):
            row_ids = input_ids[i].tolist()
            header_index = -1
            for idx in range(len(row_ids) - len(assistant_header_ids)):
                if row_ids[idx : idx + len(assistant_header_ids)] == assistant_header_ids:
                    header_index = idx + len(assistant_header_ids)
                    break

            if header_index != -1:
                labels[i, :header_index] = -100
            else:
                labels[i, :] = -100

        return labels

    def _collate_group(
        self,
        examples: List[Dict[str, Any]],
        with_images: bool,
    ) -> Dict[str, torch.Tensor]:
        full_texts: List[str] = []
        images_per_sample: List[List[Any]] = []

        for example in examples:
            modality = self._get_modality(example)
            effective_modality = "text" if not with_images else modality
            user_text = self._user_text(example, effective_modality)
            user_messages = self._user_message(user_text, effective_modality)

            if with_images:
                images_per_sample.append([example[self.image_key]])

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
            model_inputs["labels"] = self._build_labels(model_inputs["input_ids"])
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

        # Vision tensors only exist for image rows; text rows are zero-filled.
        for key in ("pixel_values", "aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask"):
            if key not in img_inputs:
                continue
            img_tensor = img_inputs[key]
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
        if not batch:
            raise ValueError("empty batch")

        text_idxs: List[int] = []
        img_idxs: List[int] = []
        for i, example in enumerate(batch):
            modality = self._get_modality(example)
            if modality in {"both", "image"}:
                if self.image_key not in example:
                    raise ValueError(f"missing `{self.image_key}` in `{modality}` example")
                img_idxs.append(i)
            else:
                text_idxs.append(i)

        if not text_idxs:
            img_examples = [batch[i] for i in img_idxs]
            return self._collate_group(img_examples, with_images=True)
        if not img_idxs:
            text_examples = [batch[i] for i in text_idxs]
            return self._collate_group(text_examples, with_images=False)

        # Llama processor cannot mix rows with/without image tokens in one call.
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
