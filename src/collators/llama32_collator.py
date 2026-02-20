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
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            })
        
        user_content = []
        if modality != "text":
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": user_text})
        
        messages.append({"role": "user", "content": user_content})
        return messages

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

    def _pad_2d(self, tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
        if tensor.shape[1] >= target_len:
            return tensor
        pad_cols = target_len - tensor.shape[1]
        pad = torch.full((tensor.shape[0], pad_cols), pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([pad, tensor], dim=1)

    def _collate_group(self, examples: List[Dict[str, Any]], with_images: bool) -> Dict[str, torch.Tensor]:
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
                full_messages = user_messages + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]
                full_texts.append(self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False))
            else:
                full_texts.append(self.processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True))

        processor_kwargs = {
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

    def _merge_group_outputs(
        self,
        batch_size: int,
        text_idxs: List[int],
        text_inputs: Dict[str, torch.Tensor],
        img_idxs: List[int],
        img_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pad_id = self.processor.tokenizer.pad_token_id or 0
        target_len = max(text_inputs["input_ids"].shape[1], img_inputs["input_ids"].shape[1])

        # Align text and image IDs/Attention (Left Padding)
        t_ids = self._pad_2d(text_inputs["input_ids"], target_len, pad_id)
        i_ids = self._pad_2d(img_inputs["input_ids"], target_len, pad_id)
        t_mask = self._pad_2d(text_inputs["attention_mask"], target_len, 0)
        i_mask = self._pad_2d(img_inputs["attention_mask"], target_len, 0)

        merged_ids = torch.full((batch_size, target_len), pad_id, dtype=i_ids.dtype, device=i_ids.device)
        merged_mask = torch.zeros((batch_size, target_len), dtype=i_mask.dtype, device=i_mask.device)

        for l_idx, b_idx in enumerate(text_idxs):
            merged_ids[b_idx], merged_mask[b_idx] = t_ids[l_idx], t_mask[l_idx]
        for l_idx, b_idx in enumerate(img_idxs):
            merged_ids[b_idx], merged_mask[b_idx] = i_ids[l_idx], i_mask[l_idx]

        output = {"input_ids": merged_ids, "attention_mask": merged_mask}

        if self.training:
            t_labels = self._pad_2d(text_inputs["labels"], target_len, -100)
            i_labels = self._pad_2d(img_inputs["labels"], target_len, -100)
            merged_labels = torch.full((batch_size, target_len), -100, dtype=i_labels.dtype, device=i_labels.device)
            for l_idx, b_idx in enumerate(text_idxs): merged_labels[b_idx] = t_labels[l_idx]
            for l_idx, b_idx in enumerate(img_idxs): merged_labels[b_idx] = i_labels[l_idx]
            output["labels"] = merged_labels

        # Fix Vision Tensors: align cross_attention_mask sequence axis with target_len
        vision_keys = ("pixel_values", "aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask")
        for key in vision_keys:
            if key not in img_inputs: continue
            
            ref_tensor = img_inputs[key]
            # Mllama cross_attention_mask shape is (batch, seq_len, max_num_images, max_num_tiles).
            # Left-pad on seq_len so text and vision attention stay position-aligned.
            if key == "cross_attention_mask":
                if ref_tensor.ndim < 2:
                    raise ValueError(
                        f"invalid cross_attention_mask rank={ref_tensor.ndim}, "
                        f"expected at least 2 dimensions"
                    )
                seq_len = ref_tensor.shape[1]
                padding_len = target_len - seq_len
                if padding_len > 0:
                    pad_shape = list(ref_tensor.shape)
                    pad_shape[1] = padding_len
                    # 0 in cross_attention_mask means "no attention" for the left padding
                    padding = torch.zeros(pad_shape, dtype=ref_tensor.dtype, device=ref_tensor.device)
                    ref_tensor = torch.cat([padding, ref_tensor], dim=1)
                elif padding_len < 0:
                    raise ValueError(
                        f"cross_attention_mask seq_len ({seq_len}) exceeds merged target_len ({target_len})"
                    )

            merged_shape = (batch_size,) + tuple(ref_tensor.shape[1:])
            merged_v_tensor = torch.zeros(merged_shape, dtype=ref_tensor.dtype, device=ref_tensor.device)
            for l_idx, b_idx in enumerate(img_idxs):
                merged_v_tensor[b_idx] = ref_tensor[l_idx]
            output[key] = merged_v_tensor

        return output

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("empty batch")

        text_idxs, img_idxs = [], []
        for i, ex in enumerate(batch):
            modality = self._get_modality(ex)
            if modality in {"both", "image"}:
                if self.image_key not in ex:
                    raise ValueError(f"missing `{self.image_key}` in `{modality}` example")
                img_idxs.append(i)
            else:
                text_idxs.append(i)
            if self.training:
                target = (ex.get(self.target_key) or "").strip()
                if not target:
                    raise ValueError("missing target_json for training")

        if not text_idxs: return self._collate_group([batch[i] for i in img_idxs], True)
        if not img_idxs: return self._collate_group([batch[i] for i in text_idxs], False)

        text_inputs = self._collate_group([batch[i] for i in text_idxs], False)
        img_inputs = self._collate_group([batch[i] for i in img_idxs], True)
        
        return self._merge_group_outputs(len(batch), text_idxs, text_inputs, img_idxs, img_inputs)
