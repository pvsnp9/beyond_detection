from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from trl import DPOTrainer
from trl.trainer.utils import flush_left, flush_right, pad, pad_to_length, selective_log_softmax


@dataclass
class Llama32DataCollatorForPreference(DataCollatorMixin):
    """Preference collator that keeps Llama 3.2 vision tensors aligned with prompt padding."""

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]

        output = {
            "prompt_input_ids": pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"),
            "prompt_attention_mask": pad(prompt_attention_mask, padding_value=0, padding_side="left"),
            "chosen_input_ids": pad(chosen_input_ids, padding_value=self.pad_token_id),
            "chosen_attention_mask": pad(chosen_attention_mask, padding_value=0),
            "rejected_input_ids": pad(rejected_input_ids, padding_value=self.pad_token_id),
            "rejected_attention_mask": pad(rejected_attention_mask, padding_value=0),
        }

        if "pixel_values" in examples[0]:
            pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor([example["image_sizes"] for example in examples])
        if "token_type_ids" in examples[0]:
            token_type_ids = [torch.tensor(example["token_type_ids"]) for example in examples]
            output["token_type_ids"] = pad(token_type_ids, padding_value=0, padding_side="left")

        if "aspect_ratio_ids" in examples[0]:
            aspect_ratio_ids = [torch.tensor(example["aspect_ratio_ids"]) for example in examples]
            output["aspect_ratio_ids"] = pad(aspect_ratio_ids, padding_value=0)
        if "aspect_ratio_mask" in examples[0]:
            aspect_ratio_mask = [torch.tensor(example["aspect_ratio_mask"]) for example in examples]
            output["aspect_ratio_mask"] = pad(aspect_ratio_mask, padding_value=0)
        if "cross_attention_mask" in examples[0]:
            cross_attention_mask = [torch.tensor(example["cross_attention_mask"]) for example in examples]
            output["cross_attention_mask"] = pad(cross_attention_mask, padding_value=0, padding_side="left")

        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = torch.tensor([example["ref_chosen_logps"] for example in examples])
            output["ref_rejected_logps"] = torch.tensor([example["ref_rejected_logps"] for example in examples])

        return output


class Llama32DPOTrainer(DPOTrainer):
    """DPO trainer variant that carries Llama 3.2 vision tensors through concat + forward."""

    @staticmethod
    def _first_item(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return value[0]
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return value[0]
        if torch.is_tensor(value):
            return value[0]
        return value

    @staticmethod
    def _to_python(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    @staticmethod
    def _pad_sequence_axis(value: Any, pad_left: int = 0, pad_right: int = 0, pad_value: int = 0) -> Any:
        tensor = torch.tensor(value)
        if pad_left > 0:
            left = torch.full((pad_left, *tensor.shape[1:]), pad_value, dtype=tensor.dtype)
            tensor = torch.cat([left, tensor], dim=0)
        if pad_right > 0:
            right = torch.full((pad_right, *tensor.shape[1:]), pad_value, dtype=tensor.dtype)
            tensor = torch.cat([tensor, right], dim=0)
        return tensor.tolist()

    @staticmethod
    def process_row(
        features: dict[str, Any],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int] | list[float] | int]:
        """
        Vision-aware row processing that preserves Llama-specific keys from dpo_data-formatted rows.
        """
        processor, tokenizer = processing_class, processing_class.tokenizer
        processed_features = processor(images=features["images"], text=features["prompt"], add_special_tokens=False)

        prompt_input_ids = Llama32DPOTrainer._to_python(
            Llama32DPOTrainer._first_item(processed_features["input_ids"])
        )
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        prepend_prompt = 0
        append_prompt = 0
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
                prepend_prompt = 1
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
                append_prompt = 1
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        output: dict[str, Any] = {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        for key in (
            "pixel_values",
            "pixel_attention_mask",
            "image_sizes",
            "token_type_ids",
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "cross_attention_mask",
        ):
            if key in processed_features:
                output[key] = Llama32DPOTrainer._to_python(
                    Llama32DPOTrainer._first_item(processed_features[key])
                )

        if "cross_attention_mask" in output:
            cross_attention_mask = torch.as_tensor(output["cross_attention_mask"])
            if cross_attention_mask.dim() == 4 and cross_attention_mask.shape[0] == 1:
                cross_attention_mask = cross_attention_mask.squeeze(0)
            if cross_attention_mask.dim() != 3:
                raise ValueError(
                    "cross_attention_mask row must have shape (seq_len, max_num_images, max_num_tiles); "
                    f"got shape={tuple(cross_attention_mask.shape)}. "
                    "Likely leading batch axis leak from process_row/cached tokenized rows."
                )
            output["cross_attention_mask"] = cross_attention_mask.tolist()

        if prepend_prompt or append_prompt:
            if "token_type_ids" in output:
                output["token_type_ids"] = (
                    [0] * prepend_prompt + output["token_type_ids"] + [0] * append_prompt
                )
            if "cross_attention_mask" in output:
                output["cross_attention_mask"] = Llama32DPOTrainer._pad_sequence_axis(
                    output["cross_attention_mask"],
                    pad_left=prepend_prompt,
                    pad_right=append_prompt,
                    pad_value=0,
                )

        if max_prompt_length is not None:
            # Keep the prompt prefix so image placeholder tokens stay aligned with vision features.
            output["prompt_input_ids"] = output["prompt_input_ids"][:max_prompt_length]
            if "token_type_ids" in output:
                output["token_type_ids"] = output["token_type_ids"][:max_prompt_length]
            if "cross_attention_mask" in output:
                output["cross_attention_mask"] = output["cross_attention_mask"][:max_prompt_length]
        if max_completion_length is not None:
            output["chosen_input_ids"] = output["chosen_input_ids"][:max_completion_length]
            output["rejected_input_ids"] = output["rejected_input_ids"][:max_completion_length]

        return output

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        extra_columns = [
            "pixel_values",
            "pixel_attention_mask",
            "image_sizes",
            "token_type_ids",
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "cross_attention_mask",
        ]
        for column in extra_columns:
            if column not in self._signature_columns:
                self._signature_columns.append(column)

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, list | torch.LongTensor], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        output = DPOTrainer.concatenated_inputs(batch, padding_value)
        for key in ("aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask"):
            if key in batch:
                output[key] = torch.cat([batch[key], batch[key]], dim=0)
        return output

    @staticmethod
    def _flush_left_nd(mask: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() < 2:
            raise ValueError("tensor rank must be at least 2 for flush-left alignment")
        if tensor.shape[:2] != mask.shape:
            raise ValueError(
                f"mask/tensor seq mismatch for flush-left: mask={tuple(mask.shape)}, tensor={tuple(tensor.shape)}"
            )

        n, m = mask.shape
        first_non_zero = mask.argmax(dim=1)
        pos = torch.arange(m, device=mask.device).unsqueeze(0)
        idx_roll = (pos + first_non_zero.unsqueeze(1)) % m
        idx_expand = idx_roll.view(n, m, *([1] * (tensor.dim() - 2))).expand_as(tensor)
        tensor_roll = tensor.gather(1, idx_expand)

        mask_roll = mask.gather(1, idx_roll)
        col_sums = mask_roll.sum(dim=0)
        empty_cols = col_sums == 0
        first_empty_col = int(empty_cols.to(torch.int8).argmax()) if empty_cols.any() else m
        return tensor_roll[:, :first_empty_col]

    @staticmethod
    def _flush_right_nd(mask: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() < 2:
            raise ValueError("tensor rank must be at least 2 for flush-right alignment")
        if tensor.shape[:2] != mask.shape:
            raise ValueError(
                f"mask/tensor seq mismatch for flush-right: mask={tuple(mask.shape)}, tensor={tuple(tensor.shape)}"
            )

        n, m = mask.shape
        flipped_mask = torch.fliplr(mask)
        first_non_zero = flipped_mask.argmax(dim=1)
        pos = torch.arange(m, device=mask.device).unsqueeze(0)
        idx_roll = (pos - first_non_zero.unsqueeze(1)) % m
        idx_expand = idx_roll.view(n, m, *([1] * (tensor.dim() - 2))).expand_as(tensor)
        tensor_roll = tensor.gather(1, idx_expand)

        mask_roll = mask.gather(1, idx_roll)
        col_sums = mask_roll.sum(dim=0)
        non_empty_cols = col_sums != 0
        first_non_empty_col = int(non_empty_cols.to(torch.int8).argmax()) if non_empty_cols.any() else m
        return tensor_roll[:, first_non_empty_col:]

    @staticmethod
    def _expand_cross_attention_for_completion(
        cross_attention_mask: torch.Tensor,
        completion_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        completion_len = completion_attention_mask.shape[1]
        if completion_len <= 0:
            return cross_attention_mask
        pad_shape = list(cross_attention_mask.shape)
        pad_shape[1] = completion_len
        completion_padding = torch.zeros(pad_shape, dtype=cross_attention_mask.dtype, device=cross_attention_mask.device)
        return torch.cat([cross_attention_mask, completion_padding], dim=1)

    @staticmethod
    def _llama_vision_model_kwargs(concatenated_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        vision_keys = (
            "pixel_values",
            "pixel_attention_mask",
            "image_sizes",
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "cross_attention_mask",
        )
        return {key: concatenated_batch[key] for key in vision_keys if key in concatenated_batch}

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, list | torch.LongTensor], is_ref_model: bool = False
    ) -> dict[str, torch.Tensor]:
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)

        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
        model_kwargs.update(self._llama_vision_model_kwargs(concatenated_batch))

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            if "token_type_ids" in concatenated_batch:
                prompt_token_type_ids = concatenated_batch["token_type_ids"]
                token_type_ids = pad_to_length(prompt_token_type_ids, input_ids.shape[1], 0)
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            cross_attention_mask = model_kwargs.pop("cross_attention_mask", None)
            if cross_attention_mask is not None:
                if cross_attention_mask.dim() != 4:
                    raise ValueError(
                        "cross_attention_mask batch tensor must have shape "
                        "(batch, seq_len, max_num_images, max_num_tiles); "
                        f"got shape={tuple(cross_attention_mask.shape)}. "
                        "Likely leading batch axis leak from process_row/cached tokenized rows."
                    )
                if cross_attention_mask.shape[0] != attention_mask.shape[0]:
                    raise ValueError(
                        "cross_attention_mask batch size does not match input batch size: "
                        f"cross_batch={cross_attention_mask.shape[0]} vs input_batch={attention_mask.shape[0]}. "
                        "Likely leading batch axis leak from process_row/cached tokenized rows."
                    )
                if cross_attention_mask.shape[1] != prompt_attention_mask.shape[1]:
                    raise ValueError(
                        "cross_attention_mask prompt sequence length does not match prompt_input sequence length: "
                        f"cross_prompt={cross_attention_mask.shape[1]} vs prompt={prompt_attention_mask.shape[1]}. "
                        "Likely leading batch axis leak from process_row/cached tokenized rows."
                    )
                cross_attention_mask = self._expand_cross_attention_for_completion(
                    cross_attention_mask,
                    completion_attention_mask,
                )
                if cross_attention_mask.shape[1] != attention_mask.shape[1]:
                    raise ValueError(
                        "cross_attention_mask sequence length does not match prompt+completion sequence length: "
                        f"cross={cross_attention_mask.shape[1]} vs input={attention_mask.shape[1]}. "
                        "Likely leading batch axis leak from process_row/cached tokenized rows."
                    )

            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    pre_flush_attention = attention_mask
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                            attention_mask, input_ids, loss_mask, token_type_ids
                        )
                    else:
                        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                    if cross_attention_mask is not None:
                        cross_attention_mask = self._flush_left_nd(pre_flush_attention, cross_attention_mask)
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                    if cross_attention_mask is not None:
                        cross_attention_mask = cross_attention_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    if "token_type_ids" in concatenated_batch:
                        pre_flush_attention = attention_mask
                        attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                            attention_mask, input_ids, loss_mask, token_type_ids
                        )
                        if cross_attention_mask is not None:
                            cross_attention_mask = self._flush_left_nd(pre_flush_attention, cross_attention_mask)
                        token_type_ids = token_type_ids[:, -self.max_length :]
                    else:
                        pre_flush_attention = attention_mask
                        attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                        if cross_attention_mask is not None:
                            cross_attention_mask = self._flush_right_nd(pre_flush_attention, cross_attention_mask)
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    if cross_attention_mask is not None:
                        cross_attention_mask = cross_attention_mask[:, -self.max_length :]
                    if "token_type_ids" in concatenated_batch:
                        pre_flush_attention = attention_mask
                        attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                            attention_mask, input_ids, loss_mask, token_type_ids
                        )
                        if cross_attention_mask is not None:
                            cross_attention_mask = self._flush_left_nd(pre_flush_attention, cross_attention_mask)
                    else:
                        pre_flush_attention = attention_mask
                        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                        if cross_attention_mask is not None:
                            cross_attention_mask = self._flush_left_nd(pre_flush_attention, cross_attention_mask)
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )
            else:
                pre_flush_attention = attention_mask
                if "token_type_ids" in concatenated_batch:
                    attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                        attention_mask, input_ids, loss_mask, token_type_ids
                    )
                else:
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                if cross_attention_mask is not None:
                    cross_attention_mask = self._flush_left_nd(pre_flush_attention, cross_attention_mask)

            if "token_type_ids" in concatenated_batch:
                model_kwargs["token_type_ids"] = token_type_ids
            if cross_attention_mask is not None:
                model_kwargs["cross_attention_mask"] = cross_attention_mask

            if self.use_logits_to_keep:
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
                model_kwargs["logits_to_keep"] = logits_to_keep

            model_kwargs["output_hidden_states"] = True

            if self.padding_free:
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps[:, 1:].sum(-1)
        output = {}

        if self.use_weighting:
            with torch.no_grad():
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            chosen_logits = logits[:num_examples, :-1] if not self.is_encoder_decoder else logits[:num_examples]
            chosen_labels = labels[:num_examples, :-1] if not self.is_encoder_decoder else labels[:num_examples]
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if "ipo" in self.loss_type:
            all_logps = all_logps / loss_mask.sum(-1)

        if self.args.ld_alpha is not None and not is_ref_model:
            completion_lengths = loss_mask.sum(dim=1)
            chosen_lengths = completion_lengths[:num_examples]
            rejected_lengths = completion_lengths[num_examples:]
            public_lengths = torch.min(chosen_lengths, rejected_lengths)
            public_lengths = torch.cat([public_lengths, public_lengths], dim=0)

            seq_len = per_token_logps.size(1)
            position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)
            ld_mask = position_ids < public_lengths.unsqueeze(1)
            mask = position_ids < completion_lengths.unsqueeze(1)

            front_mask = (ld_mask & mask).float()
            rear_mask = (~ld_mask & mask).float()
            front_logps = (per_token_logps * front_mask).sum(dim=1)
            rear_logps = (per_token_logps * rear_mask).sum(dim=1)

            all_logps = front_logps + self.args.ld_alpha * rear_logps

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        if self.padding_free:
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output
