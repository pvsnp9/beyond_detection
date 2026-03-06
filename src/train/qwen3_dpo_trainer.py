from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from trl import DPOTrainer
from trl.trainer.utils import flush_left, flush_right, pad, pad_to_length, selective_log_softmax


def shift_tokens_right(input_ids: torch.Tensor, decoder_start_token_id: int) -> torch.Tensor:
    """Shift input ids one token to the right, and pad with pad_token_id."""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids


@dataclass
class Qwen3DataCollatorForPreference(DataCollatorMixin):
    """TRL preference collator with Qwen3-specific vision keys preserved."""

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
            # Qwen3-VL expects visual patches flattened across the whole batch.
            # Keep the patch axis (do not pad to a batch axis), then concatenate.
            pixel_values = []
            for example in examples:
                value = torch.tensor(example["pixel_values"])
                if value.dim() == 1:
                    value = value.unsqueeze(0)
                pixel_values.append(value)
            output["pixel_values"] = torch.cat(pixel_values, dim=0)
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor([example["image_sizes"] for example in examples])
        if "token_type_ids" in examples[0]:
            token_type_ids = [torch.tensor(example["token_type_ids"]) for example in examples]
            output["token_type_ids"] = pad(token_type_ids, padding_value=0, padding_side="left")

        # Qwen3-VL vision metadata keys.
        if "image_grid_thw" in examples[0]:
            image_grid_thw = []
            for example in examples:
                value = torch.tensor(example["image_grid_thw"])
                if value.dim() == 1:
                    value = value.unsqueeze(0)
                image_grid_thw.append(value)
            output["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
        if "pixel_values_videos" in examples[0]:
            pixel_values_videos = []
            for example in examples:
                value = torch.tensor(example["pixel_values_videos"])
                if value.dim() == 1:
                    value = value.unsqueeze(0)
                pixel_values_videos.append(value)
            output["pixel_values_videos"] = torch.cat(pixel_values_videos, dim=0)
        if "video_grid_thw" in examples[0]:
            video_grid_thw = []
            for example in examples:
                value = torch.tensor(example["video_grid_thw"])
                if value.dim() == 1:
                    value = value.unsqueeze(0)
                video_grid_thw.append(value)
            output["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)
        if "second_grid_ts" in examples[0]:
            second_grid_ts = [torch.tensor(example["second_grid_ts"]).reshape(-1) for example in examples]
            output["second_grid_ts"] = torch.cat(second_grid_ts, dim=0)

        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = torch.tensor([example["ref_chosen_logps"] for example in examples])
            output["ref_rejected_logps"] = torch.tensor([example["ref_rejected_logps"] for example in examples])

        return output


class Qwen3DPOTrainer(DPOTrainer):
    """DPOTrainer variant that preserves and forwards Qwen3-VL vision metadata."""

    @staticmethod
    def _first_item(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
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
    def process_row(
        features: dict[str, Any],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int] | list[float] | int]:
        """
        Vision-aware row processing that also preserves Qwen3-specific metadata like `image_grid_thw`.
        """
        processor, tokenizer = processing_class, processing_class.tokenizer
        processed_features = processor(images=features["images"], text=features["prompt"], add_special_tokens=False)

        prompt_input_ids = Qwen3DPOTrainer._to_python(Qwen3DPOTrainer._first_item(processed_features["input_ids"]))
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        if max_prompt_length is not None:
            # Keep the prompt prefix so multimodal placeholder tokens stay aligned
            # with image features for Qwen3-VL.
            prompt_input_ids = prompt_input_ids[:max_prompt_length]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        output: dict[str, Any] = {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        vision_sequence_keys = {
            "pixel_values",
            "pixel_attention_mask",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_grid_ts",
        }
        for key in (
            "pixel_values",
            "pixel_attention_mask",
            "image_sizes",
            "token_type_ids",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_grid_ts",
        ):
            if key in processed_features:
                value = processed_features[key]
                if key in vision_sequence_keys:
                    output[key] = Qwen3DPOTrainer._to_python(value)
                else:
                    output[key] = Qwen3DPOTrainer._to_python(Qwen3DPOTrainer._first_item(value))

        return output

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        extra_columns = [
            "pixel_values",
            "pixel_attention_mask",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_grid_ts",
        ]
        for column in extra_columns:
            if column not in self._signature_columns:
                self._signature_columns.append(column)

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, list | torch.LongTensor], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        output = DPOTrainer.concatenated_inputs(batch, padding_value)
        for key in ("image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_grid_ts"):
            if key in batch:
                output[key] = torch.cat([batch[key], batch[key]], dim=0)
        return output

    @staticmethod
    def _qwen_vision_model_kwargs(concatenated_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        vision_keys = (
            "pixel_values",
            "pixel_attention_mask",
            "image_sizes",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "second_grid_ts",
        )
        return {key: concatenated_batch[key] for key in vision_keys if key in concatenated_batch}

    def _compute_loss_liger(
        self, model: nn.Module, batch: dict[str, list | torch.LongTensor]
    ) -> dict[str, torch.Tensor]:
        unwrapped_model = self.accelerator.unwrap_model(model)
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
        model_kwargs.update(self._qwen_vision_model_kwargs(concatenated_batch))

        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        if self.is_encoder_decoder:
            encoder_outputs = unwrapped_model.get_encoder()(
                concatenated_batch["prompt_input_ids"],
                attention_mask=concatenated_batch["prompt_attention_mask"],
                return_dict=True,
            )
            decoder_input_ids = shift_tokens_right(
                concatenated_batch["completion_input_ids"],
                unwrapped_model.config.decoder_start_token_id,
            )
            decoder_outputs = unwrapped_model.get_decoder()(
                input_ids=decoder_input_ids,
                attention_mask=concatenated_batch["completion_attention_mask"],
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=concatenated_batch["prompt_attention_mask"],
                use_cache=False,
            )
            hidden_states = decoder_outputs.last_hidden_state

            ref_hidden_states = None
            if not self.reference_free and self.ref_model is not None:
                unwrapped_ref_model = self.accelerator.unwrap_model(self.ref_model)
                ref_encoder_outputs = unwrapped_ref_model.get_encoder()(
                    concatenated_batch["prompt_input_ids"],
                    attention_mask=concatenated_batch["prompt_attention_mask"],
                    return_dict=True,
                )
                ref_decoder_outputs = unwrapped_ref_model.get_decoder()(
                    input_ids=decoder_input_ids,
                    attention_mask=concatenated_batch["completion_attention_mask"],
                    encoder_hidden_states=ref_encoder_outputs.last_hidden_state,
                    encoder_attention_mask=concatenated_batch["prompt_attention_mask"],
                    use_cache=False,
                )
                ref_hidden_states = ref_decoder_outputs.last_hidden_state
            elif not self.reference_free:
                with self.null_ref_context():
                    ref_encoder_outputs = unwrapped_model.get_encoder()(
                        concatenated_batch["prompt_input_ids"],
                        attention_mask=concatenated_batch["prompt_attention_mask"],
                        return_dict=True,
                    )
                    ref_decoder_outputs = unwrapped_model.get_decoder()(
                        input_ids=decoder_input_ids,
                        attention_mask=concatenated_batch["completion_attention_mask"],
                        encoder_hidden_states=ref_encoder_outputs.last_hidden_state,
                        encoder_attention_mask=concatenated_batch["prompt_attention_mask"],
                        use_cache=False,
                    )
                    ref_hidden_states = ref_decoder_outputs.last_hidden_state

            labels = concatenated_batch["completion_input_ids"]
            loss_mask = completion_attention_mask.bool()
        else:
            input_ids = torch.cat(
                (concatenated_batch["prompt_input_ids"], concatenated_batch["completion_input_ids"]), dim=1
            )
            attention_mask = torch.cat(
                (concatenated_batch["prompt_attention_mask"], concatenated_batch["completion_attention_mask"]),
                dim=1,
            )
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )
            else:
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

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

            if hasattr(unwrapped_model, "get_decoder") and unwrapped_model.get_decoder() is not None:
                base_model = unwrapped_model.get_decoder()
            else:
                base_attr = getattr(unwrapped_model, "base_model_prefix", self.args.base_model_attribute_name)
                base_model = getattr(unwrapped_model, base_attr, unwrapped_model)

            outputs = base_model(
                input_ids,
                use_cache=False,
                **model_kwargs,
            )
            hidden_states = outputs.last_hidden_state[:, :-1]

            ref_hidden_states = None
            if not self.reference_free and self.ref_model is not None:
                unwrapped_ref_model = self.accelerator.unwrap_model(self.ref_model)
                if hasattr(unwrapped_ref_model, "get_decoder") and unwrapped_ref_model.get_decoder() is not None:
                    ref_base_model = unwrapped_ref_model.get_decoder()
                else:
                    ref_attr = getattr(unwrapped_ref_model, "base_model_prefix", self.args.base_model_attribute_name)
                    ref_base_model = getattr(unwrapped_ref_model, ref_attr, unwrapped_ref_model)

                ref_outputs = ref_base_model(
                    input_ids,
                    use_cache=False,
                    **model_kwargs,
                )
                ref_hidden_states = ref_outputs.last_hidden_state[:, :-1]
            elif not self.reference_free:
                if hasattr(unwrapped_model, "get_decoder") and unwrapped_model.get_decoder() is not None:
                    ref_base_model = unwrapped_model.get_decoder()
                else:
                    ref_attr = getattr(unwrapped_model, "base_model_prefix", self.args.base_model_attribute_name)
                    ref_base_model = getattr(unwrapped_model, ref_attr, unwrapped_model)
                with self.null_ref_context():
                    ref_outputs = ref_base_model(
                        input_ids,
                        use_cache=False,
                        **model_kwargs,
                    )
                    ref_hidden_states = ref_outputs.last_hidden_state[:, :-1]

            masked_input_ids = torch.where(loss_mask != 0, input_ids, self.label_pad_token_id)
            labels = masked_input_ids[:, 1:]

        lm_head = unwrapped_model.get_output_embeddings()

        ref_weight = None
        ref_bias = None
        if not self.reference_free:
            if self.ref_model is not None:
                unwrapped_ref_model = self.accelerator.unwrap_model(self.ref_model)
                ref_lm_head = unwrapped_ref_model.get_output_embeddings()
            else:
                with self.null_ref_context():
                    ref_lm_head = unwrapped_model.get_output_embeddings()
            ref_weight = ref_lm_head.weight
            ref_bias = ref_lm_head.bias if hasattr(ref_lm_head, "bias") else None

        loss_output = self.dpo_loss_fn(
            lm_head.weight,
            hidden_states,
            labels,
            bias=lm_head.bias if hasattr(lm_head, "bias") else None,
            ref_input=ref_hidden_states if not self.reference_free else None,
            ref_weight=ref_weight if not self.reference_free else None,
            ref_bias=ref_bias if not self.reference_free else None,
        )
        (
            loss,
            (chosen_logps, rejected_logps, chosen_logits_mean, rejected_logits_mean, nll_loss, *aux_outputs),
        ) = loss_output

        output = {
            "loss": loss,
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "mean_chosen_logits": chosen_logits_mean,
            "mean_rejected_logits": rejected_logits_mean,
            "nll_loss": nll_loss,
            "chosen_rewards": aux_outputs[0],
            "rejected_rewards": aux_outputs[1],
        }
        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, list | torch.LongTensor], is_ref_model: bool = False
    ) -> dict[str, torch.Tensor]:
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)

        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
        model_kwargs.update(self._qwen_vision_model_kwargs(concatenated_batch))

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
            expected_patches = None
            expected_image_tokens = None
            if "pixel_values" in model_kwargs and "image_grid_thw" in model_kwargs:
                expected_patches = int(model_kwargs["image_grid_thw"].to(torch.long).prod(-1).sum().item())
                actual_patches = int(model_kwargs["pixel_values"].shape[0])
                if actual_patches != expected_patches:
                    raise ValueError(
                        "Inconsistent vision batch: pixel_values patch rows "
                        f"({actual_patches}) != image_grid_thw-derived patches ({expected_patches}). "
                        "Likely stale cached tokenized rows; disable/clear HF datasets map cache and remap."
                    )
                vision_cfg = getattr(getattr(model, "config", None), "vision_config", None)
                spatial_merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2))
                merge_factor = spatial_merge_size * spatial_merge_size
                expected_image_tokens = int(
                    (model_kwargs["image_grid_thw"].to(torch.long).prod(-1) // merge_factor).sum().item()
                )
            if "token_type_ids" in concatenated_batch:
                prompt_token_type_ids = concatenated_batch["token_type_ids"]
                token_type_ids = pad_to_length(prompt_token_type_ids, input_ids.shape[1], 0)
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                            attention_mask, input_ids, loss_mask, token_type_ids
                        )
                    else:
                        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                            attention_mask, input_ids, loss_mask, token_type_ids
                        )
                        token_type_ids = token_type_ids[:, -self.max_length :]
                    else:
                        attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                            attention_mask, input_ids, loss_mask, token_type_ids
                        )
                    else:
                        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )
            else:
                if "token_type_ids" in concatenated_batch:
                    attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                        attention_mask, input_ids, loss_mask, token_type_ids
                    )
                else:
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

            if "token_type_ids" in concatenated_batch:
                model_kwargs["token_type_ids"] = token_type_ids

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

            if expected_image_tokens is not None:
                image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
                if image_token_id is not None:
                    image_token_count = int((input_ids == image_token_id).sum().item())
                    if image_token_count != expected_image_tokens:
                        raise ValueError(
                            "Image placeholder tokens do not match image features after sequence truncation: "
                            f"tokens={image_token_count}, features={expected_image_tokens}. "
                            "Use truncation_mode='keep_start' for multimodal DPO (or increase max_length)."
                        )

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
