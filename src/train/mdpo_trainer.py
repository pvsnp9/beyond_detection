"""Unified multimodal DPO trainer implementing the true mDPO objective
(Wang et al. 2024, arXiv:2406.11839):

    L = L_DPO + conditional_weight * L_CoPO + anchor_weight * L_Anchor

- L_DPO:   standard text-preference DPO (chosen vs rejected, real image).
- L_CoPO:  conditional image preference — the chosen response scored with the
           real image must beat the same response scored with a corrupted
           (random-cropped) image. Applied to modality=="both" rows only.
- L_Anchor: keeps the chosen reward positive so both responses aren't merely
           pushed down together.

Replaces the three per-model trainers (qwen3/gemma3/llama32_dpo_trainer.py);
model-specific handling lives in src/train/vision_strategies.py.
"""
from __future__ import annotations

import json
import random
from contextlib import nullcontext
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Sampler
from transformers.data.data_collator import DataCollatorMixin
from trl import DPOTrainer
from trl.trainer.utils import flush_left, pad, pad_to_length, selective_log_softmax

from config.logistics import MDPOParams
from src.train.vision_strategies import VisionStrategy
from src.utils.image_utils import random_crop_fraction


def _first_item(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0]
    if torch.is_tensor(value):
        return value[0]
    if hasattr(value, "ndim") and getattr(value, "ndim", 0) > 0:
        return value[0]
    return value


def _to_python(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _flush_left_nd(mask: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """flush_left for tensors with rank > 2 whose dim 1 tracks `mask`'s dim 1."""
    if tensor.dim() < 2 or tensor.shape[:2] != mask.shape:
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


class ModalityGroupedBatchSampler(Sampler):
    """Yields batches that never mix with-image and text-only rows, so text
    batches carry no vision keys and every model takes a clean text forward."""

    def __init__(self, has_image: List[bool], batch_size: int, seed: int = 0) -> None:
        self.groups = [
            [i for i, h in enumerate(has_image) if h],
            [i for i, h in enumerate(has_image) if not h],
        ]
        self.batch_size = max(1, int(batch_size))
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        batches: List[List[int]] = []
        for group in self.groups:
            indices = group[:]
            rng.shuffle(indices)
            batches.extend(
                indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)
            )
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum(
            (len(group) + self.batch_size - 1) // self.batch_size for group in self.groups if group
        )


class MDPOPreferenceCollator(DataCollatorMixin):
    """Preference collator shared by all models.

    Pads prompt (left) / chosen / rejected (right), stacks the strategy's
    stored vision metadata, recomputes real pixel features from the raw PIL
    images, and builds mDPO corrupted-image features for modality=="both" rows
    with fresh random crops every step.
    """

    return_tensors = "pt"

    def __init__(
        self,
        processor: Any,
        strategy: VisionStrategy,
        mdpo_params: MDPOParams,
        pad_token_id: int,
    ) -> None:
        self.processor = processor
        self.strategy = strategy
        self.params = mdpo_params
        self.pad_token_id = pad_token_id

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]

        output: Dict[str, Any] = {
            "prompt_input_ids": pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"),
            "prompt_attention_mask": pad(prompt_attention_mask, padding_value=0, padding_side="left"),
            "chosen_input_ids": pad(chosen_input_ids, padding_value=self.pad_token_id),
            "chosen_attention_mask": pad(chosen_attention_mask, padding_value=0),
            "rejected_input_ids": pad(rejected_input_ids, padding_value=self.pad_token_id),
            "rejected_attention_mask": pad(rejected_attention_mask, padding_value=0),
        }

        modalities = [str(example.get("modality", "both")).strip().lower() for example in examples]
        with_image = [len(example.get("images") or []) > 0 for example in examples]
        if any(with_image) != all(with_image):
            raise ValueError(
                "mixed text-only and with-image rows in one batch; "
                "use ModalityGroupedBatchSampler for the train dataloader"
            )
        output["mm_mask"] = torch.tensor(
            [m == "both" for m in modalities], dtype=torch.bool
        )

        if all(with_image):
            vision_meta = [json.loads(example.get("vision_meta") or "{}") for example in examples]
            self.strategy.collate_metadata(vision_meta, output)

            pils = [example["images"][0] for example in examples]
            real_feats = self.strategy.process_images(self.processor, pils)
            all_rows = torch.arange(len(examples))
            if not self.strategy.features_match(output, all_rows, real_feats):
                raise ValueError(
                    "recomputed vision features do not match stored row metadata; "
                    "stale tokenized rows — clear the HF datasets map cache and remap"
                )
            for key, value in real_feats.items():
                if key not in output:
                    output[key] = value

            if self.params.use_conditional and bool(output["mm_mask"].any()):
                rng = random.Random()  # urandom-seeded per call: fresh, worker-safe crops
                mm_index = output["mm_mask"].nonzero(as_tuple=True)[0]
                corrupt_pils = [
                    random_crop_fraction(
                        pils[i], self.params.crop_frac_min, self.params.crop_frac_max, rng
                    )
                    for i in mm_index.tolist()
                ]
                corrupt_feats = self.strategy.process_images(self.processor, corrupt_pils)
                if self.strategy.features_match(output, mm_index, corrupt_feats):
                    for key, value in corrupt_feats.items():
                        output[f"corrupt_{key}"] = value
                else:
                    # crop+resize should be dimension-preserving; if not, skip the
                    # conditional term for this batch rather than crash
                    print("[MDPO] corrupted-image features mismatch; conditional term skipped for batch")

        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = torch.tensor([ex["ref_chosen_logps"] for ex in examples])
            output["ref_rejected_logps"] = torch.tensor([ex["ref_rejected_logps"] for ex in examples])

        return output


class MDPOTrainer(DPOTrainer):
    """DPOTrainer with per-model vision strategies and the mDPO objective."""

    def __init__(
        self,
        model: Any = None,
        ref_model: Any = None,
        args: Any = None,
        strategy: Optional[VisionStrategy] = None,
        mdpo_params: Optional[MDPOParams] = None,
        **kwargs: Any,
    ) -> None:
        if strategy is None:
            raise ValueError("MDPOTrainer requires a VisionStrategy")
        # set before super().__init__: dataset tokenization uses self.process_row
        self.strategy = strategy
        self.mdpo_params = mdpo_params or MDPOParams()
        super().__init__(model=model, ref_model=ref_model, args=args, **kwargs)
        if self.is_encoder_decoder or self.padding_free or self.use_logits_to_keep:
            raise NotImplementedError("MDPOTrainer supports decoder-only, non-padding-free models")
        if self.truncation_mode != "keep_start":
            raise ValueError("mDPO requires truncation_mode='keep_start' to keep image placeholders")

    # ---------------------------------------------------------------- rows
    def process_row(
        self,
        features: Dict[str, Any],
        processing_class: Any,
        max_prompt_length: Optional[int] = None,
        max_completion_length: Optional[int] = None,
        add_special_tokens: bool = False,
    ) -> Dict[str, Any]:
        if add_special_tokens:
            raise ValueError("prompts are already chat-templated; add_special_tokens must be False")
        processor = processing_class
        tokenizer = processor.tokenizer
        images = features.get("images") or []

        # vision metadata is stored as one JSON-string column: rows must expose a
        # consistent schema to the Arrow writer, and text rows have no vision keys
        vision_meta: Dict[str, Any] = {}
        if images:
            processed = processor(images=images, text=features["prompt"], add_special_tokens=False)
            prompt_input_ids = _to_python(_first_item(processed["input_ids"]))
            for key in self.strategy.row_keys:
                if key in processed:
                    value = processed[key]
                    if key in self.strategy.row_flat_keys:
                        vision_meta[key] = _to_python(value)
                    else:
                        vision_meta[key] = _to_python(_first_item(value))
        else:
            prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]

        eos_id = self.strategy.completion_eos_id(tokenizer)
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"] + [eos_id]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"] + [eos_id]

        if max_prompt_length is not None:
            # keep_start so image placeholder tokens stay aligned with vision features
            prompt_input_ids = prompt_input_ids[:max_prompt_length]
            for key in ("token_type_ids", "cross_attention_mask"):
                if key in vision_meta:
                    vision_meta[key] = vision_meta[key][:max_prompt_length]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "vision_meta": json.dumps(vision_meta),
        }

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        extra_columns = ["images", "modality", "mm_mask", "vision_meta"]
        for column in extra_columns:
            if column not in self._signature_columns:
                self._signature_columns.append(column)

    # ------------------------------------------------------------ batching
    def get_train_dataloader(self) -> DataLoader:
        modalities = self.train_dataset["modality"]
        has_image = [str(m).strip().lower() != "text" for m in modalities]
        batch_sampler = ModalityGroupedBatchSampler(
            has_image, self.args.per_device_train_batch_size, seed=self.args.seed
        )
        dataloader_params: Dict[str, Any] = {
            "batch_sampler": batch_sampler,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if self.args.dataloader_num_workers > 0 and self.args.dataloader_prefetch_factor:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

    def concatenated_inputs(self, batch: Dict[str, Any], padding_value: int) -> Dict[str, Any]:
        output = DPOTrainer.concatenated_inputs(batch, padding_value)
        for key in self.strategy.extra_concat_keys:
            if batch.get(key) is not None:
                output[key] = torch.cat([batch[key], batch[key]], dim=0)
        return output

    # ------------------------------------------------------------- forward
    def _forward_logps(
        self,
        model: nn.Module,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        completion_input_ids: torch.Tensor,
        completion_attention_mask: torch.Tensor,
        vision_kwargs: Dict[str, Any],
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Score completions given prompts: the shared forward core."""
        model_kwargs: Dict[str, Any] = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
        model_kwargs.update(vision_kwargs)
        # Mllama: extend prompt-level cross-attention rows over the completion
        self.strategy.expand_cross_attention(
            model_kwargs, prompt_attention_mask, completion_attention_mask
        )

        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1
        )

        cross_attention_mask = model_kwargs.pop("cross_attention_mask", None)
        pre_flush_attention = attention_mask
        if token_type_ids is not None:
            token_type_ids = pad_to_length(token_type_ids, input_ids.shape[1], 0)
            attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                attention_mask, input_ids, loss_mask, token_type_ids
            )
        else:
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
        if cross_attention_mask is not None:
            cross_attention_mask = _flush_left_nd(pre_flush_attention, cross_attention_mask)
            cross_attention_mask = cross_attention_mask[:, : input_ids.shape[1]]

        if self.max_length is not None and input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            loss_mask = loss_mask[:, : self.max_length]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, : self.max_length]
            if cross_attention_mask is not None:
                cross_attention_mask = cross_attention_mask[:, : self.max_length]

        model_kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids
        if cross_attention_mask is not None:
            model_kwargs["cross_attention_mask"] = cross_attention_mask

        self.strategy.validate_batch(model, model_kwargs, input_ids)

        outputs = model(input_ids, **model_kwargs)
        logits = outputs.logits

        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()
        if logits.shape[:2] != labels.shape[:2]:
            logits = logits[:, -labels.shape[1] :]

        labels = labels.clone()
        labels[~loss_mask] = 0
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)
        seq_logps = per_token_logps[:, 1:].sum(-1)
        if any("ipo" in lt for lt in self.loss_type):
            seq_logps = seq_logps / loss_mask.sum(-1)

        return {
            "seq_logps": seq_logps,
            "logits": logits,
            "labels": labels,
            "loss_mask": loss_mask,
            "aux_loss": getattr(outputs, "aux_loss", None),
        }

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Any], is_ref_model: bool = False
    ) -> Dict[str, torch.Tensor]:
        num_examples = batch["prompt_input_ids"].shape[0]
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)
        vision_kwargs = self.strategy.model_kwargs(concatenated_batch)
        token_type_ids = concatenated_batch.get("token_type_ids")

        fwd = self._forward_logps(
            model,
            concatenated_batch["prompt_input_ids"],
            concatenated_batch["prompt_attention_mask"],
            concatenated_batch["completion_input_ids"],
            concatenated_batch["completion_attention_mask"],
            vision_kwargs,
            token_type_ids,
        )
        seq_logps, logits, loss_mask = fwd["seq_logps"], fwd["logits"], fwd["loss_mask"]

        output: Dict[str, torch.Tensor] = {
            "chosen_logps": seq_logps[:num_examples],
            "rejected_logps": seq_logps[num_examples:],
            "mean_chosen_logits": logits[:num_examples][loss_mask[:num_examples]].mean(),
            "mean_rejected_logits": logits[num_examples:][loss_mask[num_examples:]].mean(),
        }
        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            chosen_logits = logits[:num_examples, :-1]
            chosen_labels = fwd["labels"][:num_examples, :-1]
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1),
                torch.flatten(chosen_labels, end_dim=1),
                ignore_index=0,
            )
        if self.aux_loss_enabled:
            output["aux_loss"] = fwd["aux_loss"]
        return output

    def _corrupt_chosen_logps(self, model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
        """Log-probs of the chosen response conditioned on the corrupted image
        (modality=="both" rows only)."""
        mm_index = batch["mm_mask"].bool().nonzero(as_tuple=True)[0]
        vision_kwargs: Dict[str, Any] = {}
        for key in ("pixel_values", "image_grid_thw", "aspect_ratio_ids", "aspect_ratio_mask"):
            corrupt_key = f"corrupt_{key}"
            if batch.get(corrupt_key) is not None:
                vision_kwargs[key] = batch[corrupt_key]
        # positional metadata is dimension-identical by construction: reuse it
        if batch.get("cross_attention_mask") is not None:
            vision_kwargs["cross_attention_mask"] = batch["cross_attention_mask"][mm_index]
        token_type_ids = None
        if batch.get("token_type_ids") is not None:
            token_type_ids = batch["token_type_ids"][mm_index]

        fwd = self._forward_logps(
            model,
            batch["prompt_input_ids"][mm_index],
            batch["prompt_attention_mask"][mm_index],
            batch["chosen_input_ids"][mm_index],
            batch["chosen_attention_mask"][mm_index],
            vision_kwargs,
            token_type_ids,
        )
        return fwd["seq_logps"]

    # ---------------------------------------------------------------- loss
    def get_batch_loss_metrics(
        self,
        model: Any,
        batch: Dict[str, Any],
        train_eval: Literal["train", "eval"] = "train",
    ):
        if self.args.use_liger_kernel:
            raise NotImplementedError("MDPOTrainer does not support the liger kernel path")
        metrics: Dict[str, float] = {}

        model_output = self.concatenated_forward(model, batch)

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        losses = 0
        chosen_rewards = 0
        rejected_rewards = 0
        for idx, loss_type in enumerate(self.loss_type):
            _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
                model_output["chosen_logps"],
                model_output["rejected_logps"],
                ref_chosen_logps,
                ref_rejected_logps,
                loss_type,
                model_output,
            )
            weight = self.loss_weights[idx] if self.loss_weights else 1.0
            losses = losses + _losses * weight
            chosen_rewards = chosen_rewards + _chosen_rewards * weight
            rejected_rewards = rejected_rewards + _rejected_rewards * weight

        params = self.mdpo_params
        chosen_reward_margin = self.beta * (model_output["chosen_logps"] - ref_chosen_logps)
        mm_mask = batch.get("mm_mask")

        copo_losses = None
        image_margin = None
        if (
            params.use_conditional
            and mm_mask is not None
            and bool(mm_mask.any())
            and batch.get("corrupt_pixel_values") is not None
        ):
            policy_corrupt_logps = self._corrupt_chosen_logps(model, batch)
            ref_ctx = (
                autocast(self.accelerator.device.type)
                if self._peft_has_been_casted_to_bf16
                else nullcontext()
            )
            with torch.no_grad(), ref_ctx:
                if self.ref_model is None:
                    with self.null_ref_context():
                        ref_corrupt_logps = self._corrupt_chosen_logps(self.model, batch)
                else:
                    ref_corrupt_logps = self._corrupt_chosen_logps(self.ref_model, batch)
            mm = mm_mask.bool()
            image_margin = self.beta * (
                (model_output["chosen_logps"] - ref_chosen_logps)[mm]
                - (policy_corrupt_logps - ref_corrupt_logps)
            )
            copo_losses = -F.logsigmoid(image_margin)
            copo_full = torch.zeros_like(losses)
            copo_full[mm] = copo_losses
            losses = losses + params.conditional_weight * copo_full

        anchor_losses = None
        if params.use_anchor:
            anchor_losses = -F.logsigmoid(chosen_reward_margin - params.anchor_delta)
            losses = losses + params.anchor_weight * anchor_losses

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]
        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        gather = self.accelerator.gather_for_metrics
        metrics[f"{prefix}rewards/chosen"] = gather(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = gather(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = gather(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = gather(chosen_rewards - rejected_rewards).mean().item()
        metrics[f"{prefix}logps/chosen"] = gather(model_output["chosen_logps"]).detach().mean().item()
        metrics[f"{prefix}logps/rejected"] = gather(model_output["rejected_logps"]).detach().mean().item()
        metrics[f"{prefix}logits/chosen"] = gather(model_output["mean_chosen_logits"]).detach().mean().item()
        metrics[f"{prefix}logits/rejected"] = gather(model_output["mean_rejected_logits"]).detach().mean().item()
        if mm_mask is not None:
            metrics[f"{prefix}mdpo/mm_fraction"] = gather(mm_mask.float()).mean().item()
        if copo_losses is not None:
            metrics[f"{prefix}mdpo/copo_loss"] = gather(copo_losses).detach().mean().item()
            metrics[f"{prefix}mdpo/image_margin"] = gather(image_margin).detach().mean().item()
        if anchor_losses is not None:
            metrics[f"{prefix}mdpo/anchor_loss"] = gather(anchor_losses).detach().mean().item()
        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            metrics[f"{prefix}nll_loss"] = gather(model_output["nll_loss"]).detach().mean().item()
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = gather(model_output["aux_loss"]).detach().mean().item()

        return losses.mean(), metrics
