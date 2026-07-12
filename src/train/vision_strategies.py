"""Per-model vision handling for the unified mDPO trainer.

Each strategy owns the model-specific pieces the three legacy DPO trainers
duplicated: which small vision-metadata keys a tokenized row keeps, how they
are collated, how chosen/rejected duplication treats them, which kwargs reach
the model forward, and model quirks (Mllama cross-attention expansion, Gemma3
turn-end token, Qwen3-VL patch/token guards).

Heavy pixel tensors are NOT stored in the mapped dataset (the legacy trainers
did, costing ~10-15MB/row). Rows keep only the deterministic metadata; the
preference collator recomputes pixel features from the resize-capped PIL
images at collate time (it must run the image processor anyway to build the
mDPO corrupted images) and asserts the recomputed metadata matches the stored
row metadata.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from trl.trainer.utils import pad


class VisionStrategy:
    key: str = "base"
    # small metadata keys process_row preserves per row (row-level, no batch axis)
    row_keys: tuple = ()
    # subset of row_keys stored whole (others are stored via first-item unbatching)
    row_flat_keys: tuple = ()
    # keys duplicated by torch.cat([x, x], dim=0) in concatenated_inputs beyond
    # what base TRL already duplicates (pixel_values / pixel_attention_mask /
    # image_sizes / token_type_ids)
    extra_concat_keys: tuple = ()
    # keys forwarded to the model besides input_ids/attention_mask;
    # token_type_ids and cross_attention_mask are threaded separately
    model_kwarg_keys: tuple = ()

    def completion_eos_id(self, tokenizer) -> int:
        return tokenizer.eos_token_id

    def collate_metadata(self, examples: List[Dict[str, Any]], output: Dict[str, Any]) -> None:
        """Stack stored per-row vision metadata into batch tensors."""
        return None

    def process_images(self, processor: Any, pil_images: list) -> Dict[str, torch.Tensor]:
        """Run the image processor on PILs (real or corrupted); returns batch tensors."""
        raise NotImplementedError

    def features_match(self, batch: Dict[str, Any], row_index: torch.Tensor, feats: Dict[str, torch.Tensor]) -> bool:
        """Check freshly computed features are consistent with stored row metadata."""
        return True

    def model_kwargs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: batch[k] for k in self.model_kwarg_keys if batch.get(k) is not None}

    def expand_cross_attention(
        self,
        model_kwargs: Dict[str, Any],
        prompt_attention_mask: torch.Tensor,
        completion_attention_mask: torch.Tensor,
    ) -> None:
        """Extend prompt-level cross-attention masks over completion tokens (Mllama only)."""
        return None

    def validate_batch(self, model: Any, model_kwargs: Dict[str, Any], input_ids: torch.Tensor) -> None:
        return None


class QwenVisionStrategy(VisionStrategy):
    key = "qwen"
    row_keys = ("image_grid_thw",)
    row_flat_keys = ("image_grid_thw",)
    extra_concat_keys = ("image_grid_thw",)
    model_kwarg_keys = ("pixel_values", "image_grid_thw")

    def collate_metadata(self, examples, output) -> None:
        if examples[0].get("image_grid_thw") is None:
            return
        grids = []
        for example in examples:
            grid = torch.tensor(example["image_grid_thw"])
            if grid.dim() == 1:
                grid = grid.unsqueeze(0)
            grids.append(grid)
        # Qwen3-VL keeps one (t, h, w) row per image, flattened across the batch
        output["image_grid_thw"] = torch.cat(grids, dim=0)

    def process_images(self, processor, pil_images):
        feats = processor.image_processor(images=pil_images, return_tensors="pt")
        return {
            "pixel_values": feats["pixel_values"],
            "image_grid_thw": feats["image_grid_thw"],
        }

    def features_match(self, batch, row_index, feats) -> bool:
        stored = batch["image_grid_thw"][row_index]
        return bool(torch.equal(stored, feats["image_grid_thw"]))

    def validate_batch(self, model, model_kwargs, input_ids) -> None:
        if model_kwargs.get("pixel_values") is None or model_kwargs.get("image_grid_thw") is None:
            return
        grid = model_kwargs["image_grid_thw"].to(torch.long)
        expected_patches = int(grid.prod(-1).sum().item())
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
        expected_image_tokens = int((grid.prod(-1) // merge_factor).sum().item())
        image_token_id = getattr(getattr(model, "config", None), "image_token_id", None)
        if image_token_id is not None:
            image_token_count = int((input_ids == image_token_id).sum().item())
            if image_token_count != expected_image_tokens:
                raise ValueError(
                    "Image placeholder tokens do not match image features after truncation: "
                    f"tokens={image_token_count}, features={expected_image_tokens}. "
                    "Increase max_length/max_prompt_length for multimodal DPO."
                )


class GemmaVisionStrategy(VisionStrategy):
    key = "gemma"
    row_keys = ("token_type_ids",)
    row_flat_keys = ()
    extra_concat_keys = ()  # base TRL duplicates pixel_values and token_type_ids
    model_kwarg_keys = ("pixel_values",)

    def completion_eos_id(self, tokenizer) -> int:
        # Gemma3 assistant turns end with <end_of_turn> (id 106), not <eos> (id 1);
        # appending <eos> would train a terminator the SFT model never emits.
        eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot_id is None or eot_id < 0:
            raise ValueError("Gemma tokenizer has no <end_of_turn> token")
        return eot_id

    def collate_metadata(self, examples, output) -> None:
        if examples[0].get("token_type_ids") is None:
            return
        token_type_ids = [torch.tensor(example["token_type_ids"]) for example in examples]
        # align with the left-padded prompt
        output["token_type_ids"] = pad(token_type_ids, padding_value=0, padding_side="left")

    def process_images(self, processor, pil_images):
        feats = processor.image_processor(images=pil_images, return_tensors="pt")
        return {"pixel_values": feats["pixel_values"]}


class LlamaVisionStrategy(VisionStrategy):
    key = "llama"
    row_keys = ("aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask")
    row_flat_keys = ()
    extra_concat_keys = ("aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask")
    model_kwarg_keys = ("pixel_values", "aspect_ratio_ids", "aspect_ratio_mask")

    def collate_metadata(self, examples, output) -> None:
        if examples[0].get("aspect_ratio_ids") is not None:
            aspect_ratio_ids = [torch.tensor(example["aspect_ratio_ids"]) for example in examples]
            output["aspect_ratio_ids"] = pad(aspect_ratio_ids, padding_value=0)
        if examples[0].get("aspect_ratio_mask") is not None:
            aspect_ratio_mask = [torch.tensor(example["aspect_ratio_mask"]) for example in examples]
            output["aspect_ratio_mask"] = pad(aspect_ratio_mask, padding_value=0)
        if examples[0].get("cross_attention_mask") is not None:
            cross_attention_mask = [torch.tensor(example["cross_attention_mask"]) for example in examples]
            # sequence axis aligns with the left-padded prompt
            output["cross_attention_mask"] = pad(cross_attention_mask, padding_value=0, padding_side="left")

    def process_images(self, processor, pil_images):
        # Mllama image processor expects one images-list per sample
        feats = processor.image_processor(images=[[img] for img in pil_images], return_tensors="pt")
        return {
            "pixel_values": feats["pixel_values"],
            "aspect_ratio_ids": feats["aspect_ratio_ids"],
            "aspect_ratio_mask": feats["aspect_ratio_mask"],
        }

    def features_match(self, batch, row_index, feats) -> bool:
        # tiling layout must match so the stored cross_attention_mask stays valid
        return bool(torch.equal(batch["aspect_ratio_ids"][row_index], feats["aspect_ratio_ids"]))

    def model_kwargs(self, batch):
        kwargs = super().model_kwargs(batch)
        if batch.get("cross_attention_mask") is not None:
            kwargs["cross_attention_mask"] = batch["cross_attention_mask"]
        return kwargs

    def expand_cross_attention(self, model_kwargs, prompt_attention_mask, completion_attention_mask) -> None:
        cross_attention_mask = model_kwargs.get("cross_attention_mask")
        if cross_attention_mask is None:
            return
        if cross_attention_mask.dim() != 4:
            raise ValueError(
                "cross_attention_mask must have shape (batch, seq_len, max_num_images, max_num_tiles); "
                f"got {tuple(cross_attention_mask.shape)}"
            )
        if cross_attention_mask.shape[1] != prompt_attention_mask.shape[1]:
            raise ValueError(
                "cross_attention_mask prompt length mismatch: "
                f"{cross_attention_mask.shape[1]} vs {prompt_attention_mask.shape[1]}"
            )
        completion_len = completion_attention_mask.shape[1]
        if completion_len <= 0:
            return
        # Completion tokens must keep attending to the image: repeat the LAST
        # prompt row, exactly like Mllama generation extends the mask per new
        # token. Prompts are left-padded, so index -1 is the final real prompt
        # token. Zero rows here would mask completions out of image
        # cross-attention entirely (the pre-revision bug).
        last_row = cross_attention_mask[:, -1:, :, :]
        completion_rows = last_row.expand(-1, completion_len, -1, -1)
        model_kwargs["cross_attention_mask"] = torch.cat(
            [cross_attention_mask, completion_rows], dim=1
        )


VISION_STRATEGIES = {
    "qwen": QwenVisionStrategy,
    "gemma": GemmaVisionStrategy,
    "llama": LlamaVisionStrategy,
}
