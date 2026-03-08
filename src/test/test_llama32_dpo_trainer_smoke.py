from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from src.train.llama32_dpo_trainer import Llama32DataCollatorForPreference, Llama32DPOTrainer


class _FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        ids = [3 + (ord(ch) % 20) for ch in text][:32]
        if not ids:
            ids = [7]
        return {"input_ids": ids}


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def __call__(self, images: Any, text: str, add_special_tokens: bool = False) -> dict[str, Any]:
        tok = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        seq_len = len(tok)
        return {
            "input_ids": [tok],
            "pixel_values": [[[1.0, 2.0], [3.0, 4.0]]],
            "aspect_ratio_ids": [[1, 2]],
            "aspect_ratio_mask": [[[1, 1], [1, 0]]],
            "cross_attention_mask": [[[[1, 0]]] * seq_len],
        }


class _FakeModel(nn.Module):
    def __init__(self, vocab_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.last_kwargs = {}
        self.last_input_ids = None

    def forward(self, input_ids, **kwargs):
        self.last_input_ids = input_ids
        self.last_kwargs = kwargs
        bs, seq = input_ids.shape
        logits = torch.zeros(bs, seq, self.vocab_size, dtype=torch.float32, device=input_ids.device)
        logits.scatter_(2, (input_ids % self.vocab_size).unsqueeze(-1), 5.0)
        return SimpleNamespace(logits=logits)


def _make_trainer_for_smoke() -> Llama32DPOTrainer:
    trainer = Llama32DPOTrainer.__new__(Llama32DPOTrainer)
    trainer.pad_token_id = 0
    trainer.aux_loss_enabled = False
    trainer.is_encoder_decoder = False
    trainer.max_length = None
    trainer.truncation_mode = "keep_start"
    trainer.use_logits_to_keep = False
    trainer.padding_free = False
    trainer.use_weighting = False
    trainer.loss_type = ["sigmoid"]
    trainer.args = SimpleNamespace(rpo_alpha=None, ld_alpha=None)
    return trainer


def main() -> None:
    processor = _FakeProcessor()

    row_1 = Llama32DPOTrainer.process_row(
        {
            "images": ["img-a"],
            "prompt": "Why is this sarcastic?",
            "chosen": "The caption and image conflict.",
            "rejected": "It is literal.",
        },
        processor,
        add_special_tokens=False,
    )
    row_2 = Llama32DPOTrainer.process_row(
        {
            "images": ["img-b"],
            "prompt": "Explain the irony.",
            "chosen": "The text says the opposite of the scene.",
            "rejected": "No irony appears.",
        },
        processor,
        add_special_tokens=False,
    )
    assert "pixel_values" in row_1
    assert "aspect_ratio_ids" in row_1
    assert "cross_attention_mask" in row_1

    collator = Llama32DataCollatorForPreference(pad_token_id=processor.tokenizer.pad_token_id)
    batch = collator([row_1, row_2])
    assert batch["prompt_input_ids"].shape[0] == 2
    assert batch["cross_attention_mask"].shape[0] == 2
    assert batch["cross_attention_mask"].shape[1] == batch["prompt_input_ids"].shape[1]

    trainer = _make_trainer_for_smoke()
    concatenated_batch = trainer.concatenated_inputs(batch, padding_value=trainer.pad_token_id)
    for key in ("pixel_values", "aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask"):
        assert concatenated_batch[key].shape[0] == 4
        assert torch.equal(concatenated_batch[key][:2], concatenated_batch[key][2:])

    model = _FakeModel()
    output = trainer.concatenated_forward(model, batch)

    assert "chosen_logps" in output and "rejected_logps" in output
    assert model.last_kwargs.get("pixel_values") is not None
    assert model.last_kwargs.get("aspect_ratio_ids") is not None
    assert model.last_kwargs.get("aspect_ratio_mask") is not None
    assert model.last_kwargs.get("cross_attention_mask") is not None
    assert model.last_kwargs["cross_attention_mask"].shape[0] == 4
    assert model.last_kwargs["cross_attention_mask"].shape[1] == model.last_input_ids.shape[1]
    print("Llama32 DPO trainer smoke test passed.")


if __name__ == "__main__":
    main()
