from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from src.train.qwen3_dpo_trainer import Qwen3DataCollatorForPreference, Qwen3DPOTrainer


class _FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        ids = [3 + (ord(ch) % 40) for ch in text][:32]
        if not ids:
            ids = [7]
        return {"input_ids": ids}


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def __call__(self, images: Any, text: str, add_special_tokens: bool = False) -> dict[str, Any]:
        tok = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return {
            "input_ids": [tok],
            "pixel_values": [[1.0, 2.0], [3.0, 4.0]],
            "image_grid_thw": [[1, 1, 2]],
        }


class _FakeModel(nn.Module):
    def __init__(self, vocab_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.last_image_grid_thw = None

    def forward(self, input_ids, **kwargs):
        self.last_image_grid_thw = kwargs.get("image_grid_thw")
        bs, seq = input_ids.shape
        logits = torch.zeros(bs, seq, self.vocab_size, dtype=torch.float32, device=input_ids.device)
        logits.scatter_(2, (input_ids % self.vocab_size).unsqueeze(-1), 5.0)
        return SimpleNamespace(logits=logits)


def _make_trainer_for_smoke() -> Qwen3DPOTrainer:
    trainer = Qwen3DPOTrainer.__new__(Qwen3DPOTrainer)
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

    row_1 = Qwen3DPOTrainer.process_row(
        {
            "images": ["img-a"],
            "prompt": "Why is this funny?",
            "chosen": "Because the visual and text conflict.",
            "rejected": "No sarcasm here.",
        },
        processor,
    )
    row_2 = Qwen3DPOTrainer.process_row(
        {
            "images": ["img-b"],
            "prompt": "Explain the joke.",
            "chosen": "Literal text contradicts the image.",
            "rejected": "It is straightforward.",
        },
        processor,
    )
    assert "image_grid_thw" in row_1 and "pixel_values" in row_1
    assert len(row_1["pixel_values"]) == 2

    collator = Qwen3DataCollatorForPreference(pad_token_id=processor.tokenizer.pad_token_id)
    batch = collator([row_1, row_2])
    assert "image_grid_thw" in batch
    assert batch["image_grid_thw"].shape[0] == 2

    trainer = _make_trainer_for_smoke()
    model = _FakeModel()
    output = trainer.concatenated_forward(model, batch)

    assert model.last_image_grid_thw is not None
    assert model.last_image_grid_thw.shape[0] == 4  # chosen + rejected concatenation
    assert "chosen_logps" in output and "rejected_logps" in output
    print("Qwen3 DPO trainer smoke test passed.")


if __name__ == "__main__":
    main()
