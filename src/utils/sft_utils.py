from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import os
import wandb
from transformers.trainer_callback import TrainerCallback
from config.logistics import Logistics
from src.datasets.loader import load_hf_dataset



class LogCallback(TrainerCallback):
    def __init__(self, logging_steps: int) -> None:
        self.logging_steps = logging_steps

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        if state.global_step % self.logging_steps == 0:
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            print(f"step={state.global_step} loss={loss} lr={lr}")


class EvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_fn,
        run: Optional[wandb.sdk.wandb_run.Run],
        eval_cfg: Dict[str, Any],
        eval_dataset: Any,
        eval_collator: Any,
        processor: Any,
    ) -> None:
        self.eval_fn = eval_fn
        self.run = run
        self.eval_cfg = eval_cfg
        self.eval_dataset = eval_dataset
        self.eval_collator = eval_collator
        self.processor = processor

    def on_evaluate(self, args, state, control, **kwargs):  # type: ignore[override]
        model = kwargs.get("model")
        if model is None:
            return
        try:
            metrics = self.eval_fn(
                model, self.processor, self.eval_dataset, self.eval_collator, self.eval_cfg
            )
            if self.run is not None:
                self.run.log({f"gen_eval/{k}": v for k, v in metrics.items()})
        except Exception as exc:
            print(f"Generation eval skipped: {exc}")

def get_sft_result_dir(model):
    path = os.path.join(Logistics().results_dir, model)
    os.makedirs(path, exist_ok=True)
    return path


def _dtype_from_str(dtype: str):
    import torch

    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return torch.float32


def _select_subset(dataset: Any, max_samples: Optional[int]) -> Any:
    if not max_samples:
        return dataset
    if hasattr(dataset, "select"):
        return dataset.select(range(min(len(dataset), max_samples)))
    return list(dataset.take(max_samples))


def _dataset_size(dataset: Any) -> Optional[int]:
    try:
        return len(dataset)
    except Exception:
        return None


def _load_eval_dataset(cfg: Dict[str, Any]) -> Tuple[Any, str]:
    try:
        eval_ds = load_hf_dataset(
            cfg["dataset"]["dataset_id"],
            split=cfg["dataset"]["eval_split"],
            config_name=cfg["dataset"]["lang"],
            streaming=cfg["dataset"]["streaming"],
            cache_dir=cfg["logistics"].hf_cache_dir,
            mode=cfg.get("mode")
        )
        return eval_ds, cfg["dataset"]["eval_split"]
    except Exception:
        try:
            eval_ds = load_hf_dataset(
                cfg["dataset"]["dataset_id"],
                split=cfg["dataset"]["fallback_eval_split"],
                config_name=cfg["dataset"]["lang"],
                streaming=cfg["dataset"]["streaming"],
                cache_dir=cfg["logistics"].hf_cache_dir,
            )
            return eval_ds, cfg["dataset"]["fallback_eval_split"]
        except Exception:
            train_ds = load_hf_dataset(
                cfg["dataset"]["dataset_id"],
                split=cfg["dataset"]["train_split"],
                config_name=cfg["dataset"]["lang"],
                streaming=cfg["dataset"]["streaming"],
                cache_dir=cfg["logistics"].hf_cache_dir,
            )
            print("Eval split not found; using a small slice of train for eval.")
            return _select_subset(train_ds, cfg["dataset"]["max_eval_samples"]), "train_slice"


def _count_trainable_params(model) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable, "total_params": total}


def _get_world_size() -> int:
    import torch

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1
