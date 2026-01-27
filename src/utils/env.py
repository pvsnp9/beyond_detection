from __future__ import annotations

import os
from typing import Any, Dict


def _get_cfg_value(cfg: Any, key: str) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key)
    return getattr(cfg, key, None)


def resolve_tokens_and_env(logistics_cfg: Any) -> Dict[str, Any]:
    # Resolve tokens without emitting secrets.
    hf_token = os.environ.get("HF_TOKEN") or _get_cfg_value(logistics_cfg, "hf_token")
    wandb_token = os.environ.get("WANDB_API_KEY") or _get_cfg_value(
        logistics_cfg, "wandb_token"
    )
    return {
        "hf_token": hf_token,
        "wandb_token": wandb_token,
        "has_hf_token": bool(hf_token),
        "has_wandb_token": bool(wandb_token),
    }


def set_runtime_env(resolved: Dict[str, Any]) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    hf_token = resolved.get("hf_token")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
