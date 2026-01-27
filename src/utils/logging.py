from __future__ import annotations

import os
from typing import Any, Dict, Optional

import wandb


def init_wandb(cfg: Dict[str, Any], run_name: str) -> Optional[wandb.sdk.wandb_run.Run]:
    if wandb.run is not None:
        return wandb.run

    has_api_key = bool(os.environ.get("WANDB_API_KEY"))
    if not has_api_key:
        os.environ["WANDB_DISABLED"] = "true"
        print("W&B disabled: WANDB_API_KEY not set.")
        return None

    return wandb.init(
        project=cfg.get("project"),
        entity=cfg.get("entity"),
        name=run_name,
        tags=cfg.get("tags"),
        config=cfg,
    )


def log_run_metadata(run: Optional[wandb.sdk.wandb_run.Run], meta: Dict[str, Any]) -> None:
    if run is None:
        return
    run.config.update(meta, allow_val_change=True)
