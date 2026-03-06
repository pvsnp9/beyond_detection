from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

from config.logistics import Logistics


def _extract_error_type(rejected_meta: Any) -> str:
    if isinstance(rejected_meta, dict):
        value = rejected_meta.get("error_type")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return "UNKNOWN_OR_MISSING"

    if isinstance(rejected_meta, str):
        text = rejected_meta.strip()
        if not text:
            return "UNKNOWN_OR_MISSING"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return "UNKNOWN_OR_MISSING"
        if isinstance(parsed, dict):
            value = parsed.get("error_type")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "UNKNOWN_OR_MISSING"


def _choose_config_name(dataset_id: str, preferred: str = "en") -> str | None:
    from datasets import get_dataset_config_names

    try:
        configs = list(get_dataset_config_names(dataset_id))
    except Exception:
        return None

    if not configs:
        return None
    if preferred in configs:
        return preferred
    return configs[0]


def compute_error_type_counts(dataset_id: str, split: str = "train") -> Counter[str]:
    from datasets import load_dataset

    config_name = _choose_config_name(dataset_id, preferred="en")
    if config_name is None:
        dataset = load_dataset(dataset_id, split=split)
    else:
        dataset = load_dataset(dataset_id, name=config_name, split=split)

    counts: Counter[str] = Counter()
    for row in dataset:
        counts[_extract_error_type(row.get("rejected_meta"))] += 1
    return counts


def create_error_type_chart(
    counts: Counter[str],
    output_path: str,
    title: str = "Error Type Distribution on Rejected mDPO Train Dataset",
) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    labels = [label for label, _ in items]
    values = [value for _, value in items]

    plt.style.use("default")
    # Single-column friendly footprint with horizontal bars for readable labels.
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    colors = ["#4C78A8"] * len(labels)
    bars = ax.barh(labels, values, color=colors, alpha=0.78, height=0.56, linewidth=0.0)

    ax.set_title(title, fontsize=8.0, pad=5)
    ax.set_xlabel("Count", fontsize=7.0)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=6.2, length=2.0, width=0.6)
    ax.tick_params(axis="y", labelsize=5.9, length=0)
    ax.invert_yaxis()

    # Remove chart frame for a cleaner, publication-style look.
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)

    xmax = max(values) if values else 0
    ax.set_xlim(0, xmax * 1.14 if xmax else 1)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(6, int(0.008 * xmax)),
            bar.get_y() + bar.get_height() / 2.0,
            f"{value}",
            ha="left",
            va="center",
            fontsize=6.0,
            color="#1A1A1A",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=900, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build error_type stats and bar chart for rejected mDPO train data."
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to analyze (default: train).",
    )
    args = parser.parse_args()

    logistics = Logistics()
    dataset_id = logistics.hf_dpo_ds_id

    counts = compute_error_type_counts(dataset_id=dataset_id, split=args.split)
    if not counts:
        raise RuntimeError(
            f"No rows found for dataset='{dataset_id}' split='{args.split}'."
        )

    output_dir = os.path.join(logistics.project_root_dir, "outputs", "reports", "plots")
    output_path = os.path.join(output_dir, "error_type.png")
    create_error_type_chart(counts=counts, output_path=output_path)

    print(f"Dataset ID: {dataset_id}")
    print(f"Split: {args.split}")
    print(f"Total rows: {sum(counts.values())}")
    print("Error type counts:")
    for error_type, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {error_type}: {count}")
    print(f"Saved chart: {output_path}")


if __name__ == "__main__":
    main()
