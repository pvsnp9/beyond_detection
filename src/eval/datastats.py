import json
import os
from typing import Any, Callable

from config.logistics import Logistics


def build_flag_template(flag_labels, label_names=None):
    labels = list(label_names) if label_names else ["sarcastic", "non_sarcastic", "unknown"]
    return {flag: {label: 0 for label in labels} for flag in flag_labels}


def update_counts(flag_dict, flags, label_name, none_label):
    if label_name not in flag_dict[none_label]:
        label_name = "unknown"

    if not flags:
        flag_dict[none_label][label_name] += 1
        return

    for flag in flags:
        if flag not in flag_dict:
            continue
        flag_dict[flag][label_name] += 1


def _update_quality_flag_bins(
    flag_bins: dict[str, dict[str, int]],
    label: str,
    quality_flags: Any,
) -> None:
    if label not in flag_bins:
        label = "unknown"

    if isinstance(quality_flags, list):
        normalized_flags = [f.strip() for f in quality_flags if isinstance(f, str) and f.strip()]
    else:
        normalized_flags = []

    if not normalized_flags:
        flag_bins[label]["NONE"] = flag_bins[label].get("NONE", 0) + 1
        return

    # Count each flag independently; one example can increment multiple bins.
    for flag in normalized_flags:
        flag_bins[label][flag] = flag_bins[label].get(flag, 0) + 1


def build_sft_rows_and_stats(
    raw_rows: list[dict[str, Any]],
    row_builder: Callable[..., dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = ("sarcastic", "non_sarcastic", "unknown")
    totals = {label: 0 for label in labels}
    flag_bins = {label: {} for label in labels}
    skipped = 0
    built_rows: list[dict[str, Any]] = []

    for raw in raw_rows:
        row = row_builder(raw)
        if not row or not isinstance(row, dict):
            skipped += 1
            continue

        label = row.get("label_gt")
        if label not in totals:
            label = "unknown"
        totals[label] += 1
        _update_quality_flag_bins(flag_bins, label, row.get("quality_flags"))
        built_rows.append(row)

    return built_rows, {"totals": totals, "flag_bins": flag_bins, "skipped": skipped}


def print_sft_split_stats(split: str, stats: dict[str, Any], written: int) -> None:
    totals = stats.get("totals", {})
    flag_bins = stats.get("flag_bins", {})
    skipped = stats.get("skipped", 0)
    pos = totals.get("sarcastic", 0)
    neg = totals.get("non_sarcastic", 0)
    unknown = totals.get("unknown", 0)

    print(
        f"[{split}] totals -> sarcastic(+): {pos}, non_sarcastic(-): {neg}, "
        f"unknown: {unknown}, written: {written}, skipped_raw: {skipped}"
    )

    for label in ("sarcastic", "non_sarcastic", "unknown"):
        bins = flag_bins.get(label, {})
        if not bins:
            print(f"[{split}] {label} quality_flags -> NONE:0")
            continue
        sorted_bins = sorted(bins.items(), key=lambda item: (-item[1], item[0]))
        bins_text = ", ".join(f"{flag}:{count}" for flag, count in sorted_bins)
        print(f"[{split}] {label} quality_flags -> {bins_text}")


def main():
    from datasets import load_dataset

    logistics = Logistics()
    dataset = load_dataset(logistics.hf_sft_ds_id, name="en")
    label_names = dataset["train"].features["label_gt"].names

    flag_labels = [
        "low_image_clarity",
        "possible_ocr_text_in_image",
        "caption_too_short",
        "caption_not_sarcastic_or_unclear",
        "ambiguous_sarcasm",
        "multiple_possible_incongruities",
        "requires_world_knowledge",
        "possible_identity_inference_risk",
        "NONE",
    ]
    none_label = "NONE"

    results = {}
    for split in logistics.splits:
        if split not in dataset:
            continue
        flag_dict = build_flag_template(flag_labels, label_names=label_names)

        for record in dataset[split]:
            flags = record.get("quality_flags") or []
            label_idx = record.get("label_gt")
            label_name = label_names[int(label_idx)]
            update_counts(flag_dict, flags, label_name, none_label)

        results[split] = flag_dict

    reports_dir = logistics.reports_dir
    plots_dir = os.path.join(reports_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, "quality_stats.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)

    print(json.dumps(results, indent=2, ensure_ascii=True))

    plot_path = os.path.join(plots_dir, "quality_stats.png")
    create_chart(output_path, plot_path)


def create_chart(json_path, output_path):
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for split, flags in data.items():
        for flag, labels in flags.items():
            for label, count in labels.items():
                records.append(
                    {
                        "Split": split,
                        "Quality Flag": flag,
                        "Label": label,
                        "Count": count,
                    }
                )

    df = pd.DataFrame(records)
    df["Split"] = df["Split"].str.capitalize()

    sns.set_theme(style="whitegrid")
    times_path = os.environ.get("TIMES_NEW_ROMAN_FONT_PATH")
    if times_path and os.path.isfile(times_path):
        fm.fontManager.addfont(times_path)
        plt.rcParams["font.family"] = "Times New Roman"
    elif any(font.name == "Times New Roman" for font in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = "Times New Roman"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    g = sns.catplot(
        data=df,
        kind="bar",
        x="Count",
        y="Quality Flag",
        hue="Label",
        col="Split",
        palette="magma",
        height=5,
        aspect=1.2,
        sharex=False,
    )
    g.set_axis_labels("Frequency", "Quality Flags")
    g.set_titles("{col_name}")
    g.despine(left=True)
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle(
        "Distribution of Quality Flags by Sarcasm Label and Split", fontsize=16
    )
    g.fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(g.fig)






# if __name__ == "__main__":
#     # main()
#     plots_dir = os.path.join(Logistics().project_root_dir, Logistics().reports_dir, "plots")
#     os.makedirs(plots_dir, exist_ok=True)
#     create_chart(
#         json_path=os.path.join(plots_dir,"quality_stats.json"),
#         output_path=os.path.join(plots_dir,"quality_stats.png")
#     )
