import json
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset

from config.logistics import Logistics


def build_flag_template(flag_labels):
    return {flag: {"sarcastic": 0, "non_sarcastic": 0} for flag in flag_labels}


def update_counts(flag_dict, flags, label_name, none_label):
    if not flags:
        flag_dict[none_label][label_name] += 1
        return

    for flag in flags:
        if flag not in flag_dict:
            continue
        flag_dict[flag][label_name] += 1


def main():
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
        flag_dict = build_flag_template(flag_labels)

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






if __name__ == "__main__":
    # main()
    plots_dir = os.path.join(Logistics().project_root_dir, Logistics().reports_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    create_chart(
        json_path=os.path.join(plots_dir,"quality_stats.json"),
        output_path=os.path.join(plots_dir,"quality_stats.png")
    )
