import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from datasets import ClassLabel, Image, Value, get_dataset_config_names, load_dataset
from config.logistics import Logistics

OUTPUT_FILENAME = "sft_datacard.txt"
ALLOWED_LANGS = ("en", "es", "zh")


@dataclass
class LengthStats:
    total: float = 0.0
    total_sq: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.total += value
        self.total_sq += value * value
        self.count += 1

    def mean_std(self) -> tuple[float, float]:
        if self.count == 0:
            return 0.0, 0.0
        mean = self.total / self.count
        var = max(0.0, (self.total_sq / self.count) - (mean * mean))
        return mean, math.sqrt(var)


def latex_escape(text: Any) -> str:
    value = "" if text is None else str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in value)


def to_percent(num: int, den: int) -> float:
    if den == 0:
        return 0.0
    return (100.0 * num) / den


def format_count_pct(num: int, den: int) -> str:
    return f"{num:,} ({to_percent(num, den):.1f}%)"


def _is_string_feature(feature: Any) -> bool:
    return isinstance(feature, Value) and getattr(feature, "dtype", None) == "string"


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_lang_tag(value: str) -> str | None:
    token = value.strip().lower()
    if not token:
        return None
    alias_map = {
        "en": "en",
        "eng": "en",
        "english": "en",
        "es": "es",
        "spa": "es",
        "spanish": "es",
        "zh": "zh",
        "zho": "zh",
        "chi": "zh",
        "chinese": "zh",
        "cn": "zh",
    }
    if token in alias_map:
        return alias_map[token]
    # Keep tags like en-us / zh-cn / es-mx by base prefix.
    if "-" in token:
        base = token.split("-", 1)[0]
        return alias_map.get(base)
    if "_" in token:
        base = token.split("_", 1)[0]
        return alias_map.get(base)
    return None


def _extract_flags(value: Any) -> set[str]:
    if value is None:
        return set()

    if isinstance(value, dict):
        out = set()
        for key, flag_val in value.items():
            if not isinstance(key, str):
                continue
            if isinstance(flag_val, bool):
                if flag_val:
                    out.add(key.strip())
            elif isinstance(flag_val, (int, float)):
                if flag_val != 0:
                    out.add(key.strip())
        return {f for f in out if f}

    if isinstance(value, (list, tuple, set)):
        out = {
            item.strip()
            for item in value
            if isinstance(item, str) and item.strip()
        }
        return out

    return set()


def _has_image(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        path_val = value.get("path")
        bytes_val = value.get("bytes")
        if isinstance(path_val, str) and path_val.strip():
            return True
        return bytes_val is not None
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _normalize_modality(value: Any, has_caption: bool, has_image: bool) -> str:
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"both", "paired", "multimodal", "image+caption"}:
            return "both"
        if token in {"text", "text_only", "text-only", "caption"}:
            return "text"
        if token in {"image", "image_only", "image-only", "vision"}:
            return "image"
    if has_caption and has_image:
        return "both"
    if has_caption and not has_image:
        return "text"
    if has_image and not has_caption:
        return "image"
    return "unknown"


def _normalize_label_from_string(value: str) -> str:
    token = value.strip().lower()
    if token in {"sarcastic", "sarcasm", "sarc", "1", "positive", "yes", "true"}:
        return "sarcastic"
    if token in {
        "non_sarcastic",
        "non-sarcastic",
        "non sarcastic",
        "nonsarcastic",
        "not_sarcastic",
        "literal",
        "0",
        "negative",
        "false",
        "no",
    }:
        return "non_sarcastic"
    if token in {"unknown", "uncertain", "ambiguous", "neutral", "2", "other"}:
        return "unknown"
    return "unknown"


def _normalize_label(raw_label: Any, feature: Any | None) -> str:
    if isinstance(feature, ClassLabel):
        try:
            raw_label = feature.int2str(int(raw_label))
        except Exception:
            pass

    if isinstance(raw_label, str):
        return _normalize_label_from_string(raw_label)

    try:
        iv = int(raw_label)
    except (TypeError, ValueError):
        return "unknown"

    if iv == 1:
        return "sarcastic"
    if iv == 0:
        return "non_sarcastic"
    if iv == 2:
        return "unknown"
    return "unknown"


def _get_union_columns(ds_dict: Any) -> list[str]:
    cols = set()
    for split in ds_dict.keys():
        cols.update(ds_dict[split].column_names)
    return sorted(cols)


def _feature_for_col(ds_dict: Any, col: str) -> Any | None:
    for split in ds_dict.keys():
        features = ds_dict[split].features
        if col in features:
            return features[col]
    return None


def infer_columns(ds_dict: Any) -> dict[str, str | None]:
    columns = _get_union_columns(ds_dict)
    features_by_col = {col: _feature_for_col(ds_dict, col) for col in columns}

    string_cols = [col for col in columns if _is_string_feature(features_by_col[col])]
    classlabel_cols = [col for col in columns if isinstance(features_by_col[col], ClassLabel)]
    image_cols = [col for col in columns if isinstance(features_by_col[col], Image)]

    str_len_sum = defaultdict(float)
    str_len_cnt = defaultdict(int)
    str_uniques: dict[str, set[str]] = {col: set() for col in string_cols}
    flags_like_hits = Counter()

    for split in ds_dict.keys():
        for row in ds_dict[split]:
            if not isinstance(row, dict):
                continue
            for col in string_cols:
                val = row.get(col)
                if isinstance(val, str):
                    text = val.strip()
                    if text:
                        str_len_sum[col] += len(text)
                        str_len_cnt[col] += 1
                        if len(str_uniques[col]) < 50:
                            str_uniques[col].add(text.lower())

            for col in columns:
                if col not in row:
                    continue
                flag_set = _extract_flags(row.get(col))
                if flag_set:
                    flags_like_hits[col] += 1

    if "caption" in string_cols:
        caption_col = "caption"
    else:
        caption_col = None
        best_avg = -1.0
        for col in string_cols:
            if str_len_cnt[col] == 0:
                continue
            avg_len = str_len_sum[col] / str_len_cnt[col]
            if avg_len > best_avg:
                best_avg = avg_len
                caption_col = col

    if "label" in columns:
        label_col = "label"
    elif classlabel_cols:
        rank = sorted(classlabel_cols, key=lambda c: (0 if "label" in c.lower() else 1, c))
        label_col = rank[0]
    else:
        small_unique_cols = []
        for col in string_cols:
            ucount = len(str_uniques[col])
            if 1 <= ucount <= 10:
                small_unique_cols.append(col)
        if small_unique_cols:
            rank = sorted(small_unique_cols, key=lambda c: (0 if "label" in c.lower() else 1, c))
            label_col = rank[0]
        else:
            label_col = None

    if "image" in image_cols:
        image_col = "image"
    else:
        image_col = sorted(image_cols)[0] if image_cols else None

    if "quality_flags" in columns:
        flags_col = "quality_flags"
    elif "flags" in columns:
        flags_col = "flags"
    else:
        flag_rank = [c for c, _ in flags_like_hits.most_common()]
        flags_col = flag_rank[0] if flag_rank else None

    lang_col = None
    for candidate in ("lang", "language"):
        if candidate in columns:
            lang_col = candidate
            break

    modality_col = "modality" if "modality" in columns else None

    return {
        "caption_col": caption_col,
        "label_col": label_col,
        "image_col": image_col,
        "flags_col": flags_col,
        "lang_col": lang_col,
        "modality_col": modality_col,
    }


def _extract_metadata(dataset_id: str) -> dict[str, Any]:
    out = {
        "license": "[FILL]",
        "description": "",
        "version_release": "[FILL]",
        "split_strategy": "Not specified",
        "label_source": "[FILL]",
        "annotators": "[FILL]",
        "agreement": "[FILL]",
    }
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(dataset_id)
    except Exception:
        return out

    card = getattr(info, "cardData", None)
    card = card if isinstance(card, dict) else {}

    license_val = card.get("license") or getattr(info, "license", None)
    if license_val:
        out["license"] = str(license_val)

    description_val = card.get("dataset_summary") or card.get("description") or getattr(info, "description", None)
    if isinstance(description_val, str) and description_val.strip():
        out["description"] = description_val.strip()

    sha = getattr(info, "sha", None)
    last_modified = (
        getattr(info, "lastModified", None)
        or getattr(info, "last_modified", None)
        or card.get("release_date")
    )
    if sha or last_modified:
        sha_text = f"v{str(sha)[:7]}" if sha else "v[FILL]"
        date_text = str(last_modified) if last_modified else "[FILL]"
        out["version_release"] = f"{sha_text} / {date_text}"

    split_strategy = (
        card.get("split_strategy")
        or card.get("splits")
        or card.get("data_splits")
        or card.get("split")
    )
    if split_strategy:
        out["split_strategy"] = str(split_strategy)

    label_source = card.get("label_source") or card.get("labels") or card.get("annotation_type")
    if label_source:
        out["label_source"] = str(label_source)

    annotators = card.get("annotators") or card.get("annotations_creators")
    if annotators:
        out["annotators"] = str(annotators)

    agreement = card.get("agreement") or card.get("inter_annotator_agreement")
    if agreement:
        out["agreement"] = str(agreement)

    return out


def compute_stats(ds_dict: Any, inferred: dict[str, str | None]) -> dict[str, Any]:
    split_stats: dict[str, dict[str, Any]] = {}
    overall_total = 0
    overall_classes = Counter({"sarcastic": 0, "non_sarcastic": 0, "unknown": 0})
    overall_len = LengthStats()
    overall_flags = Counter()
    overall_modality = Counter({"both": 0, "text": 0, "image": 0, "unknown": 0})
    overall_lang = Counter()
    paired_count = 0
    missing_modality_unknown_all = True
    saw_missing_modality = False

    caption_col = inferred["caption_col"]
    label_col = inferred["label_col"]
    image_col = inferred["image_col"]
    flags_col = inferred["flags_col"]
    lang_col = inferred["lang_col"]
    modality_col = inferred["modality_col"]

    label_feature = _feature_for_col(ds_dict, label_col) if label_col else None

    for split in ds_dict.keys():
        class_counts = Counter({"sarcastic": 0, "non_sarcastic": 0, "unknown": 0})
        len_stats = LengthStats()
        flag_counts = Counter()
        modality_counts = Counter({"both": 0, "text": 0, "image": 0, "unknown": 0})
        lang_counts = Counter()
        split_total = 0

        for row in ds_dict[split]:
            if not isinstance(row, dict):
                continue

            split_total += 1
            overall_total += 1

            caption_text = _safe_string(row.get(caption_col)) if caption_col else ""
            word_len = len(caption_text.split())
            len_stats.update(float(word_len))
            overall_len.update(float(word_len))

            raw_label = row.get(label_col) if label_col else None
            norm_label = _normalize_label(raw_label, label_feature)
            class_counts[norm_label] += 1
            overall_classes[norm_label] += 1

            flags = _extract_flags(row.get(flags_col)) if flags_col else set()
            for flag in sorted(flags):
                flag_counts[flag] += 1
                overall_flags[flag] += 1

            has_caption = bool(caption_text.strip())
            has_image = _has_image(row.get(image_col)) if image_col else False
            if has_caption and has_image:
                paired_count += 1

            modality = _normalize_modality(row.get(modality_col), has_caption=has_caption, has_image=has_image)
            modality_counts[modality] += 1
            overall_modality[modality] += 1

            if modality != "both":
                saw_missing_modality = True
                if norm_label != "unknown":
                    missing_modality_unknown_all = False

            if lang_col:
                lang_val = _safe_string(row.get(lang_col)).strip()
                if lang_val:
                    lang_key = _normalize_lang_tag(lang_val)
                    if lang_key in ALLOWED_LANGS:
                        lang_counts[lang_key] += 1
                        overall_lang[lang_key] += 1

        split_stats[split] = {
            "total": split_total,
            "class_counts": class_counts,
            "length": len_stats,
            "flags": flag_counts,
            "modality": modality_counts,
            "language": lang_counts,
        }

    mean_words, std_words = overall_len.mean_std()
    if saw_missing_modality and missing_modality_unknown_all:
        missing_policy = 'label="unknown" when modality missing'
    else:
        missing_policy = "Not specified"

    return {
        "overall_total": overall_total,
        "overall_classes": overall_classes,
        "overall_len": overall_len,
        "overall_mean_words": mean_words,
        "overall_std_words": std_words,
        "overall_flags": overall_flags,
        "overall_modality": overall_modality,
        "overall_lang": overall_lang,
        "paired_count": paired_count,
        "split_stats": split_stats,
        "missing_policy": missing_policy,
    }


def _merge_length_stats(target: LengthStats, source: LengthStats) -> None:
    target.total += source.total
    target.total_sq += source.total_sq
    target.count += source.count


def merge_stats(acc: dict[str, Any] | None, part: dict[str, Any]) -> dict[str, Any]:
    if acc is None:
        # Create a fresh accumulator compatible with compute_stats output.
        acc = {
            "overall_total": 0,
            "overall_classes": Counter({"sarcastic": 0, "non_sarcastic": 0, "unknown": 0}),
            "overall_len": LengthStats(),
            "overall_mean_words": 0.0,
            "overall_std_words": 0.0,
            "overall_flags": Counter(),
            "overall_modality": Counter({"both": 0, "text": 0, "image": 0, "unknown": 0}),
            "overall_lang": Counter(),
            "paired_count": 0,
            "split_stats": {},
            "missing_policy": 'label="unknown" when modality missing',
        }

    acc["overall_total"] += part["overall_total"]
    acc["overall_classes"].update(part["overall_classes"])
    _merge_length_stats(acc["overall_len"], part["overall_len"])
    acc["overall_flags"].update(part["overall_flags"])
    acc["overall_modality"].update(part["overall_modality"])
    acc["overall_lang"].update(part["overall_lang"])
    acc["paired_count"] += part["paired_count"]

    for split, split_data in part["split_stats"].items():
        if split not in acc["split_stats"]:
            acc["split_stats"][split] = {
                "total": 0,
                "class_counts": Counter({"sarcastic": 0, "non_sarcastic": 0, "unknown": 0}),
                "length": LengthStats(),
                "flags": Counter(),
                "modality": Counter({"both": 0, "text": 0, "image": 0, "unknown": 0}),
                "language": Counter(),
            }
        tgt = acc["split_stats"][split]
        tgt["total"] += split_data["total"]
        tgt["class_counts"].update(split_data["class_counts"])
        _merge_length_stats(tgt["length"], split_data["length"])
        tgt["flags"].update(split_data["flags"])
        tgt["modality"].update(split_data["modality"])
        tgt["language"].update(split_data["language"])

    if part["missing_policy"] != 'label="unknown" when modality missing':
        acc["missing_policy"] = "Not specified"

    mean_words, std_words = acc["overall_len"].mean_std()
    acc["overall_mean_words"] = mean_words
    acc["overall_std_words"] = std_words
    return acc


def load_all_configs_dataset(dataset_id: str) -> list[tuple[str | None, Any]]:
    configs: list[str] = []
    try:
        configs = list(get_dataset_config_names(dataset_id))
    except Exception:
        configs = []

    if not configs:
        return [(None, load_dataset(dataset_id))]

    out: list[tuple[str | None, Any]] = []
    for config_name in configs:
        out.append((config_name, load_dataset(dataset_id, config_name)))
    return out


def _format_split_sizes(split_stats: dict[str, dict[str, Any]]) -> str:
    lower_map = {name.lower(): name for name in split_stats.keys()}
    train_key = lower_map.get("train")
    val_key = lower_map.get("validation") or lower_map.get("val") or lower_map.get("dev")
    test_key = lower_map.get("test")

    if train_key or val_key or test_key:
        train_n = split_stats.get(train_key, {}).get("total", 0) if train_key else 0
        val_n = split_stats.get(val_key, {}).get("total", 0) if val_key else 0
        test_n = split_stats.get(test_key, {}).get("total", 0) if test_key else 0
        return f"{train_n:,} / {val_n:,} / {test_n:,}"

    parts = [f"{k}: {v['total']:,}" for k, v in split_stats.items()]
    return "; ".join(parts) if parts else "Not specified"


def _format_flag_prevalence(flag_counts: Counter[str], total: int) -> str:
    if not flag_counts:
        return "Not available"
    chunks = []
    for flag, count in sorted(flag_counts.items(), key=lambda x: x[0]):
        chunks.append(f"{flag}: {to_percent(count, total):.1f}%")
    return ", ".join(chunks)


def _format_lang_distribution(lang_counts: Counter[str], total: int) -> str:
    filtered = {lang: lang_counts.get(lang, 0) for lang in ALLOWED_LANGS if lang_counts.get(lang, 0) > 0}
    if not filtered:
        return "[FILL]"
    parts = [f"{lang}: {to_percent(count, total):.1f}%" for lang, count in filtered.items()]
    return ", ".join(parts)


def build_latex_table(dataset_id: str, stats: dict[str, Any], metadata: dict[str, Any], inferred: dict[str, str | None]) -> str:
    total = stats["overall_total"]
    class_counts = stats["overall_classes"]
    mean_words = stats["overall_mean_words"]
    std_words = stats["overall_std_words"]
    split_stats = stats["split_stats"]
    paired_count = stats["paired_count"]
    modality = stats["overall_modality"]
    lang_dist = _format_lang_distribution(stats["overall_lang"], total)
    flags_available = ", ".join(sorted(stats["overall_flags"].keys())) if stats["overall_flags"] else "Not available"
    flags_prevalence = _format_flag_prevalence(stats["overall_flags"], total)
    split_sizes = _format_split_sizes(split_stats)

    paired_text = f"{paired_count:,}/{total:,} ({to_percent(paired_count, total):.1f}%) have image+caption"
    modality_text = (
        f"both {to_percent(modality['both'], total):.1f}% / "
        f"text-only {to_percent(modality['text'], total):.1f}% / "
        f"image-only {to_percent(modality['image'], total):.1f}% / "
        f"unknown {to_percent(modality['unknown'], total):.1f}%"
    )

    rows = [
        ("Dataset", "Name", dataset_id),
        ("", "Version / Release date", metadata["version_release"]),
        ("", "Task", "Multimodal sarcasm detection (image+caption)"),
        ("Size", "Total examples", f"{total:,}"),
        ("", "Sarcastic", format_count_pct(class_counts["sarcastic"], total)),
        ("", "Non-sarcastic", format_count_pct(class_counts["non_sarcastic"], total)),
        ("", "Unknown", format_count_pct(class_counts["unknown"], total)),
        ("", "Avg caption length", f"{mean_words:.2f} ± {std_words:.2f} (words)"),
        ("Language", "Languages covered", "English (en), Spanish (es), Chinese (zh)"),
        ("", "Caption language distribution", lang_dist),
        ("", "Code-switching rate", "[FILL]"),
        ("", "Non-standard text rate", "[FILL]"),
        ("Splits", "Train/Val/Test", split_sizes),
        ("", "Split strategy", metadata["split_strategy"]),
        ("Modalities", "Paired availability", paired_text),
        ("", "Training modality mix", modality_text),
        ("", "Missing-modality policy", stats["missing_policy"]),
        ("Annotations", "Label source", metadata["label_source"]),
        ("", "Annotators", metadata["annotators"]),
        ("", "Agreement", metadata["agreement"]),
        ("Quality flags", "Available flags", flags_available),
        ("", "Flag prevalence (ALL)", flags_prevalence),
        ("Evaluation", "Primary metric", "Macro-F1 (paired/both)"),
        ("", "Reliability metrics", "JSON parse rate, schema-valid rate"),
        ("", "Unimodal robustness", "unknown-rate, hallucination-rate"),
        ("Ethics & Licensing", "Content notes", metadata["description"] if metadata["description"] else "[FILL]"),
        ("", "License", metadata["license"]),
        ("Availability", "URL / Access", f"https://huggingface.co/datasets/{dataset_id}"),
    ]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llccc|ccc|ccc}",
        r"\toprule",
        r"% {content}",
        r"Section & Field & \multicolumn{9}{l}{Value} \\",
        r"\midrule",
    ]

    active_section = ""
    for section, field, value in rows:
        if section and active_section and section != active_section:
            lines.append(r"\midrule")
        if section:
            active_section = section
        lines.append(
            f"{latex_escape(section)} & {latex_escape(field)} & "
            f"\\multicolumn{{9}}{{l}}{{{latex_escape(value)}}} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{[FILL]}",
            r"\label{tbl:[FILL]}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    logistics = Logistics()
    random.seed(logistics.seed)

    dataset_id = logistics.hf_sft_ds_id
    reports_dir = logistics.reports_dir
    if not os.path.isabs(reports_dir):
        reports_dir = os.path.join(logistics.project_root_dir, reports_dir)
    out_dir = os.path.join(reports_dir, "latexstats")
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, OUTPUT_FILENAME)

    datasets_by_config = load_all_configs_dataset(dataset_id)
    inferred_cols_by_config: dict[str, dict[str, str | None]] = {}
    per_config_summary: dict[str, Any] = {}
    stats: dict[str, Any] | None = None
    for config_name, ds_dict in datasets_by_config:
        inferred = infer_columns(ds_dict)
        config_key = config_name if config_name is not None else "default"
        inferred_cols_by_config[config_key] = inferred
        part_stats = compute_stats(ds_dict, inferred)
        per_config_summary[config_key] = {
            "total_examples": part_stats["overall_total"],
            "per_split": {
                split: split_data["total"]
                for split, split_data in part_stats["split_stats"].items()
            },
            "class_counts": dict(part_stats["overall_classes"]),
            "avg_caption_length_words": {
                "mean": round(part_stats["overall_mean_words"], 4),
                "std": round(part_stats["overall_std_words"], 4),
            },
            "flags_count": len(part_stats["overall_flags"]),
        }
        stats = merge_stats(stats, part_stats)

    if stats is None:
        raise RuntimeError(f"Could not load dataset/config for: {dataset_id}")

    metadata = _extract_metadata(dataset_id)

    # build_latex_table does not depend on inferred columns for rendering.
    latex = build_latex_table(dataset_id, stats, metadata, inferred={})
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)

    summary = {
        "totals": {
            "overall_examples": stats["overall_total"],
            "per_split": {split: split_data["total"] for split, split_data in stats["split_stats"].items()},
        },
        "class_counts": dict(stats["overall_classes"]),
        "class_counts_per_split": {
            split: dict(split_data["class_counts"])
            for split, split_data in stats["split_stats"].items()
        },
        "avg_caption_length_words": {
            "mean": round(stats["overall_mean_words"], 4),
            "std": round(stats["overall_std_words"], 4),
        },
        "flags_count": len(stats["overall_flags"]),
        "per_config": per_config_summary,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote LaTeX table to: {output_path}")
    print(
        "Inferred columns by config -> "
        f"{json.dumps(inferred_cols_by_config, ensure_ascii=False)}"
    )


if __name__ == "__main__":
    main()
