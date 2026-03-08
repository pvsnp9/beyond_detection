import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    from config.logistics import Logistics
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from config.logistics import Logistics


def _resolve_input_path(redeval_dir: Path) -> Path:
    candidates = [
        redeval_dir / "test.json",
        redeval_dir / "test" / "test.json",
        redeval_dir / "test",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find test input JSON in {redeval_dir}")


def _read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON at {path}, found {type(payload).__name__}")
    return [item for item in payload if isinstance(item, dict)]


def _label_to_text(label: Any) -> str:
    return "sarcastic" if int(label) == 1 else "non_sarcastic"


def _invert_label(label_gt: str) -> str:
    if label_gt == "sarcastic":
        return "non_sarcastic"
    if label_gt == "non_sarcastic":
        return "sarcastic"
    return label_gt


def _sample_rows(rows: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot sample from empty list")
    if len(rows) >= n:
        return rng.sample(rows, n)
    print(
        f"Warning: requested {n} rows from pool size {len(rows)}. "
        "Using oversampling with replacement to maintain target balance."
    )
    return rng.choices(rows, k=n)


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _print_stats(records: List[Dict[str, Any]], tag: str) -> None:
    print(f"\n=== {tag} Stats ===")
    print(f"total={len(records)}")

    modality_counter = Counter(rec.get("modality", "unknown") for rec in records)
    label_counter = Counter(rec.get("label_gt", "unknown") for rec in records)
    print(f"modality={dict(modality_counter)}")
    print(f"label_gt={dict(label_counter)}")

    by_modality: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        by_modality[rec.get("modality", "unknown")][rec.get("label_gt", "unknown")] += 1

    print("label_by_modality:")
    for modality in sorted(by_modality):
        print(f"  {modality}: {dict(by_modality[modality])}")

    if any("perturbation_tpye" in rec for rec in records):
        perturb_counter = Counter(rec.get("perturbation_tpye", "unknown") for rec in records)
        by_perturbation: Dict[str, Counter] = defaultdict(Counter)
        for rec in records:
            by_perturbation[rec.get("perturbation_tpye", "unknown")][
                rec.get("label_gt", "unknown")
            ] += 1
        print(f"perturbation={dict(perturb_counter)}")
        print("label_by_perturbation:")
        for ptype in sorted(by_perturbation):
            print(f"  {ptype}: {dict(by_perturbation[ptype])}")


def _build_ood_rows(filtered: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    positives = [dict(row) for row in filtered if row["label_gt"] == "sarcastic"]
    negatives = [dict(row) for row in filtered if row["label_gt"] == "non_sarcastic"]
    if not positives or not negatives:
        raise ValueError(
            "Filtered data does not include both classes: "
            f"pos={len(positives)}, neg={len(negatives)}"
        )

    # target: 400 rows with overall balanced classes (200/200)
    selected_pos = _sample_rows(positives, 200, rng)
    selected_neg = _sample_rows(negatives, 200, rng)
    rng.shuffle(selected_pos)
    rng.shuffle(selected_neg)

    # both=250 (125/125), text=75 (38/37), image=75 (37/38)
    both_rows = selected_pos[:125] + selected_neg[:125]
    text_rows = selected_pos[125:163] + selected_neg[125:162]
    image_rows = selected_pos[163:200] + selected_neg[162:200]
    rng.shuffle(both_rows)
    rng.shuffle(text_rows)
    rng.shuffle(image_rows)

    ood_rows: List[Dict[str, Any]] = []
    for row in both_rows:
        rec = dict(row)
        rec["modality"] = "both"
        ood_rows.append(rec)
    for row in text_rows:
        rec = dict(row)
        rec["modality"] = "text"
        ood_rows.append(rec)
    for row in image_rows:
        rec = dict(row)
        rec["modality"] = "image"
        ood_rows.append(rec)

    rng.shuffle(ood_rows)
    return ood_rows


def _build_perturb_rows_from_both(both_rows: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    perturb_rows = [dict(row) for row in both_rows]
    rng.shuffle(perturb_rows)

    half = len(perturb_rows) // 2
    text_targets = perturb_rows[:half]
    image_targets = perturb_rows[half:]

    for rec in text_targets:
        donor = rng.choice(image_targets)
        rec["caption"] = donor["caption"]
        rec["perturbation_tpye"] = "text"
        rec["label_gt"] = _invert_label(rec["label_gt"])

    for rec in image_targets:
        donor = rng.choice(text_targets)
        rec["id"] = donor["id"]
        rec["image"] = donor["image"]
        rec["perturbation_tpye"] = "image"
        rec["label_gt"] = _invert_label(rec["label_gt"])

    rng.shuffle(perturb_rows)
    return perturb_rows


def build_redeval_ood() -> None:
    try:
        logistics = Logistics()
        rng = random.Random(logistics.seed)

        redeval_dir = Path(logistics.project_root_dir) / logistics.data_generation_dir / "redeval"
        images_dir = redeval_dir / "images"
        input_path = _resolve_input_path(redeval_dir)
        output_path_ood = redeval_dir / "ood.jsonl"
        output_path_perturb = redeval_dir / "ood_perturb.jsonl"

        raw_records = _read_json(input_path)

        filtered: List[Dict[str, Any]] = []
        for raw in raw_records:
            image_id = str(raw.get("image_id", "")).strip()
            caption = str(raw.get("text", "")).strip()
            label = raw.get("label", None)
            image_path = images_dir / f"{image_id}.jpg"
            if not image_id or label not in (0, 1):
                continue
            if len(caption) <= 50:
                continue
            if not image_path.exists():
                continue
            filtered.append(
                {
                    "id": image_id,
                    "image": str(image_path.resolve()),
                    "caption": caption,
                    "label_gt": _label_to_text(label),
                }
            )

        ood_rows = _build_ood_rows(filtered, rng)

        both_rows = [dict(rec) for rec in ood_rows if rec.get("modality") == "both"]
        perturb_rows = _build_perturb_rows_from_both(both_rows, rng)

        _write_jsonl(output_path_ood, ood_rows)
        _write_jsonl(output_path_perturb, perturb_rows)

        print(f"Saved {len(ood_rows)} records to {output_path_ood}")
        print(f"Saved {len(perturb_rows)} records to {output_path_perturb}")
        _print_stats(ood_rows, "OOD")
        _print_stats(perturb_rows, "OOD PERTURB")
    except Exception as exc:
        print(f"Error in build_redeval_ood: {exc}")
        raise


def main() -> None:
    build_redeval_ood()


if __name__ == "__main__":
    main()
