
import logging
import tempfile
from pathlib import Path
from typing import Dict

from src.data_generation import iterate_and_curate_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_output_dirs(root: Path, suffix: str = "") -> Dict[str, Path]:
    """Return per-dataset output roots under a temp directory."""
    suffix_part = f"-{suffix}" if suffix else ""
    return {
        "alita9/muse-sarcasm-explanation": root / f"muse{suffix_part}",
        "alita9/sarcnet": root / f"sarcnet{suffix_part}",
        "coderchen01/MMSD2.0": root / f"mmsd2{suffix_part}",
    }


def _run_once(root: Path, splits: list[str], lang: str) -> None:
    """Run curation for one example per dataset/lang across splits."""
    out_dirs = _make_output_dirs(root, suffix=lang)
    try:
        iterate_and_curate_datasets(
            output_dirs=out_dirs,
            split=splits,
            lang=lang,
            streaming=True,
            limit=1,
        )
    except Exception:
        logger.exception("Curation run failed", extra={"splits": splits, "lang": lang})


def _list_created(root: Path) -> None:
    files = sorted(root.rglob("*.jsonl"))
    if not files:
        print("No jsonl files created.")
        return
    print("Created files:")
    for fp in files:
        print(f"- {fp}")


def main() -> None:
    tmp_root = Path(tempfile.mkdtemp(prefix="gen_test_"))
    print(f"Using temp output root: {tmp_root}")

    splits = ["train", "validation", "test"]
    langs = ["en", "zh"]

    # muse and mmsd use only the lang passed through, but sarcnet cares about it.
    for lang in langs:
        _run_once(tmp_root, splits=splits, lang=lang)

    _list_created(tmp_root)


if __name__ == "__main__":
    main()
