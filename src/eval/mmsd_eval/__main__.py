import argparse
import sys

from src.eval.mmsd_eval.runner import METRIC_KEYS, run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval.mmsd_eval",
        description="Unified MMSD evaluation: classification, reliability (PSR/MAR), BERTScore, FaRGE.",
    )
    parser.add_argument("--input", required=True, help="path to model-output JSONL")
    parser.add_argument(
        "--out",
        default=None,
        help="output dir (defaults to outputs/reports/mmsd/<model>/<type>/<run_mode>/ parsed from --input)",
    )
    parser.add_argument(
        "--skip",
        default="",
        help=f"comma-separated metric families to skip. Valid: {sorted(METRIC_KEYS)}",
    )
    parser.add_argument(
        "--similarity-backend",
        default="sbert",
        choices=["sbert", "tfidf"],
        help="similarity backend for FaRGE fact matching (default: sbert)",
    )
    args = parser.parse_args(argv)

    skip = [s.strip() for s in args.skip.split(",") if s.strip()]
    run(
        input_jsonl=args.input,
        out_dir=args.out,
        skip=skip,
        similarity_backend=args.similarity_backend,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
