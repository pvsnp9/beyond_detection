import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config.logistics import HallEvalConfig

logger = logging.getLogger(__name__)


def _silence_broken_stdout() -> None:
    """Redirect stdout to /dev/null after a broken pipe to avoid shutdown noise."""
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, sys.stdout.fileno())
        finally:
            os.close(devnull_fd)
    except Exception:
        pass


def _resolve_from_root(project_root_dir: str, path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(project_root_dir) / path


def _pick_all_jsonl(model_files: list[str]) -> str | None:
    for path in model_files:
        if Path(path).name == "all.jsonl":
            return path
    return None


def _serialize_flags(flags: tuple[str, ...]) -> str:
    return "|".join(flags)


def _base_row(parsed: Any) -> dict[str, Any]:
    mismatch = set(parsed.missing_modalities_pred) != set(parsed.missing_modalities_ref)
    return {
        "id": parsed.id,
        "model_key": parsed.model_key,
        "source": parsed.source,
        "modality": parsed.modality,
        "gt": parsed.gt,
        "quality_flags": _serialize_flags(parsed.quality_flags),
        "missing_modalities_pred": _serialize_flags(parsed.missing_modalities_pred),
        "missing_modalities_ref": _serialize_flags(parsed.missing_modalities_ref),
        "missing_modality_mismatch": int(mismatch),
        "is_valid": int(bool(parsed.is_valid)),
        "error": parsed.error,
    }


def _fill_null_metrics(row: dict[str, Any]) -> dict[str, Any]:
    row.update(
        {
            "tp": None,
            "fp": None,
            "fn": None,
            "n_ref": None,
            "n_pred": None,
            "mean_match_sim": None,
            "add": None,
            "omit": None,
            "p": None,
            "r": None,
            "f1": None,
            "eg": None,
            "eh": None,
            "ec": None,
            "eqs": None,
            "k_claims": None,
            "len_tokens": None,
        }
    )
    return row


def _safe_to_jsonable(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    if isinstance(value, int):
        return value
    if value is None:
        return None
    return value


def build_parser() -> argparse.ArgumentParser:
    defaults = HallEvalConfig()
    parser = argparse.ArgumentParser(description="Hallucination + explanation-quality evaluation")
    parser.add_argument("--input", required=False, help="Input JSONL path (single-file mode)")
    parser.add_argument("--output_dir", required=False, help="Output directory (single-file mode)")
    parser.add_argument(
        "--eval_model",
        type=str,
        default=None,
        help="Auto mode only: evaluate one model directory from latest results",
    )
    parser.add_argument("--tau", type=float, default=defaults.tau)
    parser.add_argument("--tau_e", type=float, default=defaults.tau_e)
    parser.add_argument("--tau_c", type=float, default=defaults.tau_c)
    parser.add_argument("--backend", type=str, default=defaults.backend)
    parser.add_argument(
        "--embed_model_name",
        "--model_name",
        dest="embed_model_name",
        type=str,
        default=defaults.model_name,
        help="Embedding backend model name (not eval model key; use --eval_model for aya/llama/gemma/qwen)",
    )
    parser.add_argument("--match_mode", type=str, default=defaults.match_mode)
    parser.add_argument("--top_k", type=int, default=defaults.top_k)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--skip_invalid", action="store_true", default=defaults.skip_invalid)
    parser.add_argument(
        "--debug_matches",
        action="store_true",
        help="Write per-example fact matching diagnostics (matched pairs + top candidates)",
    )
    parser.add_argument(
        "--debug_match_limit",
        type=int,
        default=200,
        help="Max number of examples to write into match_debug.jsonl per run",
    )
    parser.add_argument(
        "--debug_candidate_k",
        type=int,
        default=3,
        help="Top similarity candidates per predicted fact to include in debug output",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _build_cfg(args: argparse.Namespace) -> HallEvalConfig:
    base = HallEvalConfig()
    return HallEvalConfig(
        tau=args.tau,
        tau_e=args.tau_e,
        tau_c=args.tau_c,
        backend=args.backend,
        model_name=getattr(args, "embed_model_name", getattr(args, "model_name", None)),
        batch_size=base.batch_size,
        device=base.device,
        match_mode=args.match_mode,
        top_k=args.top_k,
        max_claims=base.max_claims,
        min_clause_split_len=base.min_clause_split_len,
        skip_invalid=args.skip_invalid,
        seed=base.seed,
    )


def _run_latest_results(args: argparse.Namespace, cfg: HallEvalConfig) -> int:
    try:
        from src.eval.classification import get_latest_result_files
    except ImportError as exc:
        logger.error("Could not import get_latest_result_files: %s", exc)
        return 2

    logistics = cfg.logistics
    results_root = _resolve_from_root(logistics.project_root_dir, logistics.results_dir)
    hall_reports_root = _resolve_from_root(logistics.project_root_dir, logistics.hall_report_dir)

    latest_files = get_latest_result_files(results_root)
    if not latest_files:
        logger.error("No result files found under: %s", results_root)
        return 2

    selected_items = sorted(latest_files.items())
    if args.eval_model:
        selected_items = [(m, fs) for m, fs in selected_items if m == args.eval_model]
        if not selected_items:
            logger.error("Model not found in latest results: %s", args.eval_model)
            return 2

    exit_codes: list[int] = []
    ran_any = False
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for eval_model_name, model_files in selected_items:
        eval_path = _pick_all_jsonl(model_files)
        if eval_path is None:
            logger.warning("Skipping %s: all.jsonl not found", eval_model_name)
            continue

        out_dir = hall_reports_root / eval_model_name / run_timestamp
        out_dir.mkdir(parents=True, exist_ok=True)
        ran_any = True

        logger.info("Running hall eval for model=%s", eval_model_name)
        logger.info("input=%s output_dir=%s", eval_path, out_dir)

        run_args = argparse.Namespace(**vars(args))
        run_args.input = str(eval_path)
        run_args.output_dir = str(out_dir)
        run_args.model_name = getattr(
            run_args, "embed_model_name", getattr(run_args, "model_name", None)
        )

        try:
            exit_codes.append(run(run_args))
        except Exception:  # noqa: BLE001
            logger.exception("Hall eval wrapper failed for model=%s", eval_model_name)
            exit_codes.append(1)

    if not ran_any:
        logger.error("No models had all.jsonl in latest result directories")
        return 2

    return 0 if all(code == 0 for code in exit_codes) else 1


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    cfg = _build_cfg(args)
    random.seed(cfg.seed)

    try:
        import numpy as np

        from src.eval.nlp_hall.aggregate import aggregate_all, records_to_dataframe
        from src.eval.nlp_hall.expl_metrics import compute_expl_metrics_routed
        from src.eval.nlp_hall.fact_metrics import (
            build_fact_match_debug,
            compute_fact_counts,
            compute_fact_metrics,
        )
        from src.eval.nlp_hall.io import parse_record
        from src.eval.nlp_hall.similarity import get_similarity_backend
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        return 2

    np.random.seed(cfg.seed)

    if not hasattr(args, "model_name"):
        args.model_name = getattr(args, "embed_model_name", None)

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error("Input file does not exist: %s", in_path)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_backend = get_similarity_backend(cfg)

    rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    total = 0
    parse_failed = 0

    with in_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            if args.max_records is not None and total >= args.max_records:
                break
            total += 1

            try:
                raw = json.loads(text)
            except Exception as exc:  # noqa: BLE001
                parse_failed += 1
                row = _fill_null_metrics(
                    {
                        "id": str(lineno),
                        "model_key": None,
                        "source": None,
                        "modality": None,
                        "gt": None,
                        "quality_flags": "",
                        "missing_modalities_pred": "",
                        "missing_modalities_ref": "",
                        "missing_modality_mismatch": 0,
                        "is_valid": 0,
                        "error": f"jsonl parse error at line {lineno}: {exc}",
                    }
                )
                if not cfg.skip_invalid:
                    rows.append(row)
                continue

            if not isinstance(raw, dict):
                parse_failed += 1
                row = _fill_null_metrics(
                    {
                        "id": str(lineno),
                        "model_key": None,
                        "source": None,
                        "modality": None,
                        "gt": None,
                        "quality_flags": "",
                        "missing_modalities_pred": "",
                        "missing_modalities_ref": "",
                        "missing_modality_mismatch": 0,
                        "is_valid": 0,
                        "error": f"jsonl parse error at line {lineno}: expected object",
                    }
                )
                if not cfg.skip_invalid:
                    rows.append(row)
                continue

            try:
                parsed = parse_record(raw)
            except Exception as exc:  # noqa: BLE001
                parse_failed += 1
                logger.exception("parse_record failed at line=%s", lineno)
                row = _fill_null_metrics(
                    {
                        "id": str(raw.get("id", lineno)),
                        "model_key": raw.get("model_key"),
                        "source": raw.get("source"),
                        "modality": raw.get("modality"),
                        "gt": raw.get("gt"),
                        "quality_flags": "",
                        "missing_modalities_pred": "",
                        "missing_modalities_ref": "",
                        "missing_modality_mismatch": 0,
                        "is_valid": 0,
                        "error": f"parse_record exception: {exc}",
                    }
                )
                if not cfg.skip_invalid:
                    rows.append(row)
                continue

            base = _base_row(parsed)
            if not parsed.is_valid:
                parse_failed += 1
                if not cfg.skip_invalid:
                    rows.append(_fill_null_metrics(base))
                continue

            try:
                counts = compute_fact_counts(
                    ref_facts=parsed.ref_facts,
                    pred_facts=parsed.pred_facts,
                    sim_backend=sim_backend,
                    cfg=cfg,
                )
                fact = compute_fact_metrics(counts)
                expl = compute_expl_metrics_routed(
                    parsed=parsed,
                    sim_backend=sim_backend,
                    cfg=cfg,
                )
                if args.debug_matches and len(debug_rows) < max(0, int(args.debug_match_limit)):
                    debug_rows.append(
                        {
                            "id": parsed.id,
                            "model_key": parsed.model_key,
                            "source": parsed.source,
                            "modality": parsed.modality,
                            "quality_flags": list(parsed.quality_flags),
                            "missing_modalities_pred": list(parsed.missing_modalities_pred),
                            "missing_modalities_ref": list(parsed.missing_modalities_ref),
                            "fact_debug": build_fact_match_debug(
                                ref_facts=parsed.ref_facts,
                                pred_facts=parsed.pred_facts,
                                sim_backend=sim_backend,
                                cfg=cfg,
                                candidate_k=int(args.debug_candidate_k),
                            ),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("evaluation error for id=%s", parsed.id)
                base["is_valid"] = 0
                base["error"] = f"evaluation error: {exc}"
                if not cfg.skip_invalid:
                    rows.append(_fill_null_metrics(base))
                continue

            row = {
                **base,
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
                "n_ref": counts.n_ref,
                "n_pred": counts.n_pred,
                "mean_match_sim": counts.mean_match_sim,
                "add": fact.add,
                "omit": fact.omit,
                "p": fact.p,
                "r": fact.r,
                "f1": fact.f1,
                "eg": expl.eg,
                "eh": expl.eh,
                "ec": expl.ec,
                "eqs": expl.eqs,
                "k_claims": expl.k_claims,
                "len_tokens": expl.len_tokens,
            }
            rows.append(row)

    if not rows:
        logger.error("No rows produced from input")
        return 2

    df = records_to_dataframe(rows)
    valid_count = int(df["is_valid"].astype(bool).sum()) if "is_valid" in df.columns else len(df)

    if valid_count == 0:
        logger.error("No valid rows to aggregate")
        return 2

    df.to_csv(out_dir / "per_example.csv", index=False)

    try:
        summaries = aggregate_all(df)
    except ImportError as exc:
        logger.error("Aggregation requires pandas: %s", exc)
        return 2

    overall = summaries.get("overall")
    overall_dict: dict[str, Any] = {}
    if overall is not None and not overall.empty:
        overall_dict = {k: _safe_to_jsonable(v) for k, v in overall.iloc[0].to_dict().items()}
    overall_dict["total"] = int(total)
    overall_dict["valid"] = int(valid_count)
    overall_dict["parse_failed"] = int(parse_failed)

    with (out_dir / "summary_overall.json").open("w", encoding="utf-8") as f:
        json.dump(overall_dict, f, ensure_ascii=False, indent=2)

    if "by_model" in summaries:
        summaries["by_model"].to_csv(out_dir / "by_model.csv", index=False)
    if "by_source" in summaries:
        summaries["by_source"].to_csv(out_dir / "by_source.csv", index=False)
    if "by_modality" in summaries:
        summaries["by_modality"].to_csv(out_dir / "by_modality.csv", index=False)
    if "by_model_source_modality" in summaries:
        summaries["by_model_source_modality"].to_csv(out_dir / "by_model_source_modality.csv", index=False)

    by_q = summaries.get("by_quality_flag")
    if by_q is not None and not by_q.empty:
        by_q.to_csv(out_dir / "by_quality_flag.csv", index=False)

    if args.debug_matches and debug_rows:
        debug_path = out_dir / "match_debug.jsonl"
        with debug_path.open("w", encoding="utf-8") as f:
            for item in debug_rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Wrote match debug diagnostics: %s (%d rows)", debug_path, len(debug_rows))

    try:
        print(f"total={total} valid={valid_count} parse_failed={parse_failed}", flush=True)
    except BrokenPipeError:
        _silence_broken_stdout()
        logger.debug("stdout closed before final summary print")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = _build_cfg(args)

    if bool(args.input) ^ bool(args.output_dir):
        parser.error("--input and --output_dir must be provided together in single-file mode")

    if args.input and args.output_dir:
        return run(args)

    return _run_latest_results(args, cfg)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        _silence_broken_stdout()
        raise SystemExit(0)
