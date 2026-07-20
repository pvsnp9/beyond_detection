import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from config.logistics import HallEvalConfig
from src.eval.nlp_hall.expl_metrics import compute_expl_metrics
from src.eval.nlp_hall.fact_metrics import compute_fact_counts
from src.eval.nlp_hall.similarity import get_similarity_backend
from src.eval.parsing import (
    discover_latest_results,
    extract_facts,
    iter_records,
    parse_freeform_output,
    parse_rich_freeform_output,
    parse_structured_output,
    reports_root,
    results_root,
    source_ts,
)

FARGE_MODELS = ("qwen", "gemma", "llama", "aya")
FARGE_GEN_TYPES = ("sft", "freeform_sft", "rich_freeform_sft", "mdpo", "dpo", "dpo_random")
BERTSCORE_MODEL = "roberta-large"
FARGE_KEYS = ("add", "omit", "fact_f1", "eg", "eh", "ec")
# full penalty for outputs that fail schema parsing (paper: zero credit)
ZERO_CREDIT = {
    "add": 1.0, "omit": 1.0, "fact_f1": 0.0, "eg": 0.0, "eh": 1.0, "ec": 0.0,
    "tp": 0, "fp": 0, "fn": 0, "n_ref": None, "n_pred": None, "k_claims": None,
}
# freeform has no visual_facts inventory: fact metrics stay N/A even on failure
ZERO_CREDIT_FREEFORM = {
    "add": None, "omit": None, "fact_f1": None, "eg": 0.0, "eh": 1.0, "ec": 0.0,
    "tp": None, "fp": None, "fn": None, "n_ref": None, "n_pred": None, "k_claims": None,
}


def _expl_refs(target: Dict[str, Any]) -> list:
    # both-modality reference set: gold explanation + gold incongruity (empties dropped)
    return [t for t in (target.get("explanation"), target.get("incongruity")) if isinstance(t, str) and t.strip()]


def farge_example(target: Dict[str, Any], payload: Dict[str, Any], sim, cfg: HallEvalConfig) -> Dict[str, Any]:
    counts = compute_fact_counts(
        ref_facts=extract_facts(target.get("visual_facts")),
        pred_facts=extract_facts(payload["visual_facts"]),
        sim_backend=sim,
        cfg=cfg,
    )
    # paper formulas: Add=FP/|pred| (0 if none), Omit=FN/|ref| (0 if none), FactF1=2TP/max(1, 2TP+FP+FN)
    add = counts.fp / counts.n_pred if counts.n_pred else 0.0
    omit = counts.fn / counts.n_ref if counts.n_ref else 0.0
    fact_f1 = 2 * counts.tp / max(1, 2 * counts.tp + counts.fp + counts.fn)

    expl = compute_expl_metrics(ref_texts=_expl_refs(target), explanation=payload["explanation"], sim_backend=sim, cfg=cfg)

    return {
        "add": add, "omit": omit, "fact_f1": fact_f1,
        "eg": expl.eg, "eh": expl.eh, "ec": expl.ec,
        "tp": counts.tp, "fp": counts.fp, "fn": counts.fn,
        "n_ref": counts.n_ref, "n_pred": counts.n_pred, "k_claims": expl.k_claims,
    }


def freeform_example(target: Dict[str, Any], explanation: str, sim, cfg: HallEvalConfig) -> Dict[str, Any]:
    expl = compute_expl_metrics(ref_texts=_expl_refs(target), explanation=explanation, sim_backend=sim, cfg=cfg)
    return {
        "add": None, "omit": None, "fact_f1": None,
        "eg": expl.eg, "eh": expl.eh, "ec": expl.ec,
        "tp": None, "fp": None, "fn": None,
        "n_ref": None, "n_pred": None, "k_claims": expl.k_claims,
    }


def evaluate_file(
    path: Path,
    model: str,
    gen_type: str,
    split: str,
    sim,
    scorer,
    cfg: HallEvalConfig,
    limit: Optional[int],
) -> Dict[str, Any]:
    freeform = gen_type == "freeform_sft"
    total = 0
    modality_count = {"both": 0, "image": 0, "text": 0}
    parsing_failed = 0
    bad_format = 0
    entries: list[Dict[str, Any]] = []
    bert_pairs: list[tuple[int, str, str]] = []  # (entry index, candidate, reference)

    for rec in iter_records(path):
        total += 1
        modality_count[rec["modality"]] += 1
        if rec["modality"] != "both":
            continue
        if limit is not None and len(entries) >= limit:
            continue

        target = json.loads(rec["target_json"])  # machine-written: must parse
        if freeform:
            pred_label, expl_text, status = parse_freeform_output(rec["output"])
        elif gen_type == "rich_freeform_sft":
            # rich recovers the full schema from plain text: real fact metrics
            payload, status = parse_rich_freeform_output(rec["output"])
            pred_label = payload["label"] if status == "ok" else None
        else:
            payload, status = parse_structured_output(rec["output"])
            pred_label = payload["label"] if status == "ok" else None
        if status == "parsing_failed":
            parsing_failed += 1
        elif status == "bad_format":
            bad_format += 1

        entry = {
            "id": rec["id"],
            "source": rec["source"],
            "modality": rec["modality"],
            "gt": rec["gt"],
            "pred_label": pred_label,
            "quality_flags": rec["quality_flags"],
            "parse_status": status,
            "farge": None,
            "bertscore": None,
            "file_source": str(path),
        }
        if status == "ok":
            if freeform:
                entry["farge"] = freeform_example(target, expl_text, sim, cfg)
                candidate = expl_text
            else:
                entry["farge"] = farge_example(target, payload, sim, cfg)
                candidate = payload["explanation"]
            bert_pairs.append((len(entries), candidate, target["explanation"]))
        else:
            entry["farge"] = dict(ZERO_CREDIT_FREEFORM if freeform else ZERO_CREDIT)
        entries.append(entry)

    if bert_pairs:
        _, cands, refs = zip(*bert_pairs)
        p_t, r_t, f_t = scorer.score(list(cands), list(refs))
        for (idx, _, _), p, r, f in zip(bert_pairs, p_t.tolist(), r_t.tolist(), f_t.tolist()):
            entries[idx]["bertscore"] = {"p": p, "r": r, "f1": f}

    out_dir = reports_root() / "farge" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{gen_type}_{split}_{source_ts(path)}.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"farge: wrote {out_path} ({len(entries)} rows)")

    n_both = len(entries)
    n_zero_credit = sum(1 for e in entries if e["parse_status"] != "ok")
    # None-safe macro: freeform rows carry null fact metrics
    def _mean_or_none(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    farge_macro = {k: _mean_or_none([e["farge"][k] for e in entries]) for k in FARGE_KEYS}
    bert_vals = [e["bertscore"] for e in entries if e["bertscore"] is not None]
    bert_mean = (
        {k: sum(b[k] for b in bert_vals) / len(bert_vals) for k in ("p", "r", "f1")} if bert_vals else None
    )
    return {
        "model": model, "gen_type": gen_type, "split": split,
        "source_file": str(path), "per_example_file": str(out_path),
        "total_samples": total, "modality_count": modality_count, "n_both": n_both,
        "parsing_failed_count": parsing_failed, "bad_format_count": bad_format,
        "n_zero_credit": n_zero_credit,
        "farge": farge_macro, "bertscore": bert_mean, "n_bert": len(bert_vals),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="FaRGE + BERTScore evaluation over both-modality rows")
    ap.add_argument("--similarity-backend", choices=("sbert", "tfidf"), default="sbert")
    ap.add_argument("--models", nargs="+", default=list(FARGE_MODELS))
    ap.add_argument("--gen-types", nargs="+", default=list(FARGE_GEN_TYPES), choices=FARGE_GEN_TYPES)
    ap.add_argument("--splits", nargs="+", default=["iid", "ood"], choices=("iid", "ood"))
    ap.add_argument("--limit", type=int, default=None, help="cap both-modality rows per file (smoke tests)")
    args = ap.parse_args()

    files = {
        key: path
        for key, path in discover_latest_results().items()
        if key[0] in args.models and key[1] in args.gen_types and key[2] in args.splits
    }
    if not files:
        raise FileNotFoundError(f"no matching result files under {results_root()}")

    cfg = HallEvalConfig(backend=args.similarity_backend)
    sim = get_similarity_backend(cfg)  # one backend for the run: embedding cache amortizes across files
    from bert_score import BERTScorer

    scorer = BERTScorer(model_type=BERTSCORE_MODEL, lang="en", batch_size=64)

    rows = []
    for (model, gen_type, split), path in files.items():
        print(f"farge: {model}/{gen_type}/{split} <- {path}")
        rows.append(evaluate_file(path, model, gen_type, split, sim, scorer, cfg, args.limit))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "generated_at": ts,
        "results_root": str(results_root()),
        "config": {
            "tau": cfg.tau, "tau_e": cfg.tau_e, "tau_c": cfg.tau_c,
            "backend": cfg.backend, "match_mode": cfg.match_mode,
            "bertscore_model": BERTSCORE_MODEL, "limit": args.limit,
        },
        "rows": rows,
    }
    report_dir = reports_root()
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"farge_bert_{ts}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
