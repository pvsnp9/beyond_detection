
import numpy as np

from config.logistics import FactCounts, FactMetrics, HallEvalConfig
from src.eval.nlp_hall.matching import one_to_one_match
from src.eval.nlp_hall.normalize import normalize_list
from src.eval.nlp_hall.similarity import SimilarityBackend


def compute_fact_counts(
    ref_facts: list[str],
    pred_facts: list[str],
    sim_backend: SimilarityBackend,
    cfg: HallEvalConfig,
) -> FactCounts:
    ref = normalize_list(ref_facts)
    pred = normalize_list(pred_facts)
    n_ref = len(ref)
    n_pred = len(pred)

    if n_ref == 0 and n_pred == 0:
        return FactCounts(tp=0, fp=0, fn=0, n_ref=0, n_pred=0, mean_match_sim=None)
    if n_ref == 0:
        return FactCounts(tp=0, fp=n_pred, fn=0, n_ref=0, n_pred=n_pred, mean_match_sim=None)
    if n_pred == 0:
        return FactCounts(tp=0, fp=0, fn=n_ref, n_ref=n_ref, n_pred=0, mean_match_sim=None)

    sim = sim_backend.similarity_matrix(pred, ref)
    matches = one_to_one_match(
        S=sim,
        tau=cfg.tau,
        mode=cfg.match_mode,
        top_k=cfg.top_k,
    )

    tp = len(matches)
    fp = n_pred - tp
    fn = n_ref - tp

    mean_match_sim: float | None = None
    if matches:
        values = [float(sim[i, j]) for i, j in matches]
        mean_match_sim = float(np.mean(values, dtype=np.float32))

    return FactCounts(
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        n_ref=int(n_ref),
        n_pred=int(n_pred),
        mean_match_sim=mean_match_sim,
    )


def compute_fact_metrics(counts: FactCounts) -> FactMetrics:
    tp = counts.tp
    fp = counts.fp
    fn = counts.fn
    n_pred = counts.n_pred
    n_ref = counts.n_ref

    add = (fp / n_pred) if n_pred > 0 else 0.0
    omit = (fn / n_ref) if n_ref > 0 else 0.0

    p = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    r = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    return FactMetrics(add=float(add), omit=float(omit), p=float(p), r=float(r), f1=float(f1))


def build_fact_match_debug(
    ref_facts: list[str],
    pred_facts: list[str],
    sim_backend: SimilarityBackend,
    cfg: HallEvalConfig,
    candidate_k: int = 3,
) -> dict:
    ref = normalize_list(ref_facts)
    pred = normalize_list(pred_facts)
    n_ref = len(ref)
    n_pred = len(pred)

    debug: dict = {
        "tau": float(cfg.tau),
        "match_mode": str(cfg.match_mode),
        "top_k": cfg.top_k,
        "n_ref": int(n_ref),
        "n_pred": int(n_pred),
        "ref_facts_norm": ref,
        "pred_facts_norm": pred,
        "matches": [],
        "pred_top_candidates": [],
    }

    if n_ref == 0 or n_pred == 0:
        return debug

    sim = sim_backend.similarity_matrix(pred, ref)
    matches = one_to_one_match(S=sim, tau=cfg.tau, mode=cfg.match_mode, top_k=cfg.top_k)
    matched_pairs = {(int(i), int(j)) for i, j in matches}
    matched_pred = {int(i) for i, _ in matches}
    matched_ref = {int(j) for _, j in matches}

    debug["matches"] = [
        {
            "pred_idx": int(i),
            "ref_idx": int(j),
            "sim": float(sim[i, j]),
            "pred_fact": pred[i],
            "ref_fact": ref[j],
        }
        for i, j in sorted(matches, key=lambda x: (x[0], x[1]))
    ]

    k = max(1, int(candidate_k))
    for i in range(n_pred):
        order = np.argsort(sim[i])[::-1]
        top_cols = order[: min(k, n_ref)]
        candidates = []
        for j in top_cols.tolist():
            candidates.append(
                {
                    "ref_idx": int(j),
                    "sim": float(sim[i, j]),
                    "above_tau": bool(sim[i, j] >= float(cfg.tau)),
                    "is_selected_match": (i, int(j)) in matched_pairs,
                    "ref_fact": ref[int(j)],
                }
            )
        debug["pred_top_candidates"].append(
            {
                "pred_idx": int(i),
                "pred_fact": pred[i],
                "matched": int(i) in matched_pred,
                "top_candidates": candidates,
            }
        )

    debug["unmatched_pred_indices"] = sorted(i for i in range(n_pred) if i not in matched_pred)
    debug["unmatched_ref_indices"] = sorted(j for j in range(n_ref) if j not in matched_ref)
    return debug
