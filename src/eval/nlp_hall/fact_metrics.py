import numpy as np

from config.logistics import FactCounts, HallEvalConfig
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
