from config.logistics import ExplMetrics, HallEvalConfig
from src.eval.nlp_hall.claims import extract_claims
from src.eval.nlp_hall.normalize import normalize_list
from src.eval.nlp_hall.similarity import SimilarityBackend


def compute_expl_metrics(
    ref_texts: list[str],
    explanation: str,
    sim_backend: SimilarityBackend,
    cfg: HallEvalConfig,
) -> ExplMetrics:
    ref = normalize_list(ref_texts)
    claims = extract_claims(explanation=explanation or "", cfg=cfg)

    k_claims = len(claims)
    n_ref = len(ref)

    if k_claims > 0 and n_ref > 0:
        sim = sim_backend.similarity_matrix(claims, ref)
        grounded = int((sim.max(axis=1) >= cfg.tau_e).sum())
        covered = int((sim.max(axis=0) >= cfg.tau_c).sum())
    else:
        grounded = 0
        covered = 0

    # paper edges: EG=1 iff no claims; EC=1 iff no references (K=0 covers nothing)
    eg = (grounded / k_claims) if k_claims > 0 else 1.0
    ec = (covered / n_ref) if n_ref > 0 else 1.0

    return ExplMetrics(eg=float(eg), eh=float(1.0 - eg), ec=float(ec), k_claims=int(k_claims))
