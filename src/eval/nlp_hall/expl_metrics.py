
import math

from config.logistics import ExplMetrics, HallEvalConfig, ParsedRecord
from src.eval.nlp_hall.claims import extract_claims
from src.eval.nlp_hall.normalize import normalize_list
from src.eval.nlp_hall.similarity import SimilarityBackend


def _compute_expl_metrics_from_refs(
    ref_texts: list[str],
    explanation: str,
    sim_backend: SimilarityBackend,
    cfg: HallEvalConfig,
) -> ExplMetrics:
    ref = normalize_list(ref_texts)
    claims = extract_claims(explanation=explanation or "", cfg=cfg)

    k_claims = len(claims)
    n_ref = len(ref)
    len_tokens = len((explanation or "").split())

    if k_claims > 0 and n_ref > 0:
        sim = sim_backend.similarity_matrix(claims, ref)
        claim_max = sim.max(axis=1)
        fact_max = sim.max(axis=0)
        grounded = int((claim_max >= cfg.tau_e).sum())
        covered = int((fact_max >= cfg.tau_c).sum())
    else:
        grounded = k_claims
        covered = n_ref

    eg = (grounded / k_claims) if k_claims > 0 else 1.0
    eh = 1.0 - eg
    ec = (covered / n_ref) if n_ref > 0 else 1.0

    if len_tokens <= 0:
        eqs = 0.0
    else:
        length_term = math.log(1.0 + float(len_tokens))
        eqs = (eg * ec / length_term) if length_term > 0 else 0.0

    return ExplMetrics(
        eg=float(eg),
        eh=float(eh),
        ec=float(ec),
        eqs=float(eqs),
        k_claims=int(k_claims),
        len_tokens=int(len_tokens),
    )


def compute_expl_metrics(
    ref_facts: list[str],
    explanation: str,
    sim_backend: SimilarityBackend,
    cfg: HallEvalConfig,
) -> ExplMetrics:
    # Backward-compatible helper: visual-fact grounding only.
    return _compute_expl_metrics_from_refs(
        ref_texts=ref_facts,
        explanation=explanation,
        sim_backend=sim_backend,
        cfg=cfg,
    )


def _is_sarcasm_reasoning_case(parsed: ParsedRecord) -> bool:
    gt = (parsed.gt or "").strip().lower()
    if gt in {"sarcastic", "non_sarcastic"}:
        return True
    return bool(parsed.need_explanation)


def _routed_expl_refs(parsed: ParsedRecord) -> list[str]:
    missing = {m.strip().lower() for m in parsed.missing_modalities_ref}

    # Missing modality cases: use target/reference explanation only (no incongruity).
    if "text" in missing or "image" in missing:
        return [parsed.ref_explanation]

    # Sarcasm reasoning cases: use explanation + incongruity, no visual facts.
    if _is_sarcasm_reasoning_case(parsed):
        return [parsed.ref_explanation, parsed.ref_incongruity]

    # Fallback for non-sarcasm/unknown cases: prefer available text references only.
    refs: list[str] = []
    if parsed.ref_explanation:
        refs.append(parsed.ref_explanation)
    if parsed.ref_incongruity:
        refs.append(parsed.ref_incongruity)
    if parsed.ref_text_literal:
        refs.append(parsed.ref_text_literal)
    return refs


def compute_expl_metrics_routed(
    parsed: ParsedRecord,
    sim_backend: SimilarityBackend,
    cfg: HallEvalConfig,
) -> ExplMetrics:
    ref_texts = _routed_expl_refs(parsed)
    return _compute_expl_metrics_from_refs(
        ref_texts=ref_texts,
        explanation=parsed.explanation,
        sim_backend=sim_backend,
        cfg=cfg,
    )
