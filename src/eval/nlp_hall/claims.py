
import re

from config.logistics import HallEvalConfig
from src.eval.nlp_hall.normalize import normalize_text

_SENT_SPLIT_RE = re.compile(r"[.!?]+")
_CONJ_SPLIT_RE = re.compile(
    r"\b(?:and|but|or|because|while|although|though|however|yet|so)\b",
    flags=re.IGNORECASE,
)


def extract_claims(explanation: str, cfg: HallEvalConfig) -> list[str]:
    if not isinstance(explanation, str):
        return []

    text = explanation.strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]

    claims: list[str] = []
    for sent in sentences:
        parts: list[str] = [sent]
        if len(sent) > int(cfg.min_clause_split_len):
            pieces = [p.strip(" ,;:\")") for p in _CONJ_SPLIT_RE.split(sent)]
            pieces = [p for p in pieces if p]
            if pieces:
                parts = pieces

        for part in parts:
            normalized = normalize_text(part)
            if normalized:
                claims.append(normalized)
                if len(claims) >= int(cfg.max_claims):
                    return claims

    return claims
