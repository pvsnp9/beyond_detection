
import re

_WS_RE = re.compile(r"\s+")
_DASH_RE = re.compile(r"[-‐‑‒–—]+")
_QUOTE_RE = re.compile(r"[\"'`“”‘’]")
_PUNCT_RE = re.compile(r"[()\[\]{}<>|\\/@#$%^&*_+=~,:;!?]")


def normalize_text(s: str) -> str:
    """Normalize text for robust lexical/semantic comparison."""
    if not isinstance(s, str):
        return ""
    text = s.strip().lower()
    # Normalize common typography variants that otherwise fragment TF-IDF tokens.
    text = _DASH_RE.sub(" ", text)
    text = _QUOTE_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip(" \t\n\r\f\v.")


def normalize_list(xs: list[str]) -> list[str]:
    out: list[str] = []
    for x in xs:
        y = normalize_text(x)
        if y:
            out.append(y)
    return out
