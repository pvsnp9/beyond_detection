import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from config.logistics import Logistics

VALID_LABELS = {"sarcastic", "non_sarcastic", "unknown"}
GEN_TYPES = ("rich_freeform_sft", "freeform_sft", "sft", "mdpo", "dpo_random", "dpo")
FNAME_RE = re.compile(
    r"^(?P<gen>rich_freeform_sft|freeform_sft|sft|mdpo|dpo_random|dpo)_(?P<split>iid|ood)_(?P<ts>\d{8}_\d{6})\.jsonl$"
)
# minimal keys/types the eval pipelines consume downstream
REQUIRED_OUTPUT_KEYS = {"label": str, "explanation": str, "missing_modalities": list, "visual_facts": list}


def results_root() -> Path:
    lg = Logistics()
    return Path(lg.project_root_dir) / lg.results_dir


def reports_root() -> Path:
    lg = Logistics()
    return Path(lg.project_root_dir) / lg.reports_dir


def discover_latest_results(root: Optional[Path] = None) -> Dict[Tuple[str, str, str], Path]:
    # (model, gen_type, split) -> latest results file, by timestamp in the filename
    root = Path(root) if root is not None else results_root()
    if not root.is_dir():
        raise FileNotFoundError(f"results root not found: {root}")
    latest: Dict[Tuple[str, str, str], Tuple[str, Path]] = {}
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in model_dir.glob("*.jsonl"):
            m = FNAME_RE.match(path.name)
            if not m:
                continue
            key = (model_dir.name, m["gen"], m["split"])
            if key not in latest or m["ts"] > latest[key][0]:
                latest[key] = (m["ts"], path)
    return {key: entry[1] for key, entry in sorted(latest.items())}


def source_ts(path: Path) -> str:
    m = FNAME_RE.match(Path(path).name)
    if not m:
        raise ValueError(f"unrecognized results filename: {path}")
    return m["ts"]


def iter_records(path: Path) -> Iterator[Dict[str, Any]]:
    # a corrupt machine-written line is a pipeline bug: fail fast
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"corrupt results line {path}:{lineno}") from e


def parse_structured_output(raw: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    # status: ok | parsing_failed (not valid JSON object) | bad_format (schema/label violation)
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return None, "parsing_failed"
    if not isinstance(payload, dict):
        return None, "parsing_failed"
    for key, typ in REQUIRED_OUTPUT_KEYS.items():
        if not isinstance(payload.get(key), typ):
            return None, "bad_format"
    if payload["label"] not in VALID_LABELS:
        return None, "bad_format"
    return payload, "ok"


def parse_freeform_output(raw: Any) -> Tuple[Optional[str], str, str]:
    # freeform convention: first line = label, remainder = explanation
    if not isinstance(raw, str) or not raw.strip():
        return None, "", "parsing_failed"
    first, _, rest = raw.strip().partition("\n")
    label = first.strip().lower()
    if label not in VALID_LABELS:
        return None, rest.strip(), "bad_format"
    return label, rest.strip(), "ok"


RICH_ANCHORS = ("NEED_EXPLANATION", "FACTS", "EVIDENCE", "LITERAL", "INCONGRUITY", "LABEL", "EXPLANATION", "MISSING")
_RICH_ANCHOR_RE = re.compile(rf"^({'|'.join(RICH_ANCHORS)}):\s*(.*)$")
_RICH_FACT_RE = re.compile(r"^(\d+)\.\s+(.*)$")


def parse_rich_freeform_output(raw: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """Parse the rich_freeform keyword-section format back into a target_json-shaped
    payload. Lenient on missing sections (model drift) but exact on rendered text;
    status: ok | parsing_failed (no LABEL line) | bad_format (invalid LABEL value)."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "parsing_failed"

    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in raw.strip().splitlines():
        m = _RICH_ANCHOR_RE.match(line.strip() if current != "EXPLANATION" else line)
        # inside EXPLANATION only the MISSING anchor terminates the section
        if m and (current != "EXPLANATION" or m.group(1) == "MISSING"):
            current = m.group(1)
            sections[current] = [m.group(2)] if m.group(2) else []
        elif current is not None:
            sections[current].append(line)

    if "LABEL" not in sections:
        return None, "parsing_failed"
    label = " ".join(sections["LABEL"]).strip().lower()
    if label not in VALID_LABELS:
        return None, "bad_format"

    def _joined(key: str) -> str:
        text = "\n".join(sections.get(key, [])).strip()
        return "" if text.lower() == "none" else text

    facts = []
    for line in sections.get("FACTS", []):
        fm = _RICH_FACT_RE.match(line.strip())
        if fm:
            facts.append({"id": int(fm.group(1)), "fact": fm.group(2).strip()})
    evidence = [int(tok) for tok in re.findall(r"\d+", _joined("EVIDENCE"))]
    missing = [tok.strip() for tok in _joined("MISSING").split(",") if tok.strip()]

    payload = {
        "need_explanation": _joined("NEED_EXPLANATION").lower() != "false",
        "visual_facts": facts,
        "evidence_fact_ids": evidence,
        "text_literal": _joined("LITERAL"),
        "incongruity": _joined("INCONGRUITY"),
        "label": label,
        "explanation": _joined("EXPLANATION"),
        "missing_modalities": missing,
    }
    return payload, "ok"


def extract_facts(items: Any) -> List[str]:
    facts: List[str] = []
    for item in items or []:
        if isinstance(item, dict):
            fact = item.get("fact")
            if isinstance(fact, str) and fact.strip():
                facts.append(fact.strip())
        elif isinstance(item, str) and item.strip():
            facts.append(item.strip())
    return facts
