"""CPU-only tests for rich_freeform_sft (no GPU, no downloads).

Run: pytest src/test/test_rich_freeform_sft.py -q
The round-trip gate over all real rows is the mandatory pre-training check.
"""

import json
import os

import pytest

from config.logistics import build_cfg, build_freeform_cfg, build_rich_freeform_cfg
from config.queries import Queries
from src.datasets.loader import _render_rich_target
from src.eval.parsing import FNAME_RE, parse_rich_freeform_output
from src.utils.eval_sarcasm import (
    _parse_freeform_prediction,
    _parse_prediction,
    _parse_rich_freeform_prediction,
    _resolve_pred_parser,
)

FULL_PAYLOAD = {
    "need_explanation": True,
    "visual_facts": [{"id": 1, "fact": "A dog sits in the rain."}, {"id": 2, "fact": "The sky is dark grey."}],
    "evidence_fact_ids": [1, 2],
    "text_literal": "The weather is lovely.",
    "incongruity": "Caption praises weather while the image shows rain.",
    "label": "sarcastic",
    "explanation": "The caption's praise conflicts with the rainy scene.",
    "missing_modalities": [],
}

TEXT_ONLY_PAYLOAD = {
    "need_explanation": True,
    "visual_facts": [],
    "evidence_fact_ids": [],
    "text_literal": "Great day at work.",
    "incongruity": "",
    "label": "unknown",
    "explanation": "Image is missing, so I cannot compare the caption against visual context to judge cross-modal incongruity.",
    "missing_modalities": ["image"],
}


# --- renderer ---------------------------------------------------------------

def test_render_full_row():
    text = _render_rich_target(FULL_PAYLOAD)
    assert text.splitlines()[0] == "NEED_EXPLANATION: true"
    assert "FACTS:\n1. A dog sits in the rain.\n2. The sky is dark grey." in text
    assert "EVIDENCE: 1, 2" in text
    assert "LABEL: sarcastic" in text
    assert text.splitlines()[-1] == "MISSING: none"


def test_render_text_only_row():
    text = _render_rich_target(TEXT_ONLY_PAYLOAD)
    assert "FACTS: none" in text
    assert "EVIDENCE: none" in text
    assert "INCONGRUITY: none" in text
    assert "MISSING: image" in text


def test_render_asserts_fire():
    with pytest.raises(AssertionError):
        _render_rich_target({**FULL_PAYLOAD, "explanation": "line1\nline2"})
    with pytest.raises(AssertionError):
        _render_rich_target({**FULL_PAYLOAD, "text_literal": "none"})
    bad_ids = {**FULL_PAYLOAD, "visual_facts": [{"id": 2, "fact": "x"}]}
    with pytest.raises(AssertionError):
        _render_rich_target(bad_ids)


# --- parser -----------------------------------------------------------------

def test_round_trip_unit():
    for payload in (FULL_PAYLOAD, TEXT_ONLY_PAYLOAD):
        assert parse_rich_freeform_output(_render_rich_target(payload)) == (payload, "ok")


def test_parse_multiline_explanation():
    text = _render_rich_target(FULL_PAYLOAD).replace(
        "EXPLANATION: The caption's praise conflicts with the rainy scene.",
        "EXPLANATION: First line.\nSecond line.",
    )
    payload, status = parse_rich_freeform_output(text)
    assert status == "ok"
    assert payload["explanation"] == "First line.\nSecond line."
    assert payload["missing_modalities"] == []


def test_parse_failure_modes():
    assert parse_rich_freeform_output("") == (None, "parsing_failed")
    assert parse_rich_freeform_output(None) == (None, "parsing_failed")
    assert parse_rich_freeform_output("no anchors at all") == (None, "parsing_failed")
    assert parse_rich_freeform_output("LABEL: maybe") == (None, "bad_format")


def test_parse_lenient_missing_sections():
    payload, status = parse_rich_freeform_output("LABEL: sarcastic\nEXPLANATION: mismatch.")
    assert status == "ok"
    assert payload["label"] == "sarcastic"
    assert payload["visual_facts"] == [] and payload["missing_modalities"] == []
    assert payload["need_explanation"] is True


# --- round-trip gate over ALL real rows (pre-training requirement) -----------

@pytest.mark.skipif(not os.path.isdir("data/raw/hf_cache"), reason="hf cache not present")
def test_round_trip_all_real_rows():
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    for split in ("train", "validation", "test"):
        ds = load_dataset(
            "alita9/beyond_sarcasm_detection_sft", name="en", split=split, cache_dir="data/raw/hf_cache"
        )
        for row in ds.select_columns(["target_json"]):
            payload = json.loads(row["target_json"])
            assert parse_rich_freeform_output(_render_rich_target(payload)) == (payload, "ok")


# --- discovery regex ----------------------------------------------------------

def test_fname_re_rich_and_regression():
    m = FNAME_RE.match("rich_freeform_sft_iid_20260101_000000.jsonl")
    assert m and m["gen"] == "rich_freeform_sft" and m["split"] == "iid"
    m = FNAME_RE.match("freeform_sft_iid_20260101_000000.jsonl")
    assert m and m["gen"] == "freeform_sft"
    for gen in ("sft", "mdpo", "dpo", "dpo_random"):
        m = FNAME_RE.match(f"{gen}_ood_20260101_000000.jsonl")
        assert m and m["gen"] == gen and m["split"] == "ood"
    assert FNAME_RE.match("bogus_iid_20260101_000000.jsonl") is None


# --- config -------------------------------------------------------------------

def test_build_rich_freeform_cfg():
    cfg = build_rich_freeform_cfg("org/model-x")
    assert cfg["mode"] == "rich_freeform_sft"
    assert cfg["wandb"]["project"] == "rich_freeform_sft"
    assert "rich_freeform" in cfg["wandb"]["tags"]
    assert cfg["collator"]["target_key"] == "target_text_rich"
    assert cfg["collator"]["system_prompt"] == Queries().RICH_FREEFORM_SYSTEM_PROMPT
    assert cfg["eval_parser"] == "rich_freeform"


def test_no_bleed_into_other_cfgs():
    build_rich_freeform_cfg("org/model-x")
    ff = build_freeform_cfg("org/model-y")
    assert ff["mode"] == "freeform_sft" and ff["collator"]["target_key"] == "target_text"
    assert ff["wandb"]["project"] == "free_form_sft"
    base = build_cfg("org/model-z")
    assert base["mode"] == "rationale_sft" and base["collator"]["target_key"] == "target_json"


# --- gen-eval parser ------------------------------------------------------------

def test_rich_prediction_parser():
    text = _render_rich_target(FULL_PAYLOAD)
    assert _parse_rich_freeform_prediction(text) == "sarcastic"
    assert _parse_rich_freeform_prediction("LABEL: non_sarcastic\nEXPLANATION: aligned") == "not_sarcastic"
    # fallback chain: freeform first-line, then JSON
    assert _parse_rich_freeform_prediction("unknown\nno anchors") == "unknown"
    assert _parse_rich_freeform_prediction('{"label": "sarcastic"}') == "sarcastic"


def test_resolve_pred_parser_rich():
    assert _resolve_pred_parser({"eval_parser": "rich_freeform"}) is _parse_rich_freeform_prediction
    assert _resolve_pred_parser({"eval_parser": "freeform"}) is _parse_freeform_prediction
    assert _resolve_pred_parser({}) is _parse_prediction


# --- classification MAR paths -----------------------------------------------------

def test_classification_mar_rich_vs_freeform(tmp_path):
    from src.eval.classification import evaluate_file

    rich_rec = {
        "id": "a", "modality": "both", "gt": "sarcastic",
        "output": _render_rich_target(FULL_PAYLOAD),
    }
    rich_path = tmp_path / "rich_freeform_sft_iid_20260101_000000.jsonl"
    rich_path.write_text(json.dumps(rich_rec) + "\n")
    body = evaluate_file(rich_path, "rich_freeform_sft")
    assert body["metrics"]["mar"] == 1.0
    assert body["metrics"]["both"]["accuracy"] == 1.0

    ff_rec = {"id": "b", "modality": "both", "gt": "sarcastic", "output": "sarcastic\nbecause mismatch"}
    ff_path = tmp_path / "freeform_sft_iid_20260101_000000.jsonl"
    ff_path.write_text(json.dumps(ff_rec) + "\n")
    body = evaluate_file(ff_path, "freeform_sft")
    assert body["metrics"]["mar"] is None
    assert body["metrics"]["both"]["accuracy"] == 1.0
