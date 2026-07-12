"""CPU-only tests for the freeform_sft pipeline (no GPU, no model downloads).

Run: pytest src/test/test_freeform_sft.py -q
Works under both conda envs (flash_attn_sft / qwen3vl).
"""

import json

from datasets import Dataset

from config.logistics import build_cfg, build_freeform_cfg
from config.queries import Queries
from src.collators.inference_collator import (
    gemma_inference_collator,
    llama_inference_collator,
    qwen_inference_collator,
)
from src.datasets.loader import FREEFORM_LABELS, _derive_freeform_target, add_freeform_target
from src.utils.eval_sarcasm import (
    _normalize_label,
    _parse_freeform_prediction,
    _parse_prediction,
    _resolve_pred_parser,
)


# --- target derivation -----------------------------------------------------

def test_derive_freeform_target_valid():
    tj = json.dumps({"label": "sarcastic", "explanation": "Caption contradicts the image."})
    out = _derive_freeform_target(tj)
    assert out["target_text"] == "sarcastic\nCaption contradicts the image."


def test_derive_freeform_target_non_sarcastic():
    tj = json.dumps({"label": "non_sarcastic", "explanation": "Text and image align."})
    assert _derive_freeform_target(tj)["target_text"] == "non_sarcastic\nText and image align."


def test_derive_freeform_target_invalid_json():
    out = _derive_freeform_target("{not valid json")
    assert out["target_text"] == "unknown"


def test_derive_freeform_target_empty_explanation():
    tj = json.dumps({"label": "unknown", "explanation": ""})
    assert _derive_freeform_target(tj)["target_text"] == "unknown"


def test_derive_freeform_target_bogus_label():
    tj = json.dumps({"label": "maybe", "explanation": "hmm"})
    assert _derive_freeform_target(tj)["target_text"] == "unknown\nhmm"


def test_add_freeform_target_map():
    ds = Dataset.from_dict(
        {
            "id": ["a", "b"],
            "target_json": [
                json.dumps({"label": "sarcastic", "explanation": "mismatch"}),
                json.dumps({"label": "non_sarcastic", "explanation": "aligned"}),
            ],
        }
    )
    mapped = add_freeform_target(ds)
    assert mapped["target_text"] == ["sarcastic\nmismatch", "non_sarcastic\naligned"]
    for text in mapped["target_text"]:
        assert text.splitlines()[0] in FREEFORM_LABELS
    # original column preserved (gen-eval golds still read target_json)
    assert "target_json" in mapped.column_names


# --- config ------------------------------------------------------------------

def test_build_freeform_cfg_overrides():
    cfg = build_freeform_cfg("org/model-x")
    assert cfg["mode"] == "freeform_sft"
    assert cfg["wandb"]["project"] == "free_form_sft"
    assert "freeform" in cfg["wandb"]["tags"]
    assert cfg["collator"]["target_key"] == "target_text"
    assert cfg["collator"]["system_prompt"] == Queries().FREEFORM_SYSTEM_PROMPT
    assert cfg["eval_parser"] == "freeform"


def test_build_cfg_unaffected_by_freeform():
    build_freeform_cfg("org/model-x")
    cfg = build_cfg("org/model-y")
    assert cfg["mode"] == "rationale_sft"
    assert cfg["collator"]["target_key"] == "target_json"
    assert cfg["collator"]["system_prompt"] is None
    assert "eval_parser" not in cfg
    assert "freeform" not in cfg["wandb"]["tags"]


def test_freeform_system_prompt_is_plain_text():
    prompt = Queries().FREEFORM_SYSTEM_PROMPT
    assert "JSON" in prompt  # says "no JSON"
    assert "no JSON" in prompt
    assert "sarcastic | non_sarcastic | unknown" in prompt


# --- prediction parsing -------------------------------------------------------

def test_normalize_label_non_sarcastic_underscore():
    # regression guard: 'non_sarcastic' used to fall through to 'sarcastic'
    assert _normalize_label("non_sarcastic") == "not_sarcastic"
    assert _normalize_label("non sarcastic") == "not_sarcastic"
    assert _normalize_label("sarcastic") == "sarcastic"
    assert _normalize_label("unknown") == "unknown"


def test_parse_freeform_prediction():
    assert _parse_freeform_prediction("sarcastic\nThe caption mocks the image.") == "sarcastic"
    assert _parse_freeform_prediction("non_sarcastic\nAligned.") == "not_sarcastic"
    assert _parse_freeform_prediction("unknown\nImage missing.") == "unknown"
    assert _parse_freeform_prediction("Sarcastic.") == "sarcastic"
    assert _parse_freeform_prediction("") is None
    assert _parse_freeform_prediction(None) is None


def test_parse_freeform_prediction_json_fallback():
    assert _parse_freeform_prediction('{"label": "sarcastic"}') == "sarcastic"


def test_parse_prediction_json_unchanged():
    assert _parse_prediction('{"label": "non_sarcastic", "explanation": "x"}') == "not_sarcastic"
    assert _parse_prediction("garbage with no label") is None


def test_resolve_pred_parser():
    assert _resolve_pred_parser({"eval_parser": "freeform"}) is _parse_freeform_prediction
    assert _resolve_pred_parser({}) is _parse_prediction
    assert _resolve_pred_parser({"eval_parser": None}) is _parse_prediction


# --- inference collators --------------------------------------------------------

def _system_text(messages):
    return messages[0][0]["content"][0]["text"]


def test_inference_collators_default_system_prompt():
    example = [{"query": "q?", "caption": "cap", "image": None}]
    for collate in (qwen_inference_collator, llama_inference_collator, gemma_inference_collator):
        messages, _ = collate(example, mode="text", processor=None)
        assert _system_text(messages) == Queries().SYSTEM_PROMPT


def test_inference_collators_custom_system_prompt():
    example = [{"query": "q?", "caption": "cap", "image": None}]
    freeform_prompt = Queries().FREEFORM_SYSTEM_PROMPT
    for collate in (qwen_inference_collator, llama_inference_collator, gemma_inference_collator):
        messages, _ = collate(example, mode="text", processor=None, system_prompt=freeform_prompt)
        assert _system_text(messages) == freeform_prompt
