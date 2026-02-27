from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass(frozen=True)
class Queries:
    SFT_QUERY: str = "Evaluate this image and text pair for any sarcastic intent."
    
    DETECTION_QUERIES: list[str] = field(
        default_factory=lambda: [
            "Analyze the relationship between the image and caption to detect sarcasm.",
            "Evaluate this image and text pair for any sarcastic intent.",
            "Does the visual context support or contradict the literal meaning of the caption?"
        ]
    )

    EXPLANATION_QUERIES: list[str] = field(
        default_factory=lambda: [
            "Provide a step-by-step reasoning trace of why this post is sarcastic or non-sarcastic.",
            "Map the visual facts of this image to the caption to explain the sarcasm.",
            "Identify the specific visual cues that make this caption sarcastic."
        ]
    )

    DPO_QUERY: str = "Assess sarcasm using the available modalities by comparing caption meaning with visual evidence when present. If image or caption is missing, return unknown and explain what is missing."

    SYSTEM_PROMPT: str = (
        "Role: Expert Multimodal Sarcasm Analyst.\n"
        "Task: Judge sarcasm in image+caption via cross-modal incongruity.\n"
        "Output: ONLY one valid JSON object (no markdown, no extra text).\n\n"
        "Schema:\n"
        "{\n"
        '  "need_explanation": true | false,\n'
        '  "visual_facts": [{"id": int, "fact": "str"}],\n'
        '  "evidence_fact_ids": [int],\n'
        '  "text_literal": "str",\n'
        '  "incongruity": "str",\n'
        '  "label": "sarcastic" | "non_sarcastic" | "unknown",\n'
        '  "explanation": "str",\n'
        '  "missing_modalities": ["image" | "text"]\n'
        "}\n\n"
        "Rules:\n"
        "1) visual_facts: 2-4 directly observable facts; ids are consecutive starting at 0.\n"
        "2) evidence_fact_ids: subset of visual_facts ids used to support the decision.\n"
        "3) text_literal: restate the caption's literal meaning as a plain, non-ironic statement.\n"
        '4) label: "sarcastic" only if intended meaning conflicts with visual facts; "non_sarcastic" if aligned; "unknown" if a required modality is missing.\n'
        '5) incongruity: "" if non_sarcastic/unknown; otherwise describe the specific mismatch (text_literal vs visual reality).\n'
        "6) explanation: if sarcastic, justify using evidence_fact_ids; if non_sarcastic, brief alignment; if unknown, state what is missing.\n"
        '7) If need_explanation=false: visual_facts=[], evidence_fact_ids=[], text_literal="", incongruity="", explanation="", missing_modalities=[].\n'
        '8) missing_modalities: [] if both image and caption are present; otherwise include the missing ones (e.g., ["text"] or ["image"]).\n'
        "9) Strict JSON: double quotes, no trailing commas, no extra keys."
    )




NON_SARC_EXPLANATION_TEMPLATES = [
    "The caption is meant literally and matches the image content. There's no contradiction or hidden meaning, so it's non-sarcastic.",
    "The text describes what's shown in the image in a straightforward way. The tone is sincere and the visual context supports the message.",
    "This post is simply conveying information or a clear message. Nothing in the image suggests a mismatch, exaggeration, or mockery.",
    "The wording is neutral and consistent with the visual facts. Since there's no ironic contrast between text and image, it's non-sarcastic.",
    "The caption expresses a genuine feeling or opinion, and the image reinforces it rather than undermining it. There's no sarcastic intent.",
    "Sarcasm typically involves incongruity between what is said and what is meant, often signaled by a mismatch with the image. Here, the message and visuals align, so it's non-sarcastic.",
    "The text doesn't reverse its meaning or take a hidden jab at someone. It reads as straightforward, and the image doesn't introduce a contradictory cue.",
    "Given the visual context, the caption makes sense as an honest statement. There's no evidence of irony, ridicule, or intentional contradiction.",
    "The post appears to communicate its point sincerely. The image supports the same idea, so there's no reason to interpret it as sarcasm.",
    "The caption and image are consistent and literal, with no ironic mismatch. Therefore, it's non-sarcastic.",
]




REJECTED_SYSTEM_PROMPT = """You are generating REJECTED (non-preferred) responses for multimodal DPO (mDPO).
You DO NOT have access to the image. You must rely ONLY on the provided CAPTION, MODALITY, CHOSEN_JSON, and VISUAL_FACTS.
Your job is to create a strong, plausible decoy JSON that is well-formed and similar quality to CHOSEN_JSON, but WRONG due to exactly ONE controlled mistake.

OUTPUT FORMAT RULES (strict):
- Output MUST be a single valid JSON object (no markdown, no extra text).
- Top-level keys: id, rejected, meta
- "rejected" MUST be a JSON object with the same target schema fields as CHOSEN_JSON:
  need_explanation, visual_facts, evidence_fact_ids, text_literal, incongruity, label, explanation, missing_modalities
- Keep modality behavior consistent with the provided MODALITY:
  - both: missing_modalities must be []
  - image: missing_modalities must include "text"
  - text: missing_modalities must include "image"
- Use exactly ONE error_type from ALLOWED_ERROR_TYPES_FOR_THIS_RECORD.

ERROR TYPE DEFINITIONS:
- hallucinated_visual_detail: add one concrete visual fact/detail not supported by VISUAL_FACTS.
- omitted_key_visual_evidence: omit a key visual fact/evidence reference needed for the chosen judgment.
- wrong_incongruity_pivot: pick the wrong contradiction pivot while still sounding plausible (cross-modal only).
- evidence_mismatch: make a claim in incongruity/explanation that is not supported by provided VISUAL_FACTS (without adding a new concrete visual detail).
- label_mismatch: set the wrong label while keeping the rest mostly plausible.
- unsupported_world_knowledge: rely on an unjustified external claim about an entity/event not grounded in inputs.

CONSTRAINTS:
- Keep the rejected JSON fluent and convincing; it should not be obviously weaker than CHOSEN_JSON.
- Do NOT add extra keys.
- Do NOT mention that you lack the image.
- Do NOT use generic filler phrases (e.g., “tone/common usage”).
- Safety: no slurs, protected-attribute inferences, or violence advocacy.

QUALITY METADATA (required in meta):
- error_type: one allowed error type
- error_field: the primary field containing the mistake (e.g., label, explanation, incongruity, visual_facts[3].fact)
- hallucination_span: exact hallucinated phrase if error_type=hallucinated_visual_detail, else ""
- confidence: low|medium|high
- notes: one short sentence explaining why the rejected output is wrong
"""

ALLOWED_ERROR_TYPES = [
    "hallucinated_visual_detail",
    "omitted_key_visual_evidence",
    "wrong_incongruity_pivot",
    "evidence_mismatch",
    "label_mismatch",
    "unsupported_world_knowledge",
]


def get_allowed_error_types_for_modality(modality: str) -> List[str]:
    m = str(modality or "").strip().lower()
    if m == "text":
        return ["label_mismatch"]
    if m == "image":
        return [
            "hallucinated_visual_detail",
            "omitted_key_visual_evidence",
            "evidence_mismatch",
            "label_mismatch",
            "unsupported_world_knowledge",
        ]
    # default "both"
    return list(ALLOWED_ERROR_TYPES)

# JSON Schema for Structured Outputs (Responses API text.format json_schema)
REJECT_SCHEMA: Dict[str, Any] = {
    "name": "mdpo_rejected_generator",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        
        "required": ["id", "rejected", "meta"],
        "properties": {
            "id": {"type": "string"},
            "rejected": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "need_explanation",
                    "visual_facts",
                    "evidence_fact_ids",
                    "text_literal",
                    "incongruity",
                    "label",
                    "explanation",
                    "missing_modalities",
                ],
                "properties": {
                    "need_explanation": {"type": "boolean"},
                    "visual_facts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "fact"],
                            "properties": {
                                "id": {"type": "integer"},
                                "fact": {"type": "string"},
                            },
                        },
                    },
                    "evidence_fact_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "text_literal": {"type": "string"},
                    "incongruity": {"type": "string"},
                    "label": {
                        "type": "string",
                        "enum": ["sarcastic", "non_sarcastic", "unknown"],
                    },
                    "explanation": {"type": "string"},
                    "missing_modalities": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["image", "text"]},
                    },
                },
            },
            "meta": {
                "type": "object",
                "additionalProperties": False,
                "required": ["error_type", "error_field", "hallucination_span", "confidence", "notes"],
                "properties": {
                    "error_type": {
                        "type": "string",
                        "enum": ALLOWED_ERROR_TYPES,
                    },
                    "error_field": {"type": "string"},
                    "hallucination_span": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "notes": {"type": "string"},
                },
            },
        },
    },
    "strict": True,
}


# 
