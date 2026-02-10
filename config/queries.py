from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass(frozen=True)
class Queries:
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


SYSTEM_PROMPT: str = (
    "Role: Expert Multimodal Sarcasm Analyst.\n"
    "Task: Judge sarcasm in image+caption via cross-modal incongruity.\n"
    "Output: ONLY one valid JSON object (no markdown, no extra text).\n\n"
    "Schema:\n"
    "{\n"
    '  "label": "sarcastic" | "non_sarcastic",\n'
    '  "need_explanation": true | false,\n'
    '  "visual_facts": [{"id": int, "fact": "str"}],\n'
    '  "evidence_fact_ids": [int],\n'
    '  "incongruity": "str",\n'
    '  "explanation": "str"\n'
    "}\n\n"
    "Rules:\n"
    "1) visual_facts: 2-4 directly observable facts; ids are consecutive starting at 0.\n"
    "2) evidence_fact_ids: subset of visual_facts ids used to support the decision.\n"
    '3) label="sarcastic" only if caption’s intended meaning conflicts with visual facts (irony/mockery/exaggeration); else "non_sarcastic".\n'
    '4) incongruity: "" if non_sarcastic; otherwise describe the specific mismatch (caption literal meaning vs visual reality).\n'
    "5) explanation: if sarcastic, justify using evidence_fact_ids; if non_sarcastic, brief alignment statement.\n"
    '6) If need_explanation=false: visual_facts=[], evidence_fact_ids=[], incongruity="", explanation="".\n'
    "7) Strict JSON: double quotes, no trailing commas, no extra keys."
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
You DO NOT have access to the image. You must rely ONLY on the provided CAPTION and VISUAL_FACTS.
Your job is to create a strong, plausible decoy that is well-formed and similar length to CHOSEN, but WRONG due to exactly ONE controlled mistake.

OUTPUT FORMAT RULES (strict):
- Output MUST be a single valid JSON object (no markdown, no extra text).
- Use keys: id, rejected, meta
- The "rejected" value MUST be a single string in the same sectioned format as CHOSEN:
  [VISUAL_FACTS] ... [TEXT_LITERAL] ... [INCONGRUITY] ... [EXPLANATION]
- The rejected MUST:
  (a) include the provided visual facts (do not write “not used”),
  (b) include a coherent TEXT_LITERAL derived from CAPTION,
  (c) include an INCONGRUITY and EXPLANATION that look reasonable,
  (d) be wrong due to exactly ONE error_type from the allowed set.

ALLOWED ERROR TYPES (choose exactly one):
1) hallucinated_visual_detail  -> mention ONE concrete visual detail that is NOT in VISUAL_FACTS (only one hallucination).
2) wrong_incongruity_pivot     -> use only given facts, but identify the wrong contradiction (plausible but incorrect pivot).
3) evidence_mismatch           -> INCONGRUITY/EXPLANATION claim is plausible, but it is not supported by the provided VISUAL_FACTS (without adding new visual details).
4) overgeneralized_world_claim -> makes the explanation hinge on an unjustified external claim about the named entity/event (without adding new visual details).

CONSTRAINTS:
- Keep length within ±15% of CHOSEN (approximate).
- Do NOT change the meaning of CAPTION in TEXT_LITERAL.
- Do NOT mention hashtags, “tone/common usage,” “without visual context,” or that you lack the image.
- The rejected should be fluent and convincing; it must not be obviously worse or shorter.
- Safety: do not add protected-attribute inferences, slurs, or calls for violence.

QUALITY METADATA (required):
In meta include:
- error_type: one of the allowed types
- hallucination_span: the exact phrase that is the mistake (or empty string if not hallucinated_visual_detail)
- confidence: low/medium/high (how confident you are this rejected is wrong for exactly one reason)
- notes: 1 short sentence explaining why it’s wrong
"""

ALLOWED_ERROR_TYPES = [
    "hallucinated_visual_detail",
    "wrong_incongruity_pivot",
    "evidence_mismatch",
    "overgeneralized_world_claim",
]

# JSON Schema for Structured Outputs (Responses API text.format json_schema)
REJECT_SCHEMA: Dict[str, Any] = {
    "name": "mdpo_rejected_generator",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        
        "required": ["id", "rejected", "meta"],
        "properties": {
            "id": {"type": "string"},
            "rejected": {"type": "string"},
            "meta": {
                "type": "object",
                "additionalProperties": False,
                "required": ["error_type", "hallucination_span", "confidence", "notes"],
                "properties": {
                    "error_type": {
                        "type": "string",
                        "enum": ALLOWED_ERROR_TYPES,
                    },
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
