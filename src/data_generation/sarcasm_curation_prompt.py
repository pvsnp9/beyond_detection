from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import json


@dataclass(frozen=True)
class SarcasmCurationPrompts:
    system: str
    user_caption_prefix: str = "CAPTION: "
    schema_name: str = "sarcasm_curation_v1"
    schema_version: str = "v1"

    @staticmethod
    def v1() -> "SarcasmCurationPrompts":
        system_prompt = (
            "You are a multimodal data curator for a research dataset on multimodal sarcasm reasoning.\n\n"
            "You MUST follow the procedure below in order, with ZERO deviations.\n\n"
            "DEFINITIONS\n"
            "- \"Caption\" = the provided text associated with the image (not any text printed inside the image).\n"
            "- \"Visual facts\" = only what is directly visible in the image. No guessing, no mind-reading, no unstated context.\n"
            "- \"Partisan politics\" includes: named real-world politicians, political parties, elections, rallies, campaign slogans, "
            "partisan policies, voting instructions, or identifiable political events.\n\n"
            "PROCEDURE (STRICT ORDER)\n"
            "1) VISUAL GROUNDING (ignore the caption):\n"
            "   - Extract ONLY observable visual facts from the image.\n"
            "   - List concrete entities and attributes: people, objects, scene, actions, expressions, visible text-on-image (if legible), setting cues.\n"
            "   - Do NOT infer identity unless there is explicit evidence in the image (e.g., a clearly readable name tag or headline).\n\n"
            "2) SAFETY + TOPIC SCAN (use BOTH the grounded visual facts and the caption):\n"
            "   - Detect whether the sample contains partisan politics.\n"
            "   - Political detection must be based on explicit mentions in the caption OR strong evidence from the image.\n"
            "   - Generic memes / social commentary without real-world partisan references are NON-political.\n\n"
            "3) DECISION:\n"
            "   - If partisan politics is detected: keep=false, politics_detected=true.\n"
            "     - Provide keep_reason (1 sentence).\n"
            "     - Still return valid JSON matching the schema; other fields must be present but may be minimal.\n"
            "   - Else: keep=true, politics_detected=false and proceed.\n\n"
            "4) CLEAN-CoT SARCASM REASONING (ONLY if keep=true):\n"
            "   - Produce:\n"
            "     a) visual_facts: 3–10 factual, specific observations.\n"
            "     b) text_literal: caption’s literal meaning.\n"
            "     c) incongruity: explicit mismatch between visual_facts and text_literal.\n"
            "     d) explanation: concise intended meaning grounded in the image.\n\n"
            "LANGUAGE RULE\n"
            "- Output must be in the same language as the caption when possible.\n\n"
            "HALLUCINATION RULE (CRITICAL)\n"
            "- Do NOT invent objects/actions/people/locations/events not visible.\n"
            "- If uncertain, be conservative and add an appropriate quality flag.\n\n"
            "OUTPUT FORMAT\n"
            "- Return ONLY a JSON object that strictly conforms to the provided JSON Schema.\n"
            "- No markdown, no commentary, no extra keys.\n"
        )
        return SarcasmCurationPrompts(system=system_prompt)

    def build_user_caption(self, caption: str) -> str:
        return f"{self.user_caption_prefix}{(caption or '').strip()}"

    def as_metadata(self) -> Dict[str, str]:
        return {"schema_name": self.schema_name, "schema_version": self.schema_version}


@dataclass(frozen=True)
class SarcasmCurationSpec:
    """
    Wraps prompts + Structured Outputs JSON schema for the Responses API.
    - prompts: SarcasmCurationPrompts (system prompt + caption wrapper)
    - json_schema: object to pass as text.format (Responses API)
    """
    prompts: SarcasmCurationPrompts
    json_schema: Dict[str, Any]

    @staticmethod
    def v1() -> "SarcasmCurationSpec":
        prompts = SarcasmCurationPrompts.v1()

        inner_schema = {
            "name": prompts.schema_name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "schema_version",
                    "language",
                    "keep",
                    "keep_reason",
                    "politics_detected",
                    "politics_signals",
                    "visual_facts",
                    "text_literal",
                    "incongruity",
                    "explanation",
                    "quality_flags"
                ],
                "properties": {
                    "schema_version": {"type": "string", "const": prompts.schema_version},
                    "language": {
                        "type": "string",
                        "description": "Dominant language of the caption/output. Example: 'en', 'zh'."
                    },
                    "keep": {"type": "boolean"},
                    "keep_reason": {
                        "type": "string",
                        "description": "One-sentence reason for keep/discard. If discarded, specify political signal."
                    },
                    "politics_detected": {"type": "boolean"},
                    "politics_signals": {
                        "type": "array",
                        "description": "Short phrases indicating why politics was detected (caption or image evidence). Empty if none.",
                        "items": {"type": "string"}
                    },
                    "visual_facts": {
                        "type": "array",
                        "description": "3–10 factual, specific observations about the image. If keep=false, may be 1–3 minimal facts.",
                        "minItems": 1,
                        "items": {"type": "string"}
                    },
                    "text_literal": {
                        "type": "string",
                        "description": "Literal meaning of the caption at face value. If keep=false, may be empty."
                    },
                    "incongruity": {
                        "type": "string",
                        "description": "Explicit mismatch between visual_facts and text_literal. If keep=false, may be empty."
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Concise sarcastic intent grounded in incongruity. If keep=false, may be empty."
                    },
                    "quality_flags": {
                        "type": "array",
                        "description": "Flags indicating ambiguity or extraction risks.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "low_image_clarity",
                                "possible_ocr_text_in_image",
                                "caption_too_short",
                                "caption_not_sarcastic_or_unclear",
                                "ambiguous_sarcasm",
                                "multiple_possible_incongruities",
                                "requires_world_knowledge",
                                "possible_identity_inference_risk"
                            ]
                        }
                    }
                }
            }
        }

        # Responses API expects: text={"format": {"type": "json_schema", "json_schema": inner_schema}}
        json_schema = {
            "type": "json_schema",
            "name": inner_schema["name"],
            "strict": inner_schema["strict"],
            "schema": inner_schema["schema"],
            }


        return SarcasmCurationSpec(prompts=prompts, json_schema=json_schema)

    def print_schema(self) -> None:
        """Prints the exact object you pass as text.format."""
        print(json.dumps(self.json_schema, indent=2, ensure_ascii=False))

@dataclass(frozen=True)
class SarcasmOODPrompts:
    system: str
    user_caption_template: str

    def build_user_caption(self, caption: str) -> str:
        return self.user_caption_template.format(caption=(caption or "").strip())


@dataclass(frozen=True)
class SarcasmOODGen:
    """
    Target-only prompt + Structured Outputs JSON schema for OOD data.
    This version is for generating grounded targets from image(s) + caption.
    No filtering, no politics screening, no keep/discard decision.
    """

    prompts: SarcasmOODPrompts
    json_schema: Dict[str, Any]

    @staticmethod
    def v1() -> "SarcasmOODGen":
        schema_name = "sarcasm_ood_gen_v1"
        schema_version = "sarcasm_ood_gen_v1"
        system_prompt = """
You are an expert multimodal sarcasm annotator.

Your job is to generate a grounded TARGET annotation for the given image and caption.
You MUST follow the procedure below in order, with ZERO deviations.
This is NOT a curation task and NOT a filtering task.
Always return one complete annotation in valid JSON only.

Guidelines:
1) Use only the provided image and caption.
2) Ground all reasoning in directly visible visual evidence and the literal meaning of the caption.
3) Do not speculate about hidden intent, identity, politics, or background context unless explicitly shown.
4) visual_facts:
   - Write 3 to 6 short, concrete, factual observations from the image.
   - No interpretation, no sarcasm judgment inside visual_facts.
5) text_literal:
   - State the caption's face-value meaning plainly.
6) label:
   - "sarcastic" if the image and caption create a clear contradiction, mismatch, reversal, or ironic tension.
   - "non_sarcastic" if the caption and image align or sarcasm is not sufficiently supported.
   - Be conservative: when uncertain, choose "non_sarcastic".
7) incongruity:
   - For sarcastic cases, describe the specific mismatch between the literal caption meaning and the visual evidence.
   - For non_sarcastic cases, use an empty string "".
8) explanation:
   - Give a concise grounded explanation of why the pair is sarcastic or non_sarcastic.
   - Must be supported by visual_facts and text_literal.
   - Do not invent unseen details.
9) quality_flags:
   - Return a list of quality/safety flags.
   - Use [] when there are no concerns.
10) Output only one valid JSON object matching the schema.
""".strip()

        caption_template = """
Generate the target annotation for the following image-caption pair.

CAPTION:
{caption}
""".strip()

        prompts = SarcasmOODPrompts(
            system=system_prompt,
            user_caption_template=caption_template,
        )

        inner_schema = {
            "name": schema_name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "schema_version",
                    "language",
                    "label",
                    "visual_facts",
                    "text_literal",
                    "incongruity",
                    "explanation",
                    "quality_flags",
                ],
                "properties": {
                    "schema_version": {
                        "type": "string",
                        "const": schema_version
                    },
                    "language": {
                        "type": "string",
                        "description": "Dominant language of the caption/output, e.g. 'en', 'zh'."
                    },
                    "label": {
                        "type": "string",
                        "enum": ["sarcastic", "non_sarcastic"],
                        "description": "Final sarcasm label for the image-caption pair."
                    },
                    "visual_facts": {
                        "type": "array",
                        "description": "3 to 6 factual, concrete visual observations grounded in the image(s).",
                        "minItems": 3,
                        "maxItems": 6,
                        "items": {"type": "string"}
                    },
                    "text_literal": {
                        "type": "string",
                        "description": "Literal face-value meaning of the caption."
                    },
                    "incongruity": {
                        "type": "string",
                        "description": "Specific mismatch between image evidence and text_literal. Empty string if label is non_sarcastic."
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Concise grounded explanation of why the pair is sarcastic or non_sarcastic."
                    },
                    "quality_flags": {
                        "type": "array",
                        "description": "Optional annotation quality/risk flags.",
                        "items": {"type": "string"}
                    }
                }
            }
        }

        json_schema = {
            "type": "json_schema",
            "name": inner_schema["name"],
            "strict": inner_schema["strict"],
            "schema": inner_schema["schema"],
        }

        return SarcasmOODGen(prompts=prompts, json_schema=json_schema)




# if __name__ == "__main__":
#     spec = SarcasmCurationSpec.v1()
#     spec.print_schema()

#     print("\nSystem Prompt:\n")
#     print(spec.prompts.system)
