from dataclasses import dataclass, field


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
            "Provide a step-by-step reasoning trace of why this post is sarcastic.",
            "Map the visual facts of this image to the caption to explain the sarcasm.",
            "Identify the specific visual cues that make this caption sarcastic."
        ]
    )

    CONDITIONAL_QUERIES: list[str] = field(
        default_factory=lambda: [
            "Is this sarcastic? If yes, explain why. If not, say no.",
            "Is it sarcastic? Explain only if it is sarcastic.",
            "Tell me if this is sarcastic, and explain the reason only if the answer is yes."
        ]
    )

    # Define the concise universal system prompt.
    SYSTEM_PROMPT: str = (
        "Role: Expert Multimodal Sarcasm Analyst.\n"
        "Task: Analyze image-caption pairs for sarcasm via cross-modal incongruity.\n"
        "Output: Return ONLY a valid JSON object. No markdown, no filler.\n\n"
        "JSON Schema:\n"
        "{\n"
        '  "label": "sarcastic" | "non_sarcastic",\n'
        '  "need_explanation": true | false,\n'
        '  "visual_facts": [{"id": int, "fact": "str"}],\n'
        '  "evidence_fact_ids": [int],\n'
        '  "incongruity": "str",\n'
        '  "explanation": "str"\n'
        "}\n\n"
        "Rules:\n"
        "1. Identify objective `visual_facts` first.\n"
        '2. If no sarcasm exists, `incongruity` must be "".\n'
        "3. `evidence_fact_ids` must map to `visual_facts` IDs.\n"
        "4. If `need_explanation` is false, return empty visual reasoning fields."
    )



# 
