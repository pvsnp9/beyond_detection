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



# 
