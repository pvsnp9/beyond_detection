from typing import Dict, Any
from config.logistics import MDPOParams
from config.queries import Queries
from src.utils.image_utils import normalize_image, resize_cap

def format_hf_dpo_data(example: Dict[str, Any], processor:Any)->Dict[str, Any]:
    try:
        if "image" not in example:
            raise KeyError(f"Missing 'image' in example.")
        if "query" not in example and "prompt" not in example:
            raise KeyError("Missing 'query' in example.")
        if "chosen" not in example or "rejected" not in example:
            raise KeyError(f"Missing 'chosen' or 'rejected' in example.")

        # raw HF field is `query`; keep `prompt` fallback for older datasets
        query = example.get("query", example.get("prompt", ""))
        modality = str(example.get("modality", "both")).strip().lower() or "both"

        # Modality-aware prompt construction, mirroring the SFT collators: the
        # dataset stores the full image/caption on every row, but unimodal rows
        # must not expose the "missing" modality or abstention is unlearnable.
        caption = (example.get("caption") or "").strip()
        if modality == "image" or not caption:
            user_text = str(query).strip()
        else:
            user_text = f"{query}\nCAPTION: {caption}".strip()

        if modality == "text":
            images = []
            user_content = [{"type": "text", "text": user_text}]
        else:
            mdpo_params = MDPOParams()
            img = resize_cap(
                normalize_image(example.get("image", None)),
                max_side=mdpo_params.max_image_side,
                max_pixels=mdpo_params.max_image_pixels,
            )
            if type(processor).__name__ == "AyaVisionProcessor":
                # aya SFT trained on fixed single-tile images; keep DPO consistent
                side = int(processor.img_size)
                img = img.resize((side, side))
            images = [img]
            user_content = [{"type": "image"}, {"type": "text", "text": user_text}]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": Queries().SYSTEM_PROMPT.strip()}]},
            {"role": "user", "content": user_content},
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {
            "query": str(query),
            "images": images,
            "prompt": prompt,
            "modality": modality,
        }

    except Exception as e:
        print(f"error while formatting DPO HF dataet: {e}")
        raise e
