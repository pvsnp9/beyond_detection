from typing import Dict, Any
from config.queries import Queries
from src.collators.dpo_inference_collator import _normalize_image

def format_hf_dpo_data(example: Dict[str, Any], processor:Any)->Dict[str, Any]:
    try:
        if "image" not in example:
            raise KeyError(f"Missing 'image' in example.")
        if "prompt" not in example:
            raise KeyError(f"Missing 'prompt (query)' in example.")
        if "chosen" not in example or "rejected" not in example:
            raise KeyError(f"Missing 'chosen' or 'rejected' in example.")
        
        # for eval we need query 
        query = example.get('prompt', '')
        img = _normalize_image(example.get('image', None))
        user_text = f"{example.get('prompt', '')}\nCAPTION: {example.get('caption', '')}".strip()
        
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": Queries().SYSTEM_PROMPT.strip()}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        ]
        

        prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
        )
        return {
            "query": str(query),
            "images": [img],
            "prompt": prompt
        }
        
    except Exception as e:
        print(f"error while formatting DPO HF dataet: {e}")
        raise e 