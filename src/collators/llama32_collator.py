from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from config.queries import Queries


class Llama32VisionSFTCollator:
    def __init__(
        self,
        processor: Any,
        training: bool = True,
        max_length: Optional[int] = None,
        image_key: str = "image",
        query_key: str = "query",
        caption_key: str = "caption",
        target_key: str = "target_json",
        system_prompt: Optional[str] = None,
    ) -> None:
        self.processor = processor
        self.training = training
        self.max_length = max_length
        self.image_key = image_key
        self.query_key = query_key
        self.caption_key = caption_key
        self.target_key = target_key
        self.system_prompt = system_prompt or Queries().SYSTEM_PROMPT

    def _user_text(self, example: Dict[str, Any]) -> str:
        query = (example.get(self.query_key) or "").strip()
        caption = (example.get(self.caption_key) or "").strip()
        if caption:
            return f"{query}\nCAPTION: {caption}".strip()
        return query

    def _user_message(self, user_text: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        )
        return messages

    def _prompt_length(self, image: Any, prompt_text: str) -> int:
        prompt_inputs = self.processor(
            text=prompt_text,
            images=[[image]],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        return prompt_inputs["input_ids"].shape[-1]

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [example[self.image_key] for example in batch]
        images_per_sample = [[image] for image in images]
        user_texts = [self._user_text(example) for example in batch]

        full_texts: List[str] = []
        prompt_texts: List[str] = []

        for example, user_text in zip(batch, user_texts):
            user_messages = self._user_message(user_text)

            if self.training:
                target = (example.get(self.target_key) or "").strip()
                full_messages = user_messages + [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": target}],
                    }
                ]
                full_texts.append(
                    self.processor.apply_chat_template(
                        full_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
                prompt_texts.append(
                    self.processor.apply_chat_template(
                        user_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:
                full_texts.append(
                    self.processor.apply_chat_template(
                        user_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        model_inputs = self.processor(
            text=full_texts,
            images=images_per_sample,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        if self.training:
            labels = model_inputs["input_ids"].clone()
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels[labels == pad_id] = -100
            
            # Mask everything before the assistant's response
            # The assistant header tokens for Llama 3.2 are: 
            # <|start_header_id|>assistant<|end_header_id|>\n\n
            assistant_header_ids = self.processor.tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", 
                add_special_tokens=False
            )

            for i in range(labels.shape[0]):
                # Find the start of the assistant header in this specific row
                # search for the sequence of tokens that identifies the assistant
                row_ids = model_inputs["input_ids"][i].tolist()

                # Find the end of the assistant header sequence
                header_index = -1
                for idx in range(len(row_ids) - len(assistant_header_ids)):
                    if row_ids[idx : idx + len(assistant_header_ids)] == assistant_header_ids:
                        header_index = idx + len(assistant_header_ids)
                        break
                
                if header_index != -1:
                    # Mask everything from the start of the sequence up to the end of the header
                    labels[i, :header_index] = -100
                else:
                    # Fallback: if header not found (shouldn't happen), mask everything to be safe
                    labels[i, :] = -100

            model_inputs["labels"] = labels

        return model_inputs
