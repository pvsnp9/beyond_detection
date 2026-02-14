from typing import Optional

from PIL import Image

from config.queries import Queries


def aya_inference_collator(
    batch_data,
    mode: str = None,
    processor=None,
    max_length: Optional[int] = None,
):
    try:
        mode = (mode or "both").strip().lower()
        if mode not in ["both", "text", "image"]:
            raise ValueError('choose ur modality ["both", "text", "image"]')

        if isinstance(batch_data, list):
            examples = batch_data
        else:
            def _get_item(value, idx):
                if isinstance(value, (list, tuple)):
                    return value[idx]
                return value

            batch_len = None
            for key in ("query", "caption", "image"):
                value = batch_data.get(key)
                if isinstance(value, (list, tuple)):
                    batch_len = len(value)
                    break
            if batch_len is None:
                batch_len = 1
            examples = [
                {
                    "query": _get_item(batch_data.get("query"), i),
                    "caption": _get_item(batch_data.get("caption"), i),
                    "image": _get_item(batch_data.get("image"), i),
                }
                for i in range(batch_len)
            ]

        messages_batch = []
        images_batch = None if mode == "text" else []

        for example in examples:
            query = (example.get("query") or "").strip()
            caption = (example.get("caption") or "").strip()
            user_text = f"{query}\nCAPTION: {caption}".strip() if caption else query
            content = []

            if mode == "text":
                content.append({"type": "text", "text": user_text})
            elif mode == "image":
                image = _normalize_image(example.get("image"))
                content.append({"type": "text", "text": ""})
                content.append({"type": "image", "image": image})
                images_batch.append(image)
            else:
                image = _normalize_image(example.get("image"))
                content.append({"type": "text", "text": user_text})
                content.append({"type": "image", "image": image})
                images_batch.append(image)

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": Queries().SYSTEM_PROMPT}],
                },
                {"role": "user", "content": content},
            ]
            messages_batch.append(messages)

        if processor is None:
            return messages_batch, images_batch

        text = processor.apply_chat_template(
            messages_batch, add_generation_prompt=True, tokenize=False
        )
        text_inputs = text if isinstance(text, list) else [text]

        processor_kwargs = {
            "text": text_inputs,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        if max_length is not None:
            processor_kwargs["max_length"] = max_length
        if images_batch is not None:
            processor_kwargs["images"] = images_batch

        return processor(**processor_kwargs)
    except Exception as e:
        print(f"failed to prep aya message from batch: {e}")
        raise


def qwen_inference_collator(
    batch_data,
    mode: str = None,
    processor=None,
    max_length: Optional[int] = None,
):
    try:
        mode = (mode or "both").strip().lower()
        if mode not in ["both", "text", "image"]:
            raise ValueError('choose ur modality ["both", "text", "image"]')

        if isinstance(batch_data, list):
            examples = batch_data
        else:
            def _get_item(value, idx):
                if isinstance(value, (list, tuple)):
                    return value[idx]
                return value

            batch_len = None
            for key in ("query", "caption", "image"):
                value = batch_data.get(key)
                if isinstance(value, (list, tuple)):
                    batch_len = len(value)
                    break
            if batch_len is None:
                batch_len = 1
            examples = [
                {
                    "query": _get_item(batch_data.get("query"), i),
                    "caption": _get_item(batch_data.get("caption"), i),
                    "image": _get_item(batch_data.get("image"), i),
                }
                for i in range(batch_len)
            ]

        messages_batch = []
        images_batch = None if mode == "text" else []

        for example in examples:
            query = (example.get("query") or "").strip()
            caption = (example.get("caption") or "").strip()
            user_text = f"{query}\nCAPTION: {caption}".strip() if caption else query

            if mode == "text":
                user_content = [{"type": "text", "text": user_text}]
            else:
                user_content = [{"type": "image"}]
                user_content.append({"type": "text", "text": "" if mode == "image" else user_text})

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": Queries().SYSTEM_PROMPT}],
                },
                {"role": "user", "content": user_content},
            ]
            messages_batch.append(messages)

            if images_batch is not None:
                images_batch.append(example.get("image"))

        if processor is None:
            return messages_batch, images_batch

        text = processor.apply_chat_template(
            messages_batch, add_generation_prompt=True, tokenize=False
        )
        text_inputs = text if isinstance(text, list) else [text]

        processor_kwargs = {
            "text": text_inputs,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        if max_length is not None:
            processor_kwargs["max_length"] = max_length
        if images_batch is not None:
            processor_kwargs["images"] = images_batch

        return processor(**processor_kwargs)
    except Exception as e:
        print(f"failed to prep qwen message from batch: {e}")
        raise


def llama_inference_collator(
    batch_data,
    mode: str = None,
    processor=None,
    max_length: Optional[int] = None,
):
    try:
        mode = (mode or "both").strip().lower()
        if mode not in ["both", "text", "image"]:
            raise ValueError('choose ur modality ["both", "text", "image"]')

        if isinstance(batch_data, list):
            examples = batch_data
        else:
            def _get_item(value, idx):
                if isinstance(value, (list, tuple)):
                    return value[idx]
                return value

            batch_len = None
            for key in ("query", "caption", "image"):
                value = batch_data.get(key)
                if isinstance(value, (list, tuple)):
                    batch_len = len(value)
                    break
            if batch_len is None:
                batch_len = 1
            examples = [
                {
                    "query": _get_item(batch_data.get("query"), i),
                    "caption": _get_item(batch_data.get("caption"), i),
                    "image": _get_item(batch_data.get("image"), i),
                }
                for i in range(batch_len)
            ]

        messages_batch = []
        images_batch = None if mode == "text" else []

        for example in examples:
            query = (example.get("query") or "").strip()
            caption = (example.get("caption") or "").strip()
            user_text = f"{query}\nCAPTION: {caption}".strip() if caption else query

            if mode == "text":
                user_content = [{"type": "text", "text": user_text}]
            else:
                user_content = [{"type": "image"}]
                user_content.append({"type": "text", "text": "" if mode == "image" else user_text})

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": Queries().SYSTEM_PROMPT}],
                },
                {"role": "user", "content": user_content},
            ]
            messages_batch.append(messages)

            if images_batch is not None:
                images_batch.append([_normalize_image(example.get("image"))])

        if processor is None:
            return messages_batch, images_batch

        text = processor.apply_chat_template(
            messages_batch, add_generation_prompt=True, tokenize=False
        )
        text_inputs = text if isinstance(text, list) else [text]

        processor_kwargs = {
            "text": text_inputs,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        if max_length is not None:
            processor_kwargs["max_length"] = max_length
        if images_batch is not None:
            processor_kwargs["images"] = images_batch

        return processor(**processor_kwargs)
    except Exception as e:
        print(f"failed to prep llama message from batch: {e}")
        raise


def gemma_inference_collator(
    batch_data,
    mode: str = None,
    processor=None,
    max_length: Optional[int] = None,
):
    try:
        mode = (mode or "both").strip().lower()
        if mode not in ["both", "text", "image"]:
            raise ValueError('choose ur modality ["both", "text", "image"]')

        if isinstance(batch_data, list):
            examples = batch_data
        else:
            def _get_item(value, idx):
                if isinstance(value, (list, tuple)):
                    return value[idx]
                return value

            batch_len = None
            for key in ("query", "caption", "image"):
                value = batch_data.get(key)
                if isinstance(value, (list, tuple)):
                    batch_len = len(value)
                    break
            if batch_len is None:
                batch_len = 1
            examples = [
                {
                    "query": _get_item(batch_data.get("query"), i),
                    "caption": _get_item(batch_data.get("caption"), i),
                    "image": _get_item(batch_data.get("image"), i),
                }
                for i in range(batch_len)
            ]

        messages_batch = []
        images_batch = None if mode == "text" else []

        for example in examples:
            query = (example.get("query") or "").strip()
            caption = (example.get("caption") or "").strip()
            user_text = f"{query}\nCAPTION: {caption}".strip() if caption else query

            if mode == "text":
                user_content = [{"type": "text", "text": user_text}]
            else:
                user_content = [{"type": "image"}]
                user_content.append({"type": "text", "text": "" if mode == "image" else user_text})

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": Queries().SYSTEM_PROMPT}],
                },
                {"role": "user", "content": user_content},
            ]
            messages_batch.append(messages)

            if images_batch is not None:
                images_batch.append([_normalize_image(example.get("image"))])

        if processor is None:
            return messages_batch, images_batch

        text = processor.apply_chat_template(
            messages_batch, add_generation_prompt=True, tokenize=False
        )
        text_inputs = text if isinstance(text, list) else [text]

        processor_kwargs = {
            "text": text_inputs,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        if max_length is not None:
            processor_kwargs["max_length"] = max_length
        if images_batch is not None:
            processor_kwargs["images"] = images_batch

        return processor(**processor_kwargs)
    except Exception as e:
        print(f"failed to prep gemma message from batch: {e}")
        raise

def _normalize_image(image):
    if image is None:
        raise ValueError("image is None")
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(image)
        except Exception as exc:
            shape = getattr(image, "shape", None)
            raise ValueError(
                f"invalid image type for PIL conversion: {type(image)} shape={shape}"
            ) from exc
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
