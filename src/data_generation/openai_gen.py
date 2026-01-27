"""Dataset loading helpers for sarcasm/multimodal datasets.

These helpers return Hugging Face `datasets` objects that can be fed into a
multimodal pipeline (e.g., for sending image-text pairs to GPT models).
"""

import io
import json
import logging
import os
from config import Logistics
from pathlib import Path
from typing import List, Optional, Union
import base64
from src.datasets import load_hf_dataset
from openai import APIStatusError, OpenAI, OpenAIError
from .sarcasm_curation_prompt import SarcasmCurationSpec as SPc
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from tqdm import tqdm
from src.utils import append_jsonl, make_dpo_record, make_store_record, validate_minimal, get_completed_ids


key = os.getenv("OPENAI_API_KEY","")
model = os.getenv("OPENAI_MODEL", "gpt-5.2")

logistics = Logistics()

data_dirs = {
    "alita9/muse-sarcasm-explanation": os.path.join(logistics.data_generation_dir, "muse"),
    "alita9/sarcnet": os.path.join(logistics.data_generation_dir, "sarcnet"),
    "coderchen01/MMSD2.0": os.path.join(logistics.data_generation_dir, "mmsd2"),
}



client = OpenAI(api_key=key)
spec = SPc.v1()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Datasets


# OpenAI-based curation utilities.

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.responses.create(**kwargs)

    


def curate_one(image_url: str, caption: str) -> dict:
    try:
        r = completion_with_backoff(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": spec.prompts.system}]},
                {"role": "user", "content": [
                    {"type": "input_image", "image_url": image_url},
                    {"type": "input_text", "text": spec.prompts.build_user_caption(caption)},
                ]},
            ],
            text={"format": spec.json_schema},
            temperature=0
        )
        return json.loads(r.output_text)
    except APIStatusError as exc:
        status = getattr(exc, "status_code", None)
        body = None
        if hasattr(exc, "response") and exc.response is not None:
            try:
                body = exc.response.text
            except Exception:
                body = str(exc)
        logger.error("OpenAI APIStatusError status=%s body=%s", status, body)
        raise RuntimeError("Failed to curate example (API error)") from exc
    except Exception as exc:  # pragma: no cover - surface the cause to callers
        raise RuntimeError("Failed to curate example") from exc


def curate_hf_example(example: dict) -> dict:
    """Curate a Hugging Face sample where `image` is the image field and `text` is the caption."""
    try:
        gen = curate_one(image_url=example["image"], caption=example["text"])
        validate_minimal(gen)
        return gen
    except Exception as exc:  # pragma: no cover - keep source context
        raise RuntimeError("Failed to curate HF example") from exc


def _image_to_data_url(image_obj: object) -> str:
    """
    Normalize HF image field to a data URL the API accepts.
    - If already a string (URL or data URL), return as-is.
    - If a PIL Image, encode to PNG data URL.
    - If raw bytes, encode to PNG data URL.
    """
    if isinstance(image_obj, str):
        return image_obj

    if hasattr(image_obj, "save"):  # PIL Image
        buffer = io.BytesIO()
        image_obj.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    if isinstance(image_obj, (bytes, bytearray)):
        encoded = base64.b64encode(bytes(image_obj)).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    raise TypeError(f"Unsupported image type: {type(image_obj)}")


def iterate_and_curate_datasets(
    output_dirs: Optional[dict] = None,
    hf_datasets: Optional[List[str]] = None,
    *,
    split: Union[str, List[str]] = "train",
    lang: str = "en",
    langs: Optional[List[str]] = None,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    """
    Load HF datasets, curate them with GPT, and store raw/keep/DPO records.

    Args:
        output_dirs: mapping of dataset key -> base output directory. Defaults to module-level `data_dirs`.
        split: dataset split to process, or a list of splits to process.
        lang: default language when `langs` is not provided.
        langs: list of languages to process for sarcnet (and to name output dirs). Defaults to [lang].
        streaming: whether to stream from HF.
        cache_dir: optional HF cache dir.
        limit: optional max number of samples per dataset (for quick runs).
    """
    dirs = output_dirs or data_dirs
    resolved_cache_dir = cache_dir or logistics.hf_cache_dir
    dataset_keys = hf_datasets or list(dirs.keys())
    lang_list = langs or [lang]
    split_list = split if isinstance(split, list) else [split]

    # Get completed IDs to skip already processed samples - remove later 
    completed_ids = get_completed_ids()
    train_ids = completed_ids.get("train", set())
    val_ids = completed_ids.get("validation", set())    

    for ds_key in dataset_keys:
        if ds_key not in dirs:
            logger.warning("Skipping dataset; no output dir configured", extra={"dataset": ds_key})
            continue

        for lang_val in lang_list:
            out_root = Path(dirs[ds_key])
            for split_name in split_list:
                raw_path = out_root / "raw" / lang_val / f"{split_name}.jsonl"
                keep_path = out_root / "keep" / lang_val / f"{split_name}.jsonl"
                dpo_path = out_root / "dpo_va" / lang_val / f"{split_name}.jsonl"

                try:
                    if ds_key == "alita9/sarcnet":
                        config_name = lang_val
                    elif ds_key == "coderchen01/MMSD2.0":
                        if lang_val != lang_list[0]:
                            logger.info(
                                "Skipping dataset/lang combination; single-config dataset",
                                extra={"dataset": ds_key, "lang": lang_val},
                            )
                            continue
                        config_name = "mmsd-v2"
                    elif ds_key == "alita9/muse-sarcasm-explanation":
                        if lang_val != lang_list[0]:
                            logger.info(
                                "Skipping dataset/lang combination; single-config dataset",
                                extra={"dataset": ds_key, "lang": lang_val},
                            )
                            continue
                        config_name = None
                    else:
                        logger.warning(
                            "Skipping dataset; no loader configured",
                            extra={"dataset": ds_key},
                        )
                        continue

                    ds = load_hf_dataset(
                        ds_key,
                        config_name=config_name,
                        split=split_name,
                        streaming=streaming,
                    cache_dir=resolved_cache_dir,
                    )

                    logger.info(
                        "Loaded dataset",
                        extra={"dataset": ds_key, "split": split_name, "lang": lang_val},
                    )
                except Exception:
                    logger.exception(
                        "Failed to load dataset",
                        extra={"dataset": ds_key, "split": split_name, "lang": lang_val},
                    )
                    continue

                for idx, example in enumerate(tqdm(ds, desc=f"{ds_key}-{lang_val}-{split_name}")):
                    if limit is not None and idx >= limit:
                        break

                    sample_id = str(example.get("id", f"{ds_key}-{idx}"))
                    caption = example.get("text")
                    image_raw = example.get("image")

                    # Skip already completed IDs
                    if split_name == "train" and sample_id in train_ids:
                        print("Skipping already completed train ID:", sample_id)
                        continue
                    if split_name in ["validation"] and sample_id in val_ids:
                        print("Skipping already completed validation ID:", sample_id)
                        continue
                    # Remove skip block later

                    if caption is None or image_raw is None:
                        logger.warning(
                            "Missing required fields (image/text)",
                            extra={"dataset": ds_key, "sample_id": sample_id, "lang": lang_val},
                        )
                        continue

                    try:
                        image_ref = _image_to_data_url(image_raw)
                        model_json = curate_hf_example({"image": image_ref, "text": caption})
                        raw_record = make_store_record(sample_id, caption, model_json, model_name=model)
                        append_jsonl(raw_path, raw_record)

                        if model_json.get("keep"):
                            append_jsonl(keep_path, raw_record)

                        dpo_record = make_dpo_record(sample_id, caption, model_json)
                        append_jsonl(dpo_path, dpo_record)
                    except Exception:
                        logger.exception(
                            "Failed to process sample",
                            extra={"dataset": ds_key, "sample_id": sample_id, "idx": idx, "lang": lang_val},
                        )
                        continue

                logger.info(
                    "Completed dataset",
                    extra={"dataset": ds_key, "split": split_name, "lang": lang_val},
                )



# if __name__ == "__main__":
#     hf_datasets = ["alita9/muse-sarcasm-explanation", "alita9/sarcnet", "coderchen01/MMSD2.0"]
#     extract_datasets = hf_datasets[0:2]
#     print("Starting curation...")
#     iterate_and_curate_datasets(
#         output_dirs=data_dirs,
#         hf_datasets=extract_datasets,
#         split=["validation", "test"],
#         langs=["en", "zh"]
#     )
#     print("Curation complete for datasets:", extract_datasets)
