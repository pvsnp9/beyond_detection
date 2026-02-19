
import os 
import random
import re
from config.logistics import Logistics, REPLACEMENT_WORDS
from typing import Dict, List, Any
from src.cot.build_target import (
    label_to_str, 
    make_visual_fact_objects,
    get_hf_dpo_features
)
from datasets import Dataset, DatasetDict
from src.utils.env import resolve_tokens_and_env
from src.data_generation.dpo_creation import PLAIN_TAG_TOKENS, WORD_RE
from src.utils import read_jsonl
from config.queries import Queries
import json

def replace_tags_cues(text:str)->str:
    try: 
        if not isinstance(text, str) or not text.strip():
            return text

        def _normalize(token: str) -> str:
            return re.sub(r"[^a-z0-9]", "", token.lower())

        def _replace(match: re.Match[str]) -> str:
            token = match.group(0)
            if _normalize(token) in PLAIN_TAG_TOKENS:
                return random.choice(list(REPLACEMENT_WORDS))
            return token

        return WORD_RE.sub(_replace, text)
    except Exception as e: 
        print(f"erro at reaplcement tag: {e}")
        raise


def parse_chosen_rejected(text:str)->Dict[str, Any]:
    try:
        if not isinstance(text, str):
            return {
                "visual_facts": [],
                "text_literal": "",
                "incongruity": "",
                "explanation": "",
            }

        def _section(src: str, start: str, end: str | None) -> str:
            if start not in src:
                return ""
            chunk = src.split(start, 1)[1]
            if end and end in chunk:
                chunk = chunk.split(end, 1)[0]
            return chunk.strip()

        visual_sec = _section(text, "[VISUAL_FACTS]", "[TEXT_LITERAL]")
        text_literal = _section(text, "[TEXT_LITERAL]", "[INCONGRUITY]")
        incongruity = _section(text, "[INCONGRUITY]", "[EXPLANATION]")
        explanation = _section(text, "[EXPLANATION]", None)

        visual_facts: List[str] = []
        if visual_sec:
            for line in visual_sec.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("-"):
                    line = line.lstrip("-").strip()
                if line:
                    visual_facts.append(line)

        return {
            "visual_facts": visual_facts,
            "text_literal": text_literal.strip(),
            "incongruity": incongruity.strip(),
            "explanation": explanation.strip(),
        }

    except Exception as e:
        print(f'exception at parsing text literal, incong, and eplanation: {e}')
        raise e 



def build_target(text: str, label_str: str) -> Dict[str, Any]:
    parsed = parse_chosen_rejected(text)
    visual_objs = make_visual_fact_objects(parsed.get("visual_facts", []))
    evidence_ids = [vf["id"] for vf in visual_objs[: min(3, len(visual_objs))]]
    return {
        "label": label_str,
        "need_explanation": True,
        "visual_facts": visual_objs,
        "evidence_fact_ids": evidence_ids,
        "text_literal": parsed.get("text_literal", "").strip(),
        "incongruity": parsed.get("incongruity", "").strip(),
        "explanation": parsed.get("explanation", "").strip(),
    }


def build_dpo_target(record:Dict[str, Any], mode:str ='train')->Dict[str, Any]:
    try:
        if not isinstance(record, dict):
            return {"chosen": {}, "rejected": {}}

        chosen_text = record.get("chosen", "")
        rejected_text = record.get("rejected", "")

        if mode == "train":
            label_str = label_to_str(record.get("label"))
            return {
                "chosen": build_target(chosen_text, label_str),
                "rejected": build_target(rejected_text, label_str),
            }

        return {
            "chosen": build_target(chosen_text, "sarcastic"),
            "rejected": {},
        }
    except Exception as e:
        print(f"erro at build_dpo_target: {e}")
        raise e 



def creat_train_dpo_record(record:Dict[str, Any])->Dict[str, Any]:
    try:
        if not isinstance(record, dict):
            return record

        def _normalize_quotes(text: str) -> str:
            return (
                text.replace("\u2018", "'")
                .replace("\u2019", "'")
                .replace("\u201C", '"')
                .replace("\u201D", '"')
            )

        updated = dict(record)
        if "caption" in updated and isinstance(updated["caption"], str):
            updated["caption"] = _normalize_quotes(updated["caption"])
        if "chosen" in updated and isinstance(updated["chosen"], str):
            updated["chosen"] = _normalize_quotes(replace_tags_cues(updated["chosen"]))


        if "rejected" in updated and isinstance(updated["rejected"], str):
            updated["rejected"] = _normalize_quotes(replace_tags_cues(updated["rejected"]))
        
        visual_facts = updated.get("visual_facts", [])
        if isinstance(visual_facts, list):
            cleaned_facts = []
            for fact in visual_facts:
                fact_text = str(fact)
                cleaned_facts.append(_normalize_quotes(replace_tags_cues(fact_text)))
            updated["visual_facts"] = cleaned_facts
        elif isinstance(visual_facts, str):
            updated["visual_facts"] = [_normalize_quotes(replace_tags_cues(visual_facts))]

        return updated
    except Exception as e:
        print(f"erro at  create train dpo recoed: {e}")
        raise e


def create_hf_formatted_dpo_dataset(lang: str, logistics: Logistics, out_dir:str = "dpo_final"):
    if not lang == "en": raise NotImplementedError(f"No mDPO data creation method for lang{lang}")
    random.seed(logistics.seed)
    try:
        input_dir = os.path.join(logistics.project_root_dir, logistics.processed_data_dir, "dpo", lang)
        splits = logistics.splits
        # out dir to save file
        out_dir = os.path.join(logistics.project_root_dir, logistics.processed_data_dir, out_dir, lang)
        os.makedirs(out_dir, exist_ok=True)
        for split in splits:
            split_path = os.path.join(input_dir, f"{split}.jsonl")
            if not os.path.exists(split_path):
                print(f"Missing split file: {split_path}")
                # just create an empty file 
                out_path = os.path.join(out_dir, f"{split}.jsonl")
                with open(out_path, "w", encoding="utf-8") as f:
                    pass
                print(f"[{split}] [size:0] Completed writing HF formatted DPO dataset to {out_dir}")
                continue
            
            print(f"Reading split {split} from {split_path}")
            rows = read_jsonl(split_path)
            out_path = os.path.join(out_dir, f"{split}.jsonl")
            written = 0
            skipped = 0
            failed = 0
            with open(out_path, "w", encoding="utf-8") as f:
                for idx, row in enumerate(rows):
                    try:
                        record = creat_train_dpo_record(row)
                        target = build_dpo_target(record=record, mode=split)
                    except Exception:
                        failed += 1
                        continue
                    
                    chosen = target.get("chosen")
                    rejected = target.get("rejected")
                    if not chosen:
                        skipped += 1
                        continue

                    example = {
                        "id": record.get("id"),
                        "image": record.get("local_image_path"),
                        "caption": record.get("caption"),
                        "modality": str(record.get("modality", "both")).strip().lower() if record.get("modality") else "both",
                        "prompt": Queries().DPO_QUERY,
                        "chosen": json.dumps(chosen, ensure_ascii=False),
                        "rejected": json.dumps(rejected, ensure_ascii=False),
                        "label_gt": label_to_str(record.get("label")),
                        "language": record.get("language", "en"),
                        "source": record.get("source"),
                        "quality_flags": record.get("quality_flags", []),
                        "visual_facts": record.get("visual_facts", []),
                        "rejected_meta": record.get("rejected_meta", {}),
                    }
                    
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    written += 1

            print(
                f"[{split}] [size:{written}] skipped={skipped} failed={failed} "
                f"Completed writing HF formatted DPO dataset to {out_dir}"
            )
    except Exception as e:
        print("Error:", e)
        raise e


def publish_dpo_hf_data(logistics: Logistics, data_dir: str = "dpo_final") -> None:
    try:
        lang = ["en"]
        features = get_hf_dpo_features()
        for lang in lang:
            input_dir = os.path.join(
                logistics.project_root_dir, logistics.processed_data_dir, data_dir, lang
            )
            datasets_by_split: dict[str, Dataset] = {}
            for split in logistics.splits:
                split_path = os.path.join(input_dir, f"{split}.jsonl")
                if not os.path.exists(split_path):
                    print(f"Missing split file: {split_path}")
                    datasets_by_split[split] = Dataset.from_dict(
                        {k: [] for k in features.keys()}, features=features
                    )
                    print(f"[{lang}] [{split}] size: 0")
                    continue

                rows = read_jsonl(split_path)
                if rows:
                    data_dict = {}
                    for key in features.keys():
                        if key in ("quality_flags", "visual_facts"):
                            data_dict[key] = [
                                row.get(key, []) if isinstance(row.get(key, []), list) else []
                                for row in rows
                            ]
                        else:
                            data_dict[key] = [row.get(key) for row in rows]
                else:
                    data_dict = {k: [] for k in features.keys()}

                datasets_by_split[split] = Dataset.from_dict(data_dict, features=features)
                print(f"[{lang}] [{split}] size: {len(rows)}")

            dataset_dict = DatasetDict(datasets_by_split)
            resolved = resolve_tokens_and_env(logistics_cfg=logistics)
            dataset_dict.push_to_hub(
                logistics.hf_dpo_ds_id,
                config_name=lang,
                token=resolved["hf_token"],
            )
            print(f"Published DPO dataset for {lang} to {logistics.hf_dpo_ds_id}")
    except Exception as e:
        print(f"exception while pushing to HF [id]: {logistics.hf_dpo_ds_id}: {e}")

def main():
    logistics = Logistics()
    # for lang in logistics.langs:
    create_hf_formatted_dpo_dataset(lang="en", logistics=logistics)
    publish_dpo_hf_data(logistics=logistics)
    

# if __name__ == "__main__":
#     main()
