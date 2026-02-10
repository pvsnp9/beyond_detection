from src.datasets import load_hf_dataset
from config.logistics import Logistics, LocalDataDirs
import json
from src.utils import read_jsonl
from src.utils.env import resolve_tokens_and_env
from src.cot.build_target import build_sft_rows_from_raw, get_hf_sft_features
import os
import shutil
from datasets import Dataset, DatasetDict



logistics = Logistics()
local_dirs = LocalDataDirs()

#"This script produced data leakeg (but cleaend with @data_generation/clean_dataet), 
# read data from combined dir, use it after review "

def build_create_combined_sft(langs: list[str] = ["en"], splits: list[str] = ["train", "validation", "test"]):
    try:
        dirs_name = sorted(os.listdir(os.path.join(logistics.project_root_dir, logistics.data_generation_dir)))
        for lang in langs:
            for split in splits:
                out_dir = os.path.join(logistics.project_root_dir, logistics.combined_data_dir, "sft" ,lang)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{split}.jsonl")
                combined_by_id: dict[str, dict] = {}

                for dir_name in dirs_name:
                    print(f"Data:{dir_name} Language: {lang}, split: {split}")
                    data_path = os.path.join(logistics.project_root_dir,
                                             logistics.data_generation_dir,
                                             dir_name,
                                             "keep",
                                             lang,
                                             f"{split}.jsonl")
                    if os.path.exists(data_path):
                        gen_dataset = read_jsonl(data_path)
                        # read and merge with hf dataset
                        hf_id = logistics.hf_datatset_ids.get(dir_name, "")
                        print(f"=============================\n[{data_path}]\n Directory: {dir_name}, Number of samples: {len(gen_dataset)} \n=================")
                        config_name = "mmsd-v2" if hf_id == "coderchen01/MMSD2.0" else None if hf_id == "alita9/muse-sarcasm-explanation" else lang
                        hf_dataset = load_hf_dataset(hf_id, split=split, config_name=config_name)
                        
                        # Create a mapping from ID to label for quick lookup
                        id_to_label = {
                            row['id']: (
                                1 if dir_name == "muse" else 
                                row['multi_label'] if dir_name == "sarcnet" else 
                                row['label']  # Defaults to mmsd2 logic
                            ) 
                            for row in hf_dataset
                        }

                        for entry in gen_dataset:
                            entry_id = entry.get('id')
                            if not entry_id:
                                continue
                            if entry_id in id_to_label:
                                entry['label'] = id_to_label[entry_id]
                                entry['source'] = dir_name
                                # remvoe the extension for media file name
                                entry_id = entry_id.split('.')[0]
                                local_media_apth = os.path.join(getattr(local_dirs, dir_name), f"{entry_id}.jpg")
                                entry['local_image_path'] = local_media_apth
                                entry['id'] = entry_id
                                combined_by_id[entry_id] = entry
                            else:
                                print(f"Warning: ID {entry_id} not found in HF dataset for {dir_name}.") 
                    else:
                        print(f"  Directory: {dir_name}, File not found: {data_path}")
                with open(out_path, "w", encoding="utf-8") as f:
                    for entry_id in sorted(combined_by_id.keys()):
                        entry = combined_by_id[entry_id]
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"An error occurred: {e}")


def build_create_combined_dpo(langs: list[str] = ["en"], splits: list[str] = ["train", "validation", "test"]):
    try:
        dirs_name = sorted(os.listdir(os.path.join(logistics.project_root_dir, logistics.data_generation_dir)))
        for lang in langs:
            for split in splits:
                out_dir = os.path.join(logistics.project_root_dir, logistics.combined_data_dir, "dpo", lang)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{split}.jsonl")
                combined_by_id: dict[str, dict] = {}

                for dir_name in dirs_name:
                    print(f"DPO Data:{dir_name} Language: {lang}, split: {split}")
                    data_path = os.path.join(
                        logistics.project_root_dir,
                        logistics.data_generation_dir,
                        dir_name,
                        "dpo_va",
                        lang,
                        f"{split}.jsonl",
                    )
                    if os.path.exists(data_path):
                        gen_dataset = read_jsonl(data_path)
                        hf_id = logistics.hf_datatset_ids.get(dir_name, "")
                        print(
                            f"=============================\n[{data_path}]\n Directory: {dir_name}, Number of samples: {len(gen_dataset)} \n================="
                        )
                        config_name = "mmsd-v2" if hf_id == "coderchen01/MMSD2.0" else None if hf_id == "alita9/muse-sarcasm-explanation" else lang
                        hf_dataset = load_hf_dataset(hf_id, split=split, config_name=config_name)

                        id_to_label = {
                            row["id"]: (
                                1 if dir_name == "muse" else
                                row["multi_label"] if dir_name == "sarcnet" else
                                row["label"]
                            )
                            for row in hf_dataset
                        }
                        id_to_caption = {row["id"]: row.get("text") for row in hf_dataset}

                        for entry in gen_dataset:
                            entry_id = entry.get("id")
                            if not entry_id:
                                continue
                            if entry_id in id_to_label:
                                entry["label"] = id_to_label[entry_id]
                                entry["source"] = dir_name
                                if "caption" not in entry and id_to_caption.get(entry_id):
                                    entry["caption"] = id_to_caption[entry_id]
                                entry_id = entry_id.split(".")[0]
                                local_media_path = os.path.join(getattr(local_dirs, dir_name), f"{entry_id}.jpg")
                                entry["local_image_path"] = local_media_path
                                entry["id"] = entry_id
                                combined_by_id[entry_id] = entry
                            else:
                                print(f"Warning: ID {entry_id} not found in HF dataset for {dir_name}.")
                    else:
                        print(f"  Directory: {dir_name}, File not found: {data_path}")

                with open(out_path, "w", encoding="utf-8") as f:
                    for entry_id in sorted(combined_by_id.keys()):
                        entry = combined_by_id[entry_id]
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"An error occurred: {e}")

#ends here 

def build_save_hf_formatted_sft_dataset(lang: str):
    try:
        # read the combined dataset from local path for given lang
        # split: train, validation, test (each has separate jsonl files)
        combined_dir = os.path.join(logistics.project_root_dir, logistics.combined_data_dir, "sft", lang)
        splits = logistics.splits
        # out dir to save file 
        out_dir = os.path.join(logistics.project_root_dir, logistics.processed_data_dir, "sft", lang)
        os.makedirs(out_dir, exist_ok=True)
        for split in splits:
            split_path = os.path.join(combined_dir, f"{split}.jsonl")
            if not os.path.exists(split_path):
                print(f"Missing split file: {split_path}")
                out_path = os.path.join(out_dir, f"{split}.jsonl")
                with open(out_path, "w", encoding="utf-8") as f:
                    pass
                print(f"[{split}] [size:0] Completed writing HF formatted SFT dataset to {out_dir}")
                continue

            raw_rows = read_jsonl(split_path)
            sft_rows = []
            for raw in raw_rows:
                record = build_sft_rows_from_raw(raw)
                if record:
                    sft_rows.extend(record)
                else:
                    print(f"Warning: Failed to build SFT row for record ID: {raw.get('id')}")

            out_path = os.path.join(out_dir, f"{split}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for row in sft_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"[{split}] [size:{len(sft_rows)}] Completed writing HF formatted SFT dataset to {out_dir}")

    except Exception as e:
        print(f"An error occurred while creating HF formatted dataset for language {lang}: {e}")


def build_save_hf_formatted_dpo_dataset(lang: str, dir_name:str = "dpo"):
    try:
        combined_dir = os.path.join(logistics.project_root_dir, logistics.combined_data_dir, dir_name, lang)
        splits = logistics.splits
        # out dir to save file
        out_dir = os.path.join(logistics.project_root_dir, logistics.processed_data_dir, "dpo", lang)
        os.makedirs(out_dir, exist_ok=True)
        for split in splits:
            split_path = os.path.join(combined_dir, f"{split}.jsonl")
            if not os.path.exists(split_path):
                print(f"Missing split file: {split_path}")
                out_path = os.path.join(out_dir, f"{split}.jsonl")
                with open(out_path, "w", encoding="utf-8") as f:
                    pass
                print(f"[{split}] [size:0] Completed writing HF formatted DPO dataset to {out_dir}")
                continue

            rows = read_jsonl(split_path)
            out_path = os.path.join(out_dir, f"{split}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for row in rows:
                    prompt_text = row.get("prompt", "")
                    if isinstance(prompt_text, str):
                        prompt_only = prompt_text
                        caption_only = ""
                        marker = "\nCAPTION:"
                        if marker in prompt_text:
                            prompt_only, caption_only = prompt_text.split(marker, 1)
                        row["prompt"] = prompt_only.strip()
                        row["caption"] = " ".join(caption_only.split())
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"[{split}] [size:{len(rows)}] Completed writing HF formatted DPO dataset to {out_dir}")
    except Exception as e:
        print(f"An error occurred while creating HF formatted DPO dataset for language {lang}: {e}")


#filter keep only records for dpo dataest.
#run after combined dataset is create and before hf formatting.
def filter_dpo_ds_for_keep_records(lang: str):
    try:
        backups: dict[str, str] = {}
        for split in logistics.splits:
            dpo_path = os.path.join(
                logistics.project_root_dir,
                logistics.combined_data_dir,
                "dpo",
                lang,
                f"{split}.jsonl",
            )
            sft_path = os.path.join(
                logistics.project_root_dir,
                logistics.combined_data_dir,
                "sft",
                lang,
                f"{split}.jsonl",
            )

            if not os.path.exists(dpo_path):
                print(f"Missing DPO split file: {dpo_path}")
                continue
            if not os.path.exists(sft_path):
                print(f"Missing SFT split file: {sft_path}")
                continue

            sft_rows = read_jsonl(sft_path)
            keep_ids = {row.get("id") for row in sft_rows if row.get("id")}

            dpo_rows = read_jsonl(dpo_path)
            filtered_rows = [row for row in dpo_rows if row.get("id") in keep_ids]

            backup_path = f"{dpo_path}.bak"
            shutil.copyfile(dpo_path, backup_path)
            backups[dpo_path] = backup_path

            with open(dpo_path, "w", encoding="utf-8") as f:
                for row in filtered_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(
                f"[{lang}] [{split}] kept {len(filtered_rows)}/{len(dpo_rows)} "
                f"records based on SFT keep ids"
            )
            try:
                os.remove(backup_path)
            except Exception as exc:
                print(f"Warning: failed to remove backup {backup_path}: {exc}")
    except Exception as e:
        for original_path, backup_path in backups.items():
            try:
                shutil.copyfile(backup_path, original_path)
            except Exception as restore_exc:
                print(f"Failed to restore {original_path} from backup: {restore_exc}")
        print(f"Error filtering DPO dataset for {lang}. Restore complete. Please start again. Error: {e}")
        raise


def publish_sft_dataset_to_hf():
    try:
        features = get_hf_sft_features()
        for lang in logistics.langs:
            out_dir = os.path.join(logistics.project_root_dir, logistics.processed_data_dir, "sft", lang)
            datasets_by_split: dict[str, Dataset] = {}
            for split in logistics.splits:
                split_path = os.path.join(out_dir, f"{split}.jsonl")
                if not os.path.exists(split_path):
                    print(f"Missing split file: {split_path}")
                    datasets_by_split[split] = Dataset.from_dict({k: [] for k in features.keys()}, features=features)
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
                logistics.hf_sft_ds_id,
                config_name=lang,
                token=resolved["hf_token"],
            )
            print(f"Published SFT dataset for {lang} to {logistics.hf_sft_ds_id}")
    except Exception as e: 
        print(f"An error occurred while publishing SFT datasets to HF: {e}")





# if __name__ == "__main__":
    #build_create_combined_sft(langs=['en', 'zh'])
    #build_create_combined_dpo(langs=['en', 'zh'])
    # for lang in logistics.langs:
    #     print(f"Processing SFT dataset for language: {lang}")
    #     build_save_hf_formatted_sft_dataset(lang)
        #print(f"Processing DPO dataset for language: {lang}")
        #build_save_hf_formatted_dpo_dataset(lang)
    # publish_sft_dataset_to_hf()



# Note: for DPO dataset, run filter_dpo_ds_for_keep_records(lang) 
# after combined dataset is created. (Combined -> processed/dpo (keep reocrds) )
# then use dpo_creation clean the records (remove hastags, textual cues)
