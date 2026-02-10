from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from config.logistics import Logistics

"""
RUN THIS SCRIPT TO CLEAN DPO RECORDS IN-PLACE. 
It will back up the original split files as .bak before writing cleaned versions, and attempt to restore.
only after once we have the processed DPO files from combined dir. 
"""


TAG_RE = re.compile(r"[#@]([A-Za-z0-9_-]+)")
WORD_RE = re.compile(r"[A-Za-z0-9_-]+")
MENTION_RE = re.compile(r"@([A-Za-z0-9_-]+)")
SARCASTIC_TAGS = {
    "sarcasm",
    "sarcastic",
    "sarcasmonly",
    "sarcasme",
    "sarcasmintended",
    "sarcasmintent",
    "sarcasmquote",
}
OTHER_TAGS = {
    "meme",
    "memes",
    "funny",
    "hilarious",
    "lol",
    "lmao",
    "comedy",
    "joke",
    "humor",
    "funnypics",
    "sarcasticquotes",
}
PLAIN_TAG_TOKENS = {
    "sarcasm",
    "sarcastic",
    "sarcasme",
    "sarcasmintended",
    "sarcasmintent",
    "sarcasmquote",
    "sarcasmonly",
    "meme",
    "memes",
    "funny",
    "hilarious",
    "lol",
    "lmao",
    "comedy",
    "joke",
    "humor",
    "funnypics",
    "sarcasticquotes",
}


def _normalize_tag(tag: str) -> str:
    return re.sub(r"[^a-z0-9]", "", tag.lower())


def _looks_like_tag_token(token: str) -> bool:
    if "_" in token or "-" in token:
        return True
    return any(ch.isupper() for ch in token) and any(ch.islower() for ch in token)


def _extract_tag_tokens(text: str) -> List[str]:
    tokens = set(TAG_RE.findall(text))
    for word in WORD_RE.findall(text):
        if _looks_like_tag_token(word):
            tokens.add(word)
        else:
            normalized = _normalize_tag(word)
            if normalized in PLAIN_TAG_TOKENS:
                tokens.add(word)
    return list(tokens)


def _has_sarcasm_tag(text: str) -> bool:
    tags = _extract_tag_tokens(text)
    normalized = {_normalize_tag(tag) for tag in tags}
    return any(tag in SARCASTIC_TAGS for tag in normalized)


def extract_visual_facts(chosen_text: str) -> List[str]:
    start_marker = "[VISUAL_FACTS]"
    end_marker = "[TEXT_LITERAL]"
    if start_marker not in chosen_text:
        return []
    section = chosen_text.split(start_marker, 1)[1]
    if end_marker in section:
        section = section.split(end_marker, 1)[0]
    facts: List[str] = []
    for line in section.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line.lstrip("-").strip()
        facts.append(line)
    return facts


def parse_caption(caption: str) -> Dict[str, Any]:
    tags = _extract_tag_tokens(caption)
    normalized = {_normalize_tag(tag) for tag in tags}

    had_sarcasm_tag = any(tag in SARCASTIC_TAGS for tag in normalized)
    had_other_tags = any(tag in OTHER_TAGS for tag in normalized)

    cleaned = caption
    if had_sarcasm_tag or had_other_tags:
        cleaned = TAG_RE.sub("", cleaned)

    def _mask_mentions(match: re.Match[str]) -> str:
        token = match.group(1)
        if _normalize_tag(token) in SARCASTIC_TAGS:
            return ""
        return "@HANDLE"

    cleaned = MENTION_RE.sub(_mask_mentions, cleaned)
    cleaned = cleaned.replace("#", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    keep_record = len(cleaned.strip()) >= 10
    return {
        "cleaned_text": cleaned,
        "had_sarcasm_tag": had_sarcasm_tag,
        "had_other_tags": had_other_tags,
        "keep_record": keep_record,
    }


def mask_non_sarcasm_handles(text: str) -> str:
    def _mask_mentions(match: re.Match[str]) -> str:
        token = match.group(1)
        if _normalize_tag(token) in SARCASTIC_TAGS:
            return ""
        return "@HANDLE"

    return MENTION_RE.sub(_mask_mentions, text)


def _has_disallowed_tags(text: str) -> bool:
    tags = _extract_tag_tokens(text)
    normalized = {_normalize_tag(tag) for tag in tags}
    return any(tag in SARCASTIC_TAGS or tag in OTHER_TAGS for tag in normalized)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"Skipping invalid JSON in {path} at line {line_num}: {exc}")
    except Exception as exc:
        print(f"Failed to read {path}: {exc}")
    return records


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for rec in records:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Failed to write {path}: {exc}")


def clean_dpo_records() -> None:
    """Clean DPO records in-place with per-split backups and no cross-split mixing."""
    logistics = Logistics()
    src_root = Path(logistics.project_root_dir) / logistics.processed_data_dir / "dpo"

    for lang in logistics.langs:
        for split in logistics.splits:
            src_path = src_root / lang / f"{split}.jsonl"
            if not src_path.exists():
                print(f"Missing split file: {src_path}")
                continue
            print(f"Processing {src_path}...")
            records = _read_jsonl(src_path)
            cleaned_records: List[Dict[str, Any]] = []
            dropped_empty = 0
            dropped_tagged = 0
            dropped_sarcasm = 0
            for record in records:
                chosen = record.get("chosen", "")
                if isinstance(chosen, str):
                    chosen = mask_non_sarcasm_handles(chosen)
                visual_facts_text = ""
                if isinstance(chosen, str):
                    visual_facts_list = extract_visual_facts(chosen)
                    visual_facts_text = " ".join(visual_facts_list)
                if visual_facts_text and _has_disallowed_tags(visual_facts_text):
                    dropped_tagged += 1
                    continue
                if isinstance(chosen, str):
                    record["chosen"] = chosen

                caption = record.get("caption", "")
                if not isinstance(caption, str):
                    caption = str(caption)

                parsed = parse_caption(caption)
                if _has_sarcasm_tag(caption) or _has_sarcasm_tag(visual_facts_text):
                    dropped_sarcasm += 1
                    continue
                if not parsed["keep_record"]:
                    dropped_empty += 1
                    continue

                record["caption"] = parsed["cleaned_text"]
                if isinstance(chosen, str):
                    record["visual_facts"] = visual_facts_list
                cleaned_records.append(record)

            out_path = src_root / lang / f"{split}.jsonl"
            backup_path = out_path.with_suffix(out_path.suffix + ".bak")
            try:
                shutil.copyfile(out_path, backup_path)
                _write_jsonl(out_path, cleaned_records)
                try:
                    backup_path.unlink()
                except Exception as exc:
                    print(f"Warning: failed to remove backup {backup_path}: {exc}")
            except Exception as exc:
                try:
                    if backup_path.exists():
                        shutil.copyfile(backup_path, out_path)
                except Exception as restore_exc:
                    print(f"Failed to restore {out_path} from backup: {restore_exc}")
                raise exc
            print(
                f"{lang}/{split} dropped {dropped_empty} records due to empty caption after cleaning, "
                f"{dropped_tagged} records due to tags in chosen, "
                f"and {dropped_sarcasm} records due to sarcasm tags in caption and chosen"
            )





def main():
    clean_dpo_records()

# if __name__ == "__main__":  
#     main()
