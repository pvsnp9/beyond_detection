## DPO Data Filtering

The cleaning pipeline iterates each language/split JSONL and filters records to remove tag leakage and low-signal captions. It masks non-sarcasm @mentions, extracts visual facts from the chosen text, and drops any record whose visual facts contain sarcasm/other tags. Captions are parsed to strip hashtag/tag tokens, mask non-sarcasm handles, and normalize whitespace; records are dropped if sarcasm tags appear in the original caption or visual facts, or if the cleaned caption is shorter than 10 characters. For kept records, the cleaned caption and extracted visual_facts are written back, with per-split backups taken during rewrite.

## SFT data filteration

SFT rows are built from the combined SFT JSONL via `build_save_hf_formatted_sft_dataset`, which calls `build_sft_rows_from_raw` and applies `clean_sft_record` before emitting tasks. For English records, `clean_sft_record` masks non-sarcasm handles, strips hashtag tokens and tag-like words (including variants of sarcasm/meme/funny cues), removes `#` characters, and normalizes whitespace; captions shorter than 5 characters after cleaning are dropped. It also drops any record whose visual facts contain handles or disallowed tag tokens, and rewrites `teacher.visual_facts` with cleaned entries. Teacher fields (`text_literal`, `incongruity`, `explanation`) are similarly de-tagged and handle-masked to avoid textual leakage. The cleaned record is then serialized into SFT rows, with `visual_facts` propagated into the base fields for downstream HF formatting.


**main mDPO*
quality_flags = ["None", "requires_world_knowledge", "multiple_possible_incongruities"]
OCR = ["possible_ocr_text_in_image", ""]
undecided = ["ambiguous_sarcasm", "low_image_clarity" "possible_identity_inference_risk"]
downweight = ["caption_not_sarcastic_or_unclear", ""]
