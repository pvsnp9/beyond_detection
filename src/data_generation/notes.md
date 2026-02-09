## DPO Data Filtering

The cleaning pipeline iterates each language/split JSONL and filters records to remove tag leakage and low-signal captions. It masks non-sarcasm @mentions, extracts visual facts from the chosen text, and drops any record whose visual facts contain sarcasm/other tags. Captions are parsed to strip hashtag/tag tokens, mask non-sarcasm handles, and normalize whitespace; records are dropped if sarcasm tags appear in the original caption or visual facts, or if the cleaned caption is shorter than 10 characters. For kept records, the cleaned caption and extracted visual_facts are written back, with per-split backups taken during rewrite.


**main mDPO*
quality_flags = ["None", "requires_world_knowledge", "multiple_possible_incongruities"]
OCR = ["possible_ocr_text_in_image", ""]
undecided = ["ambiguous_sarcasm", "low_image_clarity" "possible_identity_inference_risk"]
downweight = ["caption_not_sarcastic_or_unclear", ""]