## DPO Data Filtering

The cleaning pipeline iterates each language/split JSONL and filters records to remove tag leakage and low-signal captions. It masks non-sarcasm @mentions, extracts visual facts from the chosen text, and drops any record whose visual facts contain sarcasm/other tags. Captions are parsed to strip hashtag/tag tokens, mask non-sarcasm handles, and normalize whitespace; records are dropped if sarcasm tags appear in the original caption or visual facts, or if the cleaned caption is shorter than 10 characters. For kept records, the cleaned caption and extracted visual_facts are written back, with per-split backups taken during rewrite.

## SFT data filteration

SFT rows are built from the combined SFT JSONL via `build_save_hf_formatted_sft_dataset`, which calls `build_sft_rows_from_raw` and applies `clean_sft_record` before emitting tasks. For English records, `clean_sft_record` masks non-sarcasm handles, strips hashtag tokens and tag-like words (including variants of sarcasm/meme/funny cues), removes `#` characters, and normalizes whitespace; captions shorter than 5 characters after cleaning are dropped. It also drops any record whose visual facts contain handles or disallowed tag tokens, and rewrites `teacher.visual_facts` with cleaned entries. Teacher fields (`text_literal`, `incongruity`, `explanation`) are similarly de-tagged and handle-masked to avoid textual leakage. The cleaned record is then serialized into SFT rows, with `visual_facts` propagated into the base fields for downstream HF formatting.


**main mDPO*
quality_flags = ["None", "requires_world_knowledge", "multiple_possible_incongruities"]
OCR = ["possible_ocr_text_in_image", ""]
undecided = ["ambiguous_sarcasm", "low_image_clarity" "possible_identity_inference_risk"]
downweight = ["caption_not_sarcastic_or_unclear", ""]


## Report Smoke test
Dataset size: 33982
dataset_loaded config_name=en
dataset_id=alita9/beyond_sarcasm_detection_sft
dataset_lang=en
samples_loaded=4
sample_idx=0 sample_id=1000
sample_idx=1 sample_id=1000
sample_idx=2 sample_id=1001
sample_idx=3 sample_id=1001

=== COLLATOR SMOKE ===
model_id=CohereLabs/aya-vision-8b
output_keys=['attention_mask', 'input_ids', 'labels', 'pixel_values']
input_ids_shape=(4, 1468)
attention_mask_shape=(4, 1468)
labels_shape=(4, 1468)
supervised_tokens_total=754
supervised_tokens_per_example=[51, 269, 51, 383]
pad_after_supervised_per_example=[False, False, False, False]
supervised_tokens_decoded_per_example=
[0] <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{"label": "sarcastic", "need_explanation": false, "visual_facts": [], "evidence_fact_ids": [], "text_literal": "", "incongruity": "", "explanation": ""}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>
[1] <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{"label": "sarcastic", "need_explanation": true, "visual_facts": [{"id": 1, "fact": "A stylized cartoon animal character is shown in side profile facing right."}, {"id": 2, "fact": "The character wears an orange knit beanie and dark sunglasses."}, {"id": 3, "fact": "The character wears a black hoodie."}, {"id": 4, "fact": "The character's tongue is sticking out."}, {"id": 5, "fact": "The background is a solid purple color."}], "evidence_fact_ids": [1, 2, 3], "text_literal": "The speaker asks someone to show they are bragging without saying it directly, then states they minted three @HANDLE items and tags NFT-related hashtags.", "incongruity": "The caption describes bragging about minting NFTs, but the image only shows a generic cartoon character with no visible NFTs, text, or minting-related context.", "explanation": "The post uses a random \"cool\" avatar image as a stand-in for bragging; the amusement is that they claim they're not explicitly flexing while they very clearly are, and the image doesn't actually provide proof—just attitude."}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>
[2] <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{"label": "sarcastic", "need_explanation": false, "visual_facts": [], "evidence_fact_ids": [], "text_literal": "", "incongruity": "", "explanation": ""}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>
[3] <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{"label": "sarcastic", "need_explanation": true, "visual_facts": [{"id": 1, "fact": "The image is a collage of multiple social media posts/screenshots about Pride parades."}, {"id": 2, "fact": "Several photos show crowds outdoors on city streets during what appears to be Pride events (rainbow flags and colorful outfits are visible)."}, {"id": 3, "fact": "Some images depict nude or partially nude adults with pixelation/blur blocks covering genitals and/or breasts."}, {"id": 4, "fact": "One screenshot includes a video player interface with a view count and a short duration bar."}, {"id": 5, "fact": "Multiple posts contain text discussing \"Pride Parade,\" \"Toronto,\" \"Seattle,\" \"kids/children,\" and \"naked\" (exact wording varies across screenshots)."}, {"id": 6, "fact": "One image shows a person wearing a bunny mask in a street crowd (as part of a post about a Pride parade)."}, {"id": 7, "fact": "Some screenshots show usernames, dates, and platform UI elements typical of Twitter/social media."}], "evidence_fact_ids": [1, 2, 3], "text_literal": "The caption asks why Franklin residents would think a Pride festival could lead to that kind of behavior.", "incongruity": "The caption implies skepticism that a Pride festival would lead to inappropriate public behavior, but the collage is presented as \"evidence\" of nude/sexualized behavior occurring at Pride parades in public settings.", "explanation": "The contradiction is that the question pretends it's unreasonable for Franklin residents to worry, while the accompanying collage is meant to suggest those worries are justified because it shows/claims public nudity at Pride events."}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>
image_tensor_keys=['pixel_values']
pixel_values_shape=(4, 3, 364, 364) dtype=torch.float32

=== COLLATOR SMOKE ===
model_id=meta-llama/Llama-3.2-11B-Vision-Instruct
output_keys=['aspect_ratio_ids', 'aspect_ratio_mask', 'attention_mask', 'cross_attention_mask', 'input_ids', 'labels', 'pixel_values']
input_ids_shape=(4, 791)
attention_mask_shape=(4, 791)
labels_shape=(4, 791)
supervised_tokens_total=750
supervised_tokens_per_example=[47, 273, 47, 383]
pad_after_supervised_per_example=[True, True, True, False]
supervised_tokens_decoded_per_example=
[0] {"label": "sarcastic", "need_explanation": false, "visual_facts": [], "evidence_fact_ids": [], "text_literal": "", "incongruity": "", "explanation": ""}<|eot_id|>
[1] {"label": "sarcastic", "need_explanation": true, "visual_facts": [{"id": 1, "fact": "A stylized cartoon animal character is shown in side profile facing right."}, {"id": 2, "fact": "The character wears an orange knit beanie and dark sunglasses."}, {"id": 3, "fact": "The character wears a black hoodie."}, {"id": 4, "fact": "The character's tongue is sticking out."}, {"id": 5, "fact": "The background is a solid purple color."}], "evidence_fact_ids": [1, 2, 3], "text_literal": "The speaker asks someone to show they are bragging without saying it directly, then states they minted three @HANDLE items and tags NFT-related hashtags.", "incongruity": "The caption describes bragging about minting NFTs, but the image only shows a generic cartoon character with no visible NFTs, text, or minting-related context.", "explanation": "The post uses a random \"cool\" avatar image as a stand-in for bragging; the amusement is that they claim they're not explicitly flexing while they very clearly are, and the image doesn't actually provide proof—just attitude."}<|eot_id|>
[2] {"label": "sarcastic", "need_explanation": false, "visual_facts": [], "evidence_fact_ids": [], "text_literal": "", "incongruity": "", "explanation": ""}<|eot_id|>
[3] {"label": "sarcastic", "need_explanation": true, "visual_facts": [{"id": 1, "fact": "The image is a collage of multiple social media posts/screenshots about Pride parades."}, {"id": 2, "fact": "Several photos show crowds outdoors on city streets during what appears to be Pride events (rainbow flags and colorful outfits are visible)."}, {"id": 3, "fact": "Some images depict nude or partially nude adults with pixelation/blur blocks covering genitals and/or breasts."}, {"id": 4, "fact": "One screenshot includes a video player interface with a view count and a short duration bar."}, {"id": 5, "fact": "Multiple posts contain text discussing \"Pride Parade,\" \"Toronto,\" \"Seattle,\" \"kids/children,\" and \"naked\" (exact wording varies across screenshots)."}, {"id": 6, "fact": "One image shows a person wearing a bunny mask in a street crowd (as part of a post about a Pride parade)."}, {"id": 7, "fact": "Some screenshots show usernames, dates, and platform UI elements typical of Twitter/social media."}], "evidence_fact_ids": [1, 2, 3], "text_literal": "The caption asks why Franklin residents would think a Pride festival could lead to that kind of behavior.", "incongruity": "The caption implies skepticism that a Pride festival would lead to inappropriate public behavior, but the collage is presented as \"evidence\" of nude/sexualized behavior occurring at Pride parades in public settings.", "explanation": "The contradiction is that the question pretends it's unreasonable for Franklin residents to worry, while the accompanying collage is meant to suggest those worries are justified because it shows/claims public nudity at Pride events."}<|eot_id|>
image_tensor_keys=['pixel_values']
pixel_values_shape=(4, 1, 4, 3, 560, 560) dtype=torch.float32

=== COLLATOR SMOKE ===
model_id=google/gemma-3-12b-it
output_keys=['attention_mask', 'input_ids', 'labels', 'pixel_values', 'token_type_ids']
input_ids_shape=(4, 1006)
attention_mask_shape=(4, 1006)
labels_shape=(4, 1006)
supervised_tokens_total=739
supervised_tokens_per_example=[48, 265, 48, 378]
pad_after_supervised_per_example=[False, False, False, False]
supervised_tokens_decoded_per_example=
[0] {"label": "sarcastic", "need_explanation": false, "visual_facts": [], "evidence_fact_ids": [], "text_literal": "", "incongruity": "", "explanation": ""}<end_of_turn>

[1] {"label": "sarcastic", "need_explanation": true, "visual_facts": [{"id": 1, "fact": "A stylized cartoon animal character is shown in side profile facing right."}, {"id": 2, "fact": "The character wears an orange knit beanie and dark sunglasses."}, {"id": 3, "fact": "The character wears a black hoodie."}, {"id": 4, "fact": "The character's tongue is sticking out."}, {"id": 5, "fact": "The background is a solid purple color."}], "evidence_fact_ids": [1, 2, 3], "text_literal": "The speaker asks someone to show they are bragging without saying it directly, then states they minted three @HANDLE items and tags NFT-related hashtags.", "incongruity": "The caption describes bragging about minting NFTs, but the image only shows a generic cartoon character with no visible NFTs, text, or minting-related context.", "explanation": "The post uses a random \"cool\" avatar image as a stand-in for bragging; the amusement is that they claim they're not explicitly flexing while they very clearly are, and the image doesn't actually provide proof—just attitude."}<end_of_turn>

[2] {"label": "sarcastic", "need_explanation": false, "visual_facts": [], "evidence_fact_ids": [], "text_literal": "", "incongruity": "", "explanation": ""}<end_of_turn>

[3] {"label": "sarcastic", "need_explanation": true, "visual_facts": [{"id": 1, "fact": "The image is a collage of multiple social media posts/screenshots about Pride parades."}, {"id": 2, "fact": "Several photos show crowds outdoors on city streets during what appears to be Pride events (rainbow flags and colorful outfits are visible)."}, {"id": 3, "fact": "Some images depict nude or partially nude adults with pixelation/blur blocks covering genitals and/or breasts."}, {"id": 4, "fact": "One screenshot includes a video player interface with a view count and a short duration bar."}, {"id": 5, "fact": "Multiple posts contain text discussing \"Pride Parade,\" \"Toronto,\" \"Seattle,\" \"kids/children,\" and \"naked\" (exact wording varies across screenshots)."}, {"id": 6, "fact": "One image shows a person wearing a bunny mask in a street crowd (as part of a post about a Pride parade)."}, {"id": 7, "fact": "Some screenshots show usernames, dates, and platform UI elements typical of Twitter/social media."}], "evidence_fact_ids": [1, 2, 3], "text_literal": "The caption asks why Franklin residents would think a Pride festival could lead to that kind of behavior.", "incongruity": "The caption implies skepticism that a Pride festival would lead to inappropriate public behavior, but the collage is presented as \"evidence\" of nude/sexualized behavior occurring at Pride parades in public settings.", "explanation": "The contradiction is that the question pretends it's unreasonable for Franklin residents to worry, while the accompanying collage is meant to suggest those worries are justified because it shows/claims public nudity at Pride events."}<end_of_turn>

image_tensor_keys=['pixel_values']
pixel_values_shape=(4, 3, 896, 896) dtype=torch.float32

=== COLLATOR SMOKE ===
model_id=Qwen/Qwen3-VL-8B-Instruct
output_keys=['attention_mask', 'image_grid_thw', 'input_ids', 'labels', 'pixel_values']
input_ids_shape=(4, 1172)
attention_mask_shape=(4, 1172)
labels_shape=(4, 1172)
supervised_tokens_total=754
supervised_tokens_per_example=[48, 274, 48, 384]
pad_after_supervised_per_example=[True, True, True, False]
supervised_tokens_decoded_per_example=
[0] {"label": "sarcastic", "need_explanation": false, "visual_facts": [], "evidence_fact_ids": [], "text_literal": "", "incongruity": "", "explanation": ""}<|im_end|>

[1] {"label": "sarcastic", "need_explanation": true, "visual_facts": [{"id": 1, "fact": "A stylized cartoon animal character is shown in side profile facing right."}, {"id": 2, "fact": "The character wears an orange knit beanie and dark sunglasses."}, {"id": 3, "fact": "The character wears a black hoodie."}, {"id": 4, "fact": "The character's tongue is sticking out."}, {"id": 5, "fact": "The background is a solid purple color."}], "evidence_fact_ids": [1, 2, 3], "text_literal": "The speaker asks someone to show they are bragging without saying it directly, then states they minted three @HANDLE items and tags NFT-related hashtags.", "incongruity": "The caption describes bragging about minting NFTs, but the image only shows a generic cartoon character with no visible NFTs, text, or minting-related context.", "explanation": "The post uses a random \"cool\" avatar image as a stand-in for bragging; the amusement is that they claim they're not explicitly flexing while they very clearly are, and the image doesn't actually provide proof—just attitude."}<|im_end|>

[2] {"label": "sarcastic", "need_explanation": false, "visual_facts": [], "evidence_fact_ids": [], "text_literal": "", "incongruity": "", "explanation": ""}<|im_end|>

[3] {"label": "sarcastic", "need_explanation": true, "visual_facts": [{"id": 1, "fact": "The image is a collage of multiple social media posts/screenshots about Pride parades."}, {"id": 2, "fact": "Several photos show crowds outdoors on city streets during what appears to be Pride events (rainbow flags and colorful outfits are visible)."}, {"id": 3, "fact": "Some images depict nude or partially nude adults with pixelation/blur blocks covering genitals and/or breasts."}, {"id": 4, "fact": "One screenshot includes a video player interface with a view count and a short duration bar."}, {"id": 5, "fact": "Multiple posts contain text discussing \"Pride Parade,\" \"Toronto,\" \"Seattle,\" \"kids/children,\" and \"naked\" (exact wording varies across screenshots)."}, {"id": 6, "fact": "One image shows a person wearing a bunny mask in a street crowd (as part of a post about a Pride parade)."}, {"id": 7, "fact": "Some screenshots show usernames, dates, and platform UI elements typical of Twitter/social media."}], "evidence_fact_ids": [1, 2, 3], "text_literal": "The caption asks why Franklin residents would think a Pride festival could lead to that kind of behavior.", "incongruity": "The caption implies skepticism that a Pride festival would lead to inappropriate public behavior, but the collage is presented as \"evidence\" of nude/sexualized behavior occurring at Pride parades in public settings.", "explanation": "The contradiction is that the question pretends it's unreasonable for Franklin residents to worry, while the accompanying collage is meant to suggest those worries are justified because it shows/claims public nudity at Pride events."}<|im_end|>

image_tensor_keys=['image_grid_thw', 'pixel_values']
image_grid_thw_shape=(4, 3) dtype=torch.int64
pixel_values_shape=(4496, 1536) dtype=torch.float32
Dataset size: 33982





## Data stats 


=== MAXLEN STATS ===
model_id=Qwen/Qwen3-VL-8B-Instruct
split=train lang=en
model_max_length=None max_len_used=1536
total_examples=33982
max_full_length=2291
trimmed_count=6098
bad_image_counts={}
length_bins={'<=1024': 11990, '>1024<=1536': 15894, '>1536<=2048': 5906, '>2048': 192}
Dataset size: 2276



=== MAXLEN STATS ===
model_id=CohereLabs/aya-vision-8b
split=train lang=en
model_max_length=None max_len_used=1536
total_examples=33982
max_full_length=0
trimmed_count=0
bad_image_counts={}
length_bins={'<=1024': 0, '>1024<=1536': 0, '>1536<=2048': 0, '>2048': 0}

=== MAXLEN STATS ===
model_id=meta-llama/Llama-3.2-11B-Vision-Instruct
split=train lang=en
model_max_length=None max_len_used=1536
total_examples=33982
max_full_length=1088
trimmed_count=0
bad_image_counts={}
length_bins={'<=1024': 33980, '>1024<=1536': 2, '>1536<=2048': 0, '>2048': 0}

=== MAXLEN STATS ===
model_id=google/gemma-3-12b-it
split=train lang=en
model_max_length=None max_len_used=1536
total_examples=33982
max_full_length=1181
trimmed_count=0
bad_image_counts={}
length_bins={'<=1024': 33680, '>1024<=1536': 302, '>1536<=2048': 0, '>2048': 0}






=== MAXLEN STATS ===
model_id=meta-llama/Llama-3.2-11B-Vision-Instruct
split=train lang=zh
model_max_length=None max_len_used=1536
total_examples=2276
max_full_length=1588
trimmed_count=1
bad_image_counts={}
length_bins={'<=1024': 2231, '>1024<=1536': 44, '>1536<=2048': 1, '>2048': 0}

=== MAXLEN STATS ===
model_id=google/gemma-3-12b-it
split=train lang=zh
model_max_length=None max_len_used=1536
trimmed_decode_failed id=427 err=Gemma3 collate failed: target tokens not found in input_ids
total_examples=2276
max_full_length=1602
trimmed_count=1
bad_image_counts={}
length_bins={'<=1024': 2047, '>1024<=1536': 228, '>1536<=2048': 1, '>2048': 0}

=== MAXLEN STATS ===
model_id=Qwen/Qwen3-VL-8B-Instruct
split=train lang=zh
model_max_length=None max_len_used=1536
total_examples=2276
max_full_length=2370
trimmed_count=521
bad_image_counts={}
length_bins={'<=1024': 1107, '>1024<=1536': 648, '>1536<=2048': 474, '>2048': 47}
Dataset size: 3688




max_full_length=0
trimmed_count=0
bad_image_counts={}
length_bins={'<=1024': 0, '>1024<=1536': 0, '>1536<=2048': 0, '>2048': 0}


=== MAXLEN STATS ===
model_id=Qwen/Qwen3-VL-8B-Instruct
split=test lang=zh
model_max_length=None max_len_used=1536
total_examples=712
max_full_length=2809
trimmed_count=161
bad_image_counts={}
length_bins={'<=1024': 333, '>1024<=1536': 218, '>1536<=2048': 139, '>2048': 22}

=== MAXLEN STATS ===
model_id=CohereLabs/aya-vision-8b
split=test lang=zh
model_max_length=None max_len_used=1536
total_examples=712
max_full_length=0
trimmed_count=0
bad_image_counts={}
length_bins={'<=1024': 0, '>1024<=1536': 0, '>1536<=2048': 0, '>2048': 0}

=== MAXLEN STATS ===
model_id=meta-llama/Llama-3.2-11B-Vision-Instruct
split=test lang=zh
model_max_length=None max_len_used=1536
total_examples=712
max_full_length=1727
trimmed_count=1
bad_image_counts={}
length_bins={'<=1024': 689, '>1024<=1536': 22, '>1536<=2048': 1, '>2048': 0}

=== MAXLEN STATS ===
model_id=google/gemma-3-12b-it
split=test lang=zh
model_max_length=None max_len_used=1536
trimmed_decode_failed id=3079 err=Gemma3 collate failed: target tokens not found in input_ids
trimmed_decode_failed id=3260 err=Gemma3 collate failed: target tokens not found in input_ids
total_examples=712
max_full_length=1742
trimmed_count=2
bad_image_counts={}
length_bins={'<=1024': 627, '>1024<=1536': 83, '>1536<=2048': 2, '>2048': 0}

