
# Codex Agent Instruction (NEW): Implement Hallucination + Explanation-Quality Evaluation Pipeline

You are a coding agent. Implement a production-grade evaluation pipeline under:

`src/eva/nlp_hall/`

```
src/eva/nlp_hall/
**init**.py
io.py                 # parse + sanitize JSON (no dataclasses here)
normalize.py          # text normalization helpers
similarity.py         # embedding + cosine + optional NLI stub
matching.py           # bipartite matching (1-1)
claims.py             # claim extraction from explanation
fact_metrics.py       # TP/FP/FN, Add/Omit, P/R/F1
expl_metrics.py       # EG/EH/EC/EQS
aggregate.py          # macro/micro, groupby slices
run_hall_eval.py      # CLI entrypoint

````

No dataclasses/condig should be declared inside `src/eva/nlp_hall/`.

---

## Example Record Behavior (must be handled correctly)
Records have `target_json` and `output` as JSON-encoded strings. The record-level fields can conflict with the model prediction fields:

- record-level `modality` can be `"both"` even if `output.missing_modalities` claims `"text"` is missing.
- images may contain OCR-like text and this may appear inside `visual_facts` (e.g., “top text reads …”).
- `quality_flags` is a list of strings.
- `id` appears at record-level (not inside target_json).

Your code must parse and evaluate robustly without assuming perfect consistency between:
- record fields vs output JSON fields
- target_json vs output

---

## Goal
Given JSONL records with `target_json` and `output`, compute:

1) **Fact-level hallucination** from `visual_facts`:
- 1-to-1 semantic alignment → TP/FP/FN
- Add/Omit/Precision/Recall/F1 (piecewise definitions)

2) **Explanation quality** from `output.explanation`:
- claim extraction
- EG/EH/EC/EQS (per the paper section)

3) **Aggregation**:
- macro averages
- micro (facts): sum TP/FP/FN then compute P/R/F1
- slice/groupby: `model_key`, `source`, `modality`, and `quality_flags` (explode flags)

4) **CLI**:
- `run_hall_eval.py` reads JSONL, runs evaluation, writes outputs.

---

## Constraints
- Must be robust to bad inputs (parse failures, missing keys, wrong types).
- Must not crash on a single bad record: log + continue.
- Deterministic by default (seed where relevant).
- Efficient for large datasets: batch embedding + caching.
- Keep dependencies minimal; avoid heavy frameworks unless already present.
- Use Python typing everywhere.
- Return metrics consistent with the paper:
  - PRF uses piecewise definitions
  - Add/Omit uses cases
- Never call the network.

---

## 1) Data Contract

Each input JSONL record is a dict with at least:
- `id` (str|int)
- `target_json` (str JSON)
- `output` (str JSON)

Optional keys for slicing (preserve if present):
- `model_key`, `source`, `modality`, `quality_flags`, `gt`, `query`

Both `target_json` and `output` JSON typically contain:
- `visual_facts`: list of dicts like `{"id": int, "fact": "..."}` (id  starts at 1)
- `missing_modalities`: list[str] (optional)
- `explanation`: str (in output)

---

## 2) Config + Dataclasses (MUST be in `config/logistics.py`)

Create/extend dataclasses and config objects so that these imports work:

```python
from config.logistics import (
    HallEvalConfig,
    ParsedRecord,
    FactCounts,
    FactMetrics,
    ExplMetrics,
)


### 2.1 HallEvalConfig

Include at minimum:

* `tau: float = 0.82`
* `tau_e: float = 0.80`
* `tau_c: float = 0.78`
* `backend: str = "sbert"`  # fallback to "tfidf"
* `model_name: str | None = None`
* `batch_size: int = 32`
* `device: str | None = None`
* `match_mode: str = "max_weight"`  # or "max_cardinality" or "greedy"
* `top_k: int | None = None`
* `max_claims: int = 20`
* `min_clause_split_len: int = 120`
* `skip_invalid: bool = False`
* `seed: int = 1234`

### 2.2 ParsedRecord

Must contain:

* `id: str`
* `model_key: str | None`
* `source: str | None`
* `modality: str | None`
* `quality_flags: tuple[str, ...]`
* `gt: str | None`
* `query: str | None`
* `ref_facts: list[str]`
* `pred_facts: list[str]`
* `explanation: str`
* `missing_modalities_pred: tuple[str, ...]`
* `missing_modalities_ref: tuple[str, ...]`
* `is_valid: bool`
* `error: str | None`

### 2.3 FactCounts

* `tp: int`, `fp: int`, `fn: int`
* `n_ref: int`, `n_pred: int`
* `mean_match_sim: float | None` (optional)

### 2.4 FactMetrics

* `add: float`, `omit: float`
* `p: float`, `r: float`, `f1: float`

### 2.5 ExplMetrics

* `eg: float`, `eh: float`, `ec: float`
* `eqs: float | None`
* `k_claims: int`
* `len_tokens: int`

**Important:** Do not redefine these dataclasses elsewhere.

---

## 3) Metric Definitions 

### 3.1 Fact matching + one-to-one alignment

* Similarity score `s(f_hat, f) ∈ [0,1]` from embeddings cosine similarity (default).
* Admissible match if `s >= tau`.
* One-to-one matching over admissible edges (each predicted and reference fact matched at most once).
* Prefer max-weight if possible; otherwise max-cardinality; otherwise greedy (document).

Counts:

* `TP = |M|`
* `FP = |Fhat| - TP`
* `FN = |F| - TP`

### 3.2 Fact metrics (piecewise; EXACT)

Let `n_pred = |Fhat|`, `n_ref = |F|`.

* `Add = FP/n_pred if n_pred>0 else 0`
* `Omit = FN/n_ref if n_ref>0 else 0`
* `P = TP/(TP+FP) if TP+FP>0 else 1`
* `R = TP/(TP+FN) if TP+FN>0 else 1`
* `F1 = 2PR/(P+R) if P+R>0 else 0`

### 3.3 Explanation metrics

Extract `Claims(e_hat) = {c1..cK}` via sentence segmentation and optional clause splitting.

Compute similarity between claims and reference facts:

* Groundedness:

  * claim grounded if `max_f s(c,f) >= tau_e`
  * `EG = grounded/K if K>0 else 1`
  * `EH = 1 - EG`
* Completeness:

  * fact covered if `max_c s(c,f) >= tau_c`
  * `EC = covered/n_ref if n_ref>0 else 1`
* Optional:

  * `Len = log(1 + token_count(e_hat))` where token_count is whitespace split
  * `EQS = (EG*EC)/Len` (if Len==0 set EQS=0 or None; be consistent)

### 3.4 Aggregation

* Macro: mean across examples for each metric (exclude invalid if configured).
* Micro (facts): sum TP/FP/FN then compute micro P/R/F1.
* Also report macro EG/EH/EC/EQS.

---

## 4) Implementation Plan by File (src/eva/nlp_hall)

### 4.1 `__init__.py`

Export public API:

* `evaluate_records(records, cfg) -> list[dict]`
* `evaluate_jsonl(path, cfg) -> pandas.DataFrame`
* `aggregate_all(df) -> dict[str, pandas.DataFrame]`
  Define `__all__`.

Do NOT define dataclasses here.

---

### 4.2 `io.py` — parse + sanitize JSON

Implement:

* `safe_json_loads(x: Any) -> tuple[dict|None, str|None]`
* `parse_record(raw: dict) -> ParsedRecord` (dataclass is imported from `config.logistics`)

Parsing rules:

* Accept `target_json` and `output` as strings or dicts.
* Extract `visual_facts` safely; if missing or malformed treat as empty list.
* Extract `fact` strings from list entries; skip entries without valid `fact`.
* Extract `explanation` from output.
* Extract `missing_modalities` from target and output (default empty).
* Extract metadata fields (id/model_key/source/modality/quality_flags/gt/query).
* On parse error:

  * set `is_valid=False`, populate `error`, keep id/metadata.

Add optional diagnostic:

* `missing_modality_mismatch` (not required for metrics; useful for analysis).

---

### 4.3 `normalize.py`

Implement:

* `normalize_text(s: str) -> str` (lower, strip, collapse whitespace; optional punctuation trim)
* `normalize_list(xs: list[str]) -> list[str]` filter empties

---

### 4.4 `similarity.py`

Implement similarity backend with fallback:

* Prefer SBERT (`sentence-transformers`) if available.
* Fallback to TF-IDF cosine similarity using sklearn.

Expose:

* `get_similarity_backend(cfg: HallEvalConfig) -> SimilarityBackend`
* `SimilarityBackend.similarity_matrix(a: list[str], b: list[str]) -> np.ndarray`

Requirements:

* batch embeddings (cfg.batch_size)
* in-memory cache mapping normalized_text → vector
* deterministic behavior
* ensure `np.float32` matrix

Optional NLI:


---

### 4.5 `matching.py`

Implement one-to-one matching:

Public function:

* `one_to_one_match(S: np.ndarray, tau: float, mode: str, top_k: int|None) -> list[tuple[int,int]]`

Modes:

* `"max_weight"`: use Hungarian if `scipy` is available; else fall back to greedy.
* `"max_cardinality"`: maximize #pairs above tau (greedy acceptable if documented).
* `"greedy"`: sort admissible edges by similarity desc and match if row/col unused.

Support `top_k` (keep only top_k per predicted row among similarities >= tau).

Return matched index pairs. Never match below tau.

---

### 4.6 `claims.py`

Implement:

* `extract_claims(explanation: str, cfg: HallEvalConfig) -> list[str]`

Rules:

* sentence split via regex on `[.!?]`
* optional clause split on conjunctions only if sentence length > cfg.min_clause_split_len
* normalize text
* cap to cfg.max_claims

Deterministic. No LLM calls.

---

### 4.7 `fact_metrics.py`

Implement:

* `compute_fact_counts(ref_facts, pred_facts, sim_backend, cfg) -> FactCounts`
* `compute_fact_metrics(counts: FactCounts) -> FactMetrics`

Steps:

* normalize facts
* compute similarity matrix `S = sim_backend.similarity_matrix(pred, ref)`
* run `one_to_one_match` with cfg.tau/mode/top_k
* compute TP/FP/FN
* compute Add/Omit/P/R/F1 with exact piecewise definitions

Optionally compute `mean_match_sim` over matched pairs (if any).

---

### 4.8 `expl_metrics.py`

Implement:

* `compute_expl_metrics(ref_facts, explanation, sim_backend, cfg) -> ExplMetrics`

Steps:

* normalize ref facts
* extract claims
* compute similarity matrix claims x ref_facts
* EG/EH using tau_e
* EC using tau_c
* EQS using token length

Edge cases must match paper:

* if no claims: EG=1, EH=0
* if no ref facts: EC=1

---

### 4.9 `aggregate.py`

Use pandas if available.

Implement:

* `records_to_dataframe(rows: list[dict]) -> pd.DataFrame`
* `aggregate_macro(df, metrics: list[str], group_cols: list[str]) -> pd.DataFrame`
* `aggregate_micro_facts(df, group_cols: list[str]) -> pd.DataFrame`
* `explode_quality_flags(df) -> pd.DataFrame`
* `aggregate_all(df) -> dict[str, pd.DataFrame]`

Outputs:

* overall
* by model_key
* by source
* by modality
* by model_key+source+modality
* by quality_flag

Micro facts: sum TP/FP/FN then compute micro P/R/F1.

---

### 4.10 `run_hall_eval.py` — CLI entrypoint

Implement argparse CLI:

* `--input`, `--output_dir`

**## **use cinfig for the follwoing**:
* `--tau`, `--tau_e`, `--tau_c`
* `--backend`, `--model_name`
* `--match_mode`, `--top_k`
* `--max_records`, `--skip_invalid`, `--verbose`
* seed from cfg.seed

Behavior:

* stream JSONL
* parse_record
* if invalid: skip if cfg.skip_invalid else write row with error + is_valid=0
* compute fact + explanation metrics
* write:

  * per_example.csv
  * summary_overall.json (macro + micro)
  * by_model.csv, by_modality.csv, by_model_source_modality.csv
  * by_quality_flag.csv (if flags present)

Must print counts: total, valid, parse_failed.

Exit code nonzero only if input missing or no valid rows.

---

## 5) Output Row Schema (per-example)

Each row dict must include:

* identifiers: `id`, `model_key`, `source`, `modality`, `gt`
* `quality_flags` serialized (e.g., `"flag1|flag2"`)
* counts: `tp, fp, fn, n_ref, n_pred`
* fact metrics: `add, omit, p, r, f1`
* expl metrics: `eg, eh, ec, eqs, k_claims, len_tokens`
* parsing: `is_valid`, `error`
* diagnostics: `missing_modalities_pred`, `missing_modalities_ref` serialized (optional)

---

## 6) Quality Gates

Add simple self-checks (no need for pytest unless already used):

* ref=[], pred=[] → TP=0 FP=0 FN=0; Add=0 Omit=0; P=1 R=1; F1=1 (per your piecewise)
* ref>0, pred=0 → Add=0 Omit=1; P=1 R=0; F1=0
* ref=0, pred>0 → Add=1 Omit=0; P=0 R=1; F1=0
* duplicate predicted paraphrases → one-to-one ensures TP ≤ |ref|

Log parsing failures and continue.

---

## 7) Dependencies Notes

* If `sentence-transformers` exists: implement SBERT backend.
* Else TF-IDF backend must work end-to-end.
* If `scipy` exists: use Hungarian for max-weight.
* Else greedy matching acceptable (document + expose via cfg.match_mode).

---

## Deliverable Expectations

Produce all files in `src/eva/nlp_hall/` and update `config/logistics.py` with config + dataclasses.
No unused code, no network calls, deterministic behavior.
try/excpet 
if you write test code, should be in src/test (single file)
no-unnecessary code.

Ensure:
`python -m src.eva.nlp_hall.run_hall_eval --help` works.
