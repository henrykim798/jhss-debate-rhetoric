# Replicable LLM Rhetoric Classification — How to Run

This pipeline re-measures the three rhetorical categories with a fixed, documented rubric
(`RUBRIC.md`) and a strong language model, so the labels are reproducible (temperature = 0)
and require no manual coding.

## Steps
1. `pip install openai pandas`  (or `pip install anthropic pandas`)
2. `export OPENAI_API_KEY=sk-...`  (or `export ANTHROPIC_API_KEY=...` and `export LLM_BACKEND=anthropic`)
3. Run:
   ```bash
   python llm_classify.py \
     --corpus ../../R1/Data_Files/corpus_sentences.csv \
     --gold   ../../R1/Data_Files/gold_sample.json \
     --out    ../llm_labels.csv \
     --counts ../llm_counts.csv
   ```
4. It first prints **gold-set validation** (per-category precision/recall/F1), then classifies
   all sentences, caches to `llm_cache.jsonl`, and writes per-sentence labels + candidate-debate
   counts.
5. Send me `llm_labels.csv` / `llm_counts.csv` and I'll merge them into the poll panel and
   re-run every regression in `reanalysis_master.py`, then report what holds.

## Why this is the right method
- An LLM reads **context**, so it distinguishes *condemning* discrimination from *committing*
  it — the exact failure that sank the keyword measure (e.g., "he called them rapists",
  "those people who saved our lives", "go back to community college").
- The rubric is fixed and versioned; the run is deterministic; the gold validation is built in.
- Cost: ~1,230 API calls (batch 20) over 24,546 sentences — a few dollars on `gpt-4o-mini`.

## Honesty checks built in
- Validation runs **before** you trust any counts.
- Discriminatory is defined strictly (protected-group derogation only); expect it to remain
  rare. If its gold precision/recall stays poor, report it as a combined "uncivil" measure or
  drop it — do not force a standalone result.
