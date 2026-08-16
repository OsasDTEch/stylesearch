# Style Search · a working demo for Madam

Meaning based hairstyle discovery, built by Omons Wisdom (omonswisdom.ict@gmail.com).

A user asks the way she would ask a friend:

> "low manipulation protective style for 4c hair, shoulder length, no heat"

and gets ranked, explained results. Three signals are fused for every match:

1. **Dense embeddings** (fastembed, BAAI bge-small-en-v1.5) for meaning
2. **BM25** for exact terms that matter ("4c", "knotless", "no heat")
3. **A facet layer that knows hair**: hair type compatibility, heat, manipulation level,
   longevity, swim safety, occasion. Extracted from the query in code, and able to
   penalize wrong answers (a silk press will never rank for a "no heat" query).

Every result shows its score breakdown and plain English reasons ("suits 4c hair",
"no heat", "lasts 8 weeks"). No black box.

## Run it

```bash
uv run app.py
# open http://localhost:8000
```

Or, if you prefer an explicit venv:

```bash
uv sync
uv run app.py
```

First run downloads the embedding model (~130MB) from Hugging Face. If the machine
is offline, the engine automatically falls back to TF-IDF + SVD latent vectors and
says so in the UI, so the demo always runs.

## What the production version looks like

This sketch uses 50 curated styles and text only. The real build for Madam:

- Index the actual catalog with **image + text embeddings** (a style is a picture first)
- A **knowledge graph** connecting styles, hair types, products, creators and Jars,
  so recommendations come from real relationships, not just text similarity
- **An evaluation harness before tuning**: a labeled set of real user queries scored
  for retrieval quality, so every change to the ranker is measured, not guessed
- Same engine powers "more like this" on any saved item and smarter Jar suggestions

Built in an afternoon. Imagine a quarter.
