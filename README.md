# Token Saver (Prompt Skipper)

A RAG context filter that uses **O(log N) parallel API rounds** to quickly prune irrelevant chunks before sending them to your LLM — saving tokens, reducing cost, and improving answer quality.

---

## The Problem

A typical RAG pipeline looks like this:

```
vector search → top-K chunks → stuff all into prompt → LLM answers
```

This has three issues:

| Problem | Details |
|---------|---------|
| **Token waste** | top-K results contain lots of irrelevant content that still occupies the context window |
| **Quality drop** | Noise interferes with the model ("Lost in the Middle" problem) |
| **Higher cost** | More input tokens = higher API bills |

Re-ranking (sort and take top-N) is the common fix, but it's still a sequential O(N) operation.

---

## Three Filtering Strategies

### Strategy A — Divide & Conquer (`filter_chunks`)

Treats chunks like a binary tree. Each level's checks run **in parallel**:

```
              [all 16 chunks]
             relevant? YES → split
            /                    \
      [first 8]              [last 8]
     YES ↓   NO ✗          YES ↓   NO ✗
   [first 4] (discard)   [last 4] (discard)
   YES ↓                    NO ✗
 [first 2]               (discard)
 YES↓  NO✗
[chunk1] (discard)
```

**Complexity:**

| Metric | Sequential scan | Strategy A |
|--------|----------------|------------|
| API rounds | O(N) | O(log N) |
| Worst-case calls | N | 2N − 1 |
| Best-case calls | N | 2 |
| Parallelism | None | Full per round |

> 1024 chunks → at most **10 rounds** instead of 1024.

---

### Strategy B — Marker Streaming (`filter_chunks_by_marker`)

1. Tag each chunk with a unique marker: `[§0]`, `[§1]`, …
2. Send all chunks in **one API call**: "stream back the markers of irrelevant chunks"
3. As each `[§N]` token arrives → drop that chunk immediately
4. As soon as `[DONE]` is received → **abort the stream**, don't wait for the full response

**Complexity:** 1 API call, output ≈ `(irrelevant_count × 5)` tokens

---

### Strategy C — Parallel Marker Streaming (`filter_chunks_by_marker_parallel`)

Combines Strategy B with parallelism:

1. Split chunks into N partitions
2. Fire N marker-streaming calls **in parallel** via `asyncio.gather()`
3. Each worker stops as soon as it sees `[DONE]`
4. Merge surviving chunks back in original order

---

## Benchmark Results

Tested on the complete **Sherlock Holmes** collection (~593 KB, ~200K tokens),
split into 32 chunks (~18 KB / 5,000 words each),
using a local Ollama `gemma3:27b` server.

Query: *"Who is the King of Bohemia and what does he want?"*

| Metric | Strategy A (Divide & Conquer) | Strategy B (Marker Stream) | Strategy C (Parallel Marker) |
|--------|-------------------------------|----------------------------|-------------------------------|
| API calls | 22 (4 rounds) | 1 | 4 (parallel) |
| Elapsed time | 122.07 s | **14.29 s** | 40.09 s |
| Speedup vs A | baseline | **8.5×** | 3× |

**Why is Strategy B fastest for large files?**
Strategy A must re-read overlapping context across multiple grouped queries.
Strategy B reads everything once and streams back only the short marker tokens,
then aborts the connection early — minimal redundant inference.

**Pruning accuracy test** ("How do I make a lava chocolate cake?" → detective novel):
> chunks: 32 → **1** (31 pruned — **97% pruned**, correct result)

---

## Quick Start

```bash
pip install openai
```

```python
import asyncio
from token_saver.prompt_skipper import ask, SkipperConfig

chunks = [...]   # your RAG chunks
query  = "your question"

config = SkipperConfig(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    filter_model="gpt-4o-mini",   # cheap/fast model for YES/NO filtering
    answer_model="gpt-4o",        # capable model for final answer
)

answer, stats = asyncio.run(ask(chunks, query, config))
print(answer)
print(stats)
# rounds=4 | api_calls=14 | chunks: 16 → 4 (12 pruned)
```

---

## API Reference

### `filter_chunks` — filter only (Strategy A)

```python
relevant_chunks, stats = await filter_chunks(chunks, query, config)
```

### `filter_chunks_by_marker` — filter only (Strategy B)

```python
relevant_chunks, stats = await filter_chunks_by_marker(chunks, query, config)
```

### `filter_chunks_by_marker_parallel` — filter only (Strategy C)

```python
relevant_chunks, stats = await filter_chunks_by_marker_parallel(
    chunks, query, config, num_workers=4
)
```

### `ask` — full pipeline (Strategy A)

```python
answer, stats = await ask(chunks, query, config)
```

### `ask_with_marker` — full pipeline (Strategy B or C)

```python
answer, stats = await ask_with_marker(chunks, query, config, parallel=False)
answer, stats = await ask_with_marker(chunks, query, config, parallel=True, num_workers=4)
```

---

## Configuration

```python
@dataclass
class SkipperConfig:
    base_url: str         # any OpenAI-compatible endpoint
    api_key: str
    filter_model: str     # lightweight model for relevance checks
    answer_model: str     # capable model for the final answer
    leaf_size: int = 1    # stop recursing at this group size (Strategy A only)
```

**`leaf_size` guide (Strategy A):**

| leaf_size | Behavior | When to use |
|-----------|----------|-------------|
| `1` | Exact single-chunk precision | Default — most accurate |
| `3`–`5` | Keep small groups together | Short chunks, scattered relevance |
| Too large | Degrades to single full check | Loses divide & conquer benefit |

---

## Compatible Endpoints

Uses the `openai` package's `base_url` parameter — works with any OpenAI-compatible API:

```python
# OpenAI
config = SkipperConfig(base_url="https://api.openai.com/v1", api_key="sk-...")

# Ollama (local)
config = SkipperConfig(base_url="http://localhost:11434/v1", api_key="ollama",
                       filter_model="llama3.2", answer_model="llama3.1:70b")

# Azure OpenAI
config = SkipperConfig(base_url="https://<resource>.openai.azure.com/openai/deployments/<model>",
                       api_key="<azure-key>")

# Anthropic (via compatibility layer)
config = SkipperConfig(base_url="https://api.anthropic.com/v1",
                       api_key="<anthropic-key>",
                       filter_model="claude-haiku-4-5-20251001",
                       answer_model="claude-sonnet-4-6")
```

---

## When to Use Each Strategy

| Strategy | Best for |
|----------|---------|
| **A — Divide & Conquer** | Limited context windows; cheap small filter models; many parallel requests acceptable |
| **B — Marker Streaming** | Large context models (e.g. gemma3:27b, GPT-4o); single powerful server; fastest overall |
| **C — Parallel Marker** | Multi-GPU Ollama servers; want to balance load across parallel inference slots |

**Works best when:**
- You have many chunks (> 20)
- Relevant chunks are sparse (< 30% of total)
- You need to control final prompt length

**Less effective when:**
- Very few chunks (< 8)
- Most chunks are relevant (little to prune)
- Chunks are strongly interdependent (splitting hurts YES/NO accuracy)

---

## Repository Structure

```
prompt_skipper.py          # original implementation (Strategy A)
example.py                 # basic usage example
token saver/
├── prompt_skipper.py      # full implementation (Strategies A, B, C)
├── example.py             # basic usage
├── example_ollama.py      # Ollama-specific example
├── benchmark_large.py     # large-file stress test script
├── sanity_check.py        # pruning accuracy validation
└── benchmark_result.txt   # raw benchmark output
```
