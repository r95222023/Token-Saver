"""
Prompt Skipper — Divide & Conquer RAG context filtering.

Algorithm:
  1. Split chunks in half
  2. Check both halves for relevance IN PARALLEL
  3. Recurse on relevant halves, discard irrelevant ones
  4. Repeat until leaf_size is reached
  5. Pass surviving chunks to the final LLM call

Complexity: O(log N) rounds of parallel API calls (vs O(N) sequential)
"""

import asyncio
from dataclasses import dataclass, field
from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SkipperConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = "your-api-key"
    filter_model: str = "gpt-4o-mini"   # fast/cheap model for relevance checks
    answer_model: str = "gpt-4o"        # capable model for final answer
    leaf_size: int = 1                  # recurse until groups of this size
    max_tokens_filter: int = 10         # YES/NO needs very few tokens


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class FilterStats:
    api_calls: int = 0
    rounds: int = 0
    original_chunks: int = 0
    remaining_chunks: int = 0

    @property
    def pruned_chunks(self) -> int:
        return self.original_chunks - self.remaining_chunks

    def __str__(self) -> str:
        return (
            f"rounds={self.rounds} | "
            f"api_calls={self.api_calls} | "
            f"chunks: {self.original_chunks} → {self.remaining_chunks} "
            f"({self.pruned_chunks} pruned)"
        )


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

async def _empty() -> list[str]:
    return []


async def _check_relevance(
    client: AsyncOpenAI,
    config: SkipperConfig,
    chunks: list[str],
    query: str,
    stats: FilterStats,
) -> bool:
    """Ask the filter model if ANY chunk in the list is relevant to the query."""
    stats.api_calls += 1

    chunks_text = "\n---\n".join(
        f"[Chunk {i}]\n{chunk}" for i, chunk in enumerate(chunks)
    )

    response = await client.chat.completions.create(
        model=config.filter_model,
        max_tokens=config.max_tokens_filter,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a relevance checker. "
                    "Answer only YES or NO, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Query: {query}\n\n"
                    f"Chunks:\n{chunks_text}\n\n"
                    "Do any of these chunks contain information useful for "
                    "answering the query? Answer YES or NO only."
                ),
            },
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip().upper()
    return answer.startswith("Y")


async def _filter_recursive(
    client: AsyncOpenAI,
    config: SkipperConfig,
    chunks: list[str],
    query: str,
    stats: FilterStats,
    depth: int = 0,
) -> list[str]:
    if not chunks:
        return []

    stats.rounds = max(stats.rounds, depth + 1)

    # Leaf: check this small group directly
    if len(chunks) <= config.leaf_size:
        relevant = await _check_relevance(client, config, chunks, query, stats)
        return chunks if relevant else []

    # Split in half
    mid = len(chunks) // 2
    left, right = chunks[:mid], chunks[mid:]

    # Check both halves IN PARALLEL
    left_rel, right_rel = await asyncio.gather(
        _check_relevance(client, config, left, query, stats),
        _check_relevance(client, config, right, query, stats),
    )

    # Recurse on relevant halves IN PARALLEL
    left_task = (
        _filter_recursive(client, config, left, query, stats, depth + 1)
        if left_rel else _empty()
    )
    right_task = (
        _filter_recursive(client, config, right, query, stats, depth + 1)
        if right_rel else _empty()
    )

    left_result, right_result = await asyncio.gather(left_task, right_task)
    return left_result + right_result


async def filter_chunks(
    chunks: list[str],
    query: str,
    config: SkipperConfig,
) -> tuple[list[str], FilterStats]:
    """
    Filter RAG chunks using divide & conquer relevance checking.

    Returns:
        (relevant_chunks, stats)
    """
    stats = FilterStats(original_chunks=len(chunks))
    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)

    relevant = await _filter_recursive(client, config, chunks, query, stats)
    stats.remaining_chunks = len(relevant)
    return relevant, stats


async def ask(
    chunks: list[str],
    query: str,
    config: SkipperConfig,
    system_prompt: str = "You are a helpful assistant. Answer based on the provided context.",
) -> tuple[str, FilterStats]:
    """
    Full pipeline: filter irrelevant chunks, then answer the query.

    Returns:
        (answer, stats)
    """
    relevant_chunks, stats = await filter_chunks(chunks, query, config)

    if not relevant_chunks:
        return "No relevant information found in the provided context.", stats

    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(relevant_chunks)
    )

    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
    response = await client.chat.completions.create(
        model=config.answer_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
    )

    return response.choices[0].message.content, stats
