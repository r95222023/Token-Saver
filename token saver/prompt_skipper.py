"""
Prompt Skipper — RAG context filtering, two complementary strategies:

[A] Divide & Conquer  (filter_chunks)
  1. Split chunks in half
  2. Check both halves for relevance IN PARALLEL
  3. Recurse on relevant halves, discard irrelevant ones
  4. Repeat until leaf_size is reached
  Complexity: O(log N) parallel rounds

[B] Marker Streaming  (filter_chunks_by_marker)               ← NEW
  1. Tag every chunk with a unique skip-marker  [§0], [§1], …
  2. One LLM call: "stream back the markers of irrelevant chunks"
  3. As each [§N] token arrives in the stream → drop chunk N immediately
  4. Stream is cancelled the moment [DONE] is received
  Complexity: 1 API call, output ≈ (irrelevant_count × 5) tokens
"""

import asyncio
import re
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
    """
    Ask the filter model if ANY chunk in the list is relevant to the query.

    Uses streaming: the moment we receive the first 'Y' or 'N' character,
    we close the stream immediately — no need to wait for the full response.
    """
    stats.api_calls += 1

    chunks_text = "\n---\n".join(
        f"[Chunk {i}]\n{chunk}" for i, chunk in enumerate(chunks)
    )

    result: bool = False  # default to "not relevant" if nothing arrives

    stream = await client.chat.completions.create(
        model=config.filter_model,
        max_tokens=config.max_tokens_filter,
        stream=True,
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

    async with stream:
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            # 掃描每個 token 裡的字元，遇到 Y/N 立刻停止
            for ch in delta.upper():
                if ch == "Y":
                    result = True
                    return result   # 立即中斷 stream
                elif ch == "N":
                    result = False
                    return result   # 立即中斷 stream
                # 跳過空白、標點等非決定性字元

    return result


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


# ---------------------------------------------------------------------------
# Strategy B — Marker Streaming Filter
# ---------------------------------------------------------------------------

_MARKER_RE = re.compile(r'\[§(\d+)\]')
_DONE_TOKEN = "[DONE]"

_MARKER_SYSTEM_INSTR = (
    "You are a precision relevance filter. "
    "You will receive several text chunks, each prefixed with a tag like [§0], [§1], etc. "
    "Analyze each chunk against the user's query. "
    "Your task: IDENTIFY chunks that are IRRELEVANT (contain no useful info for the query). "
    "Output ONLY the tags [§N] of the irrelevant chunks. "
    "After listing them, output [DONE] and stop immediately. "
    "If all chunks are relevant, output only [DONE]. "
    "Example output:\n[§2]\n[§5]\n[§8]\n[DONE]"
)


async def filter_chunks_by_marker(
    chunks: list[str],
    query: str,
    config: SkipperConfig,
) -> tuple[list[str], FilterStats]:
    """
    Single-call streaming filter using special skip markers.
    """
    stats = FilterStats(original_chunks=len(chunks))
    stats.rounds = 1
    stats.api_calls = 1

    labeled_chunks = [f"[§{i}]\n{chunk}" for i, chunk in enumerate(chunks)]
    chunks_text = "\n\n".join(labeled_chunks)

    max_output_tokens = (len(chunks) + 1) * 10
    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)

    stream = await client.chat.completions.create(
        model=config.filter_model,
        max_tokens=max_output_tokens,
        stream=True,
        temperature=0,
        messages=[
            {"role": "system", "content": _MARKER_SYSTEM_INSTR},
            {"role": "user", "content": f"Query: {query}\n\nChunks:\n{chunks_text}"},
        ],
    )

    skip: set[int] = set()
    buffer = ""

    async with stream:
        async for event in stream:
            delta = event.choices[0].delta.content
            if not delta:
                continue
            buffer += delta

            if _DONE_TOKEN in buffer:
                pre_done = buffer.split(_DONE_TOKEN, 1)[0]
                for m in _MARKER_RE.finditer(pre_done):
                    skip.add(int(m.group(1)))
                break

            consumed_up_to = 0
            for m in _MARKER_RE.finditer(buffer):
                skip.add(int(m.group(1)))
                consumed_up_to = m.end()

            if consumed_up_to:
                buffer = buffer[consumed_up_to:]

    relevant = [c for i, c in enumerate(chunks) if i not in skip]
    stats.remaining_chunks = len(relevant)
    return relevant, stats


# ---------------------------------------------------------------------------
# Strategy C — Parallel Marker Streaming Filter
# ---------------------------------------------------------------------------

async def _marker_worker(
    client: AsyncOpenAI,
    config: SkipperConfig,
    partition: list[str],
    global_indices: list[int],
    query: str,
) -> list[int]:
    """
    Run marker-streaming on one partition of chunks.
    Returns the global indices of chunks that survived (are relevant).
    Partition-local markers [§0], [§1], … are mapped back to global indices.
    """
    labeled = [f"[§{i}]\n{chunk}" for i, chunk in enumerate(partition)]
    chunks_text = "\n\n".join(labeled)
    max_output_tokens = (len(partition) + 1) * 10

    stream = await client.chat.completions.create(
        model=config.filter_model,
        max_tokens=max_output_tokens,
        stream=True,
        temperature=0,
        messages=[
            {"role": "system", "content": _MARKER_SYSTEM_INSTR},
            {"role": "user", "content": f"Query: {query}\n\nChunks:\n{chunks_text}"},
        ],
    )

    local_skip: set[int] = set()
    buffer = ""

    async with stream:
        async for event in stream:
            delta = event.choices[0].delta.content
            if not delta:
                continue
            buffer += delta

            # Early exit on [DONE]
            if _DONE_TOKEN in buffer:
                pre_done = buffer.split(_DONE_TOKEN, 1)[0]
                for m in _MARKER_RE.finditer(pre_done):
                    local_skip.add(int(m.group(1)))
                break

            # Consume complete markers
            consumed_up_to = 0
            for m in _MARKER_RE.finditer(buffer):
                local_skip.add(int(m.group(1)))
                consumed_up_to = m.end()
            if consumed_up_to:
                buffer = buffer[consumed_up_to:]

    # Map local skip → surviving global indices
    return [
        global_indices[local_i]
        for local_i in range(len(partition))
        if local_i not in local_skip
    ]


async def filter_chunks_by_marker_parallel(
    chunks: list[str],
    query: str,
    config: SkipperConfig,
    num_workers: int = 4,
) -> tuple[list[str], FilterStats]:
    """
    Parallel marker-streaming filter  (Strategy C).

    Splits chunks into `num_workers` partitions and fires one marker-streaming
    LLM call per partition — all in parallel via asyncio.gather().
    Each stream stops as soon as it sees [DONE].
    Surviving chunks are merged back in their original order.

    Args:
        num_workers: how many parallel LLM calls to make (default 4).
                     Tune to your Ollama server's concurrency capacity.

    Returns:
        (relevant_chunks, stats)
    """
    stats = FilterStats(original_chunks=len(chunks))
    stats.rounds = 1          # all workers run in the same round
    stats.api_calls = num_workers

    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)

    # ── Partition chunks while keeping track of original indices ──────
    partitions: list[list[str]] = [[] for _ in range(num_workers)]
    global_idx_maps: list[list[int]] = [[] for _ in range(num_workers)]

    for i, chunk in enumerate(chunks):
        slot = i % num_workers
        partitions[slot].append(chunk)
        global_idx_maps[slot].append(i)

    # ── Fire all workers in parallel ──────────────────────────────────
    surviving_per_worker: list[list[int]] = await asyncio.gather(*[
        _marker_worker(client, config, partitions[w], global_idx_maps[w], query)
        for w in range(num_workers)
        if partitions[w]          # skip empty partitions (when N < num_workers)
    ])

    # ── Merge: collect surviving global indices, sort to restore order ─
    surviving_global = sorted(
        idx for worker_result in surviving_per_worker for idx in worker_result
    )

    relevant = [chunks[i] for i in surviving_global]
    stats.remaining_chunks = len(relevant)
    return relevant, stats


async def ask_with_marker(
    chunks: list[str],
    query: str,
    config: SkipperConfig,
    parallel: bool = False,
    num_workers: int = 4,
    system_prompt: str = "You are a helpful assistant. Answer based on the provided context.",
) -> tuple[str, FilterStats]:
    """
    Full pipeline using the Marker Streaming strategy.
    """
    if parallel:
        relevant_chunks, stats = await filter_chunks_by_marker_parallel(
            chunks, query, config, num_workers=num_workers
        )
    else:
        relevant_chunks, stats = await filter_chunks_by_marker(chunks, query, config)

    if not relevant_chunks:
        return "No relevant information found.", stats

    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(relevant_chunks)
    )

    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
    response = await client.chat.completions.create(
        model=config.answer_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )

    return response.choices[0].message.content, stats
