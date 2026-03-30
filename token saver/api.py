"""
Token Saver REST API
====================
Exposes the three RAG chunk-filtering strategies over HTTP.

Run:
    pip install -r requirements.txt
    uvicorn api:app --host 0.0.0.0 --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from prompt_skipper import (
    SkipperConfig,
    ask,
    ask_with_marker,
    filter_chunks,
    filter_chunks_by_marker,
    filter_chunks_by_marker_parallel,
)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Token Saver API",
    description=(
        "RAG context filter — prune irrelevant chunks before sending to your LLM.\n\n"
        "**Strategy A** — Divide & Conquer: O(log N) parallel rounds\n\n"
        "**Strategy B** — Marker Streaming: 1 API call, stream-abort on [DONE]\n\n"
        "**Strategy C** — Parallel Marker: N parallel marker-streaming workers"
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Strategy(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class LLMConfig(BaseModel):
    base_url: str = Field(
        "https://api.openai.com/v1",
        description="Any OpenAI-compatible endpoint (OpenAI, Ollama, Azure, Anthropic, …)",
        examples=["http://localhost:11434/v1"],
    )
    api_key: str = Field(..., description="API key for the endpoint")
    filter_model: str = Field(
        "gpt-4o-mini",
        description="Lightweight model used for relevance checks (YES/NO)",
        examples=["gpt-4o-mini", "gemma3:27b", "llama3.2"],
    )
    answer_model: str = Field(
        "gpt-4o",
        description="Capable model used for the final answer",
        examples=["gpt-4o", "gemma3:27b", "llama3.1:70b"],
    )
    leaf_size: int = Field(
        1,
        ge=1,
        description="(Strategy A only) Stop recursing when group size reaches this value",
    )
    max_tokens_filter: int = Field(
        10,
        ge=1,
        description="(Strategy A only) Max tokens for YES/NO filter response",
    )


class FilterRequest(BaseModel):
    chunks: list[str] = Field(..., min_length=1, description="RAG chunks to filter")
    query: str = Field(..., min_length=1, description="The user's question")
    strategy: Strategy = Field(Strategy.B, description="Filtering strategy to use")
    num_workers: int = Field(
        4,
        ge=1,
        description="(Strategy C only) Number of parallel marker-streaming workers",
    )
    llm: LLMConfig


class AskRequest(BaseModel):
    chunks: list[str] = Field(..., min_length=1, description="RAG chunks to filter")
    query: str = Field(..., min_length=1, description="The user's question")
    strategy: Strategy = Field(Strategy.B, description="Filtering strategy to use")
    num_workers: int = Field(
        4,
        ge=1,
        description="(Strategy C only) Number of parallel marker-streaming workers",
    )
    system_prompt: str = Field(
        "You are a helpful assistant. Answer based on the provided context.",
        description="System prompt sent to the answer model",
    )
    llm: LLMConfig


class StatsResponse(BaseModel):
    rounds: int
    api_calls: int
    original_chunks: int
    remaining_chunks: int
    pruned_chunks: int


class FilterResponse(BaseModel):
    relevant_chunks: list[str]
    stats: StatsResponse


class AskResponse(BaseModel):
    answer: str
    relevant_chunks: list[str]
    stats: StatsResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_config(llm: LLMConfig) -> SkipperConfig:
    return SkipperConfig(
        base_url=llm.base_url,
        api_key=llm.api_key,
        filter_model=llm.filter_model,
        answer_model=llm.answer_model,
        leaf_size=llm.leaf_size,
        max_tokens_filter=llm.max_tokens_filter,
    )


def _stats_dict(stats) -> StatsResponse:
    return StatsResponse(
        rounds=stats.rounds,
        api_calls=stats.api_calls,
        original_chunks=stats.original_chunks,
        remaining_chunks=stats.remaining_chunks,
        pruned_chunks=stats.pruned_chunks,
    )


async def _run_filter(req: FilterRequest):
    config = _build_config(req.llm)
    if req.strategy == Strategy.A:
        return await filter_chunks(req.chunks, req.query, config)
    elif req.strategy == Strategy.B:
        return await filter_chunks_by_marker(req.chunks, req.query, config)
    else:  # C
        return await filter_chunks_by_marker_parallel(
            req.chunks, req.query, config, num_workers=req.num_workers
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health():
    """Returns 200 OK when the server is running."""
    return {"status": "ok"}


@app.post("/filter", response_model=FilterResponse, tags=["Filter"])
async def filter_endpoint(req: FilterRequest):
    """
    Filter RAG chunks — return only the relevant ones.

    - **Strategy A**: Divide & conquer, O(log N) parallel rounds. Best for limited context windows.
    - **Strategy B**: Single LLM call with marker streaming. Fastest on large-context models.
    - **Strategy C**: Parallel marker streaming across N workers. Best for multi-GPU servers.
    """
    try:
        relevant, stats = await _run_filter(req)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return FilterResponse(relevant_chunks=relevant, stats=_stats_dict(stats))


@app.post("/ask", response_model=AskResponse, tags=["Ask"])
async def ask_endpoint(req: AskRequest):
    """
    Full RAG pipeline: filter irrelevant chunks, then answer the query.

    Uses `filter_model` for relevance filtering and `answer_model` for the final response.
    Returns the filtered chunks alongside the answer so you can inspect what the model saw.
    """
    config = _build_config(req.llm)
    try:
        # Filter first, then answer using the surviving chunks
        filter_req = FilterRequest(
            chunks=req.chunks,
            query=req.query,
            strategy=req.strategy,
            num_workers=req.num_workers,
            llm=req.llm,
        )
        relevant, stats = await _run_filter(filter_req)

        if not relevant:
            return AskResponse(
                answer="No relevant information found in the provided context.",
                relevant_chunks=[],
                stats=_stats_dict(stats),
            )

        from openai import AsyncOpenAI

        context = "\n\n---\n\n".join(
            f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(relevant)
        )
        client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
        response = await client.chat.completions.create(
            model=config.answer_model,
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"},
            ],
        )
        answer = response.choices[0].message.content

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return AskResponse(answer=answer, relevant_chunks=relevant, stats=_stats_dict(stats))
