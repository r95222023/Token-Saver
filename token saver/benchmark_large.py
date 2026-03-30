"""
Benchmark for Prompt Skipper using a large text file (Sherlock Holmes).
Tests filtering performance on ~600KB of text.
"""

import asyncio
import time
from prompt_skipper import (
    ask,
    ask_with_marker,
    filter_chunks,
    filter_chunks_by_marker,
    filter_chunks_by_marker_parallel,
    SkipperConfig
)

# ------------------------------------------------------------------
# Load and Chunk Large File
# ------------------------------------------------------------------
def load_and_chunk(filename, num_chunks=32):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Simple fixed-size chunking for benchmarking
    chunk_size = len(text) // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else len(text)
        chunks.append(text[start:end])
    return chunks

FILE_PATH = "sherlock.txt"
QUERY = "Who is the King of Bohemia and what does he want from Sherlock Holmes?"

config = SkipperConfig(
    base_url="http://192.168.32.68:11434/v1",
    api_key="ollama",
    filter_model="gemma3:27b",
    answer_model="gemma3:27b",
    leaf_size=4,             # Larger leaf size for better context in large files
    max_tokens_filter=20,
)

async def run_benchmark():
    print(f"Loading {FILE_PATH}...")
    # Using 32 chunks for a balanced test of parallelization
    chunks = load_and_chunk(FILE_PATH, num_chunks=32)
    print(f"Total Chunks: {len(chunks)}")
    print(f"Query: {QUERY}\n")

    # 1. Strategy A: Divide & Conquer
    print("--- [Strategy A] Divide & Conquer ---")
    start = time.time()
    relevant_a, stats_a = await filter_chunks(chunks, QUERY, config)
    duration_a = time.time() - start
    print(f"Stats: {stats_a}")
    print(f"Time: {duration_a:.2f}s\n")

    # 2. Strategy B: Single-call Marker Streaming
    print("--- [Strategy B] Marker Streaming ---")
    start = time.time()
    relevant_b, stats_b = await filter_chunks_by_marker(chunks, QUERY, config)
    duration_b = time.time() - start
    print(f"Stats: {stats_b}")
    print(f"Time: {duration_b:.2f}s\n")

    # 3. Strategy C: Parallel Marker Streaming (4 workers)
    print("--- [Strategy C] Parallel Marker (4 workers) ---")
    start = time.time()
    relevant_c, stats_c = await filter_chunks_by_marker_parallel(
        chunks, QUERY, config, num_workers=4
    )
    duration_c = time.time() - start
    print(f"Stats: {stats_c}")
    print(f"Time: {duration_c:.2f}s\n")

    # Final Summary Answer (using Strategy B's result)
    print("--- Final Answer (using Strategy B survivors) ---")
    if relevant_b:
        answer, _ = await ask_with_marker(relevant_b, QUERY, config)
        print(f"Answer:\n{answer}")
    else:
        print("No relevant chunks found.")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
