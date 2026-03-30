"""
Sanity check for pruning accuracy.
Query: Something definitely NOT in Sherlock Holmes.
"""

import asyncio
from prompt_skipper import filter_chunks_by_marker, SkipperConfig

async def sanity_check():
    with open("sherlock.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # 20 chunks
    num_chunks = 20
    chunk_size = len(text) // num_chunks
    chunks = [text[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    
    query = "How to bake a chocolate cake with lava center?"
    
    config = SkipperConfig(
        base_url="http://192.168.32.68:11434/v1",
        api_key="ollama",
        filter_model="gemma3:27b",
        answer_model="gemma3:27b",
    )

    print(f"Query: {query}")
    print("Testing Marker Filter...")
    relevant, stats = await filter_chunks_by_marker(chunks, query, config)
    print(f"Stats: {stats}")
    if len(relevant) > 0:
        print(f"Sample from 'relevant' chunk: {relevant[0][:200]}...")

if __name__ == "__main__":
    asyncio.run(sanity_check())
