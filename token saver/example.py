"""
Example usage of Prompt Skipper.

Replace base_url / api_key / models to match your provider.
"""

import asyncio
from prompt_skipper import ask, filter_chunks, SkipperConfig

# ------------------------------------------------------------------
# Simulated RAG chunks (mix of relevant and irrelevant)
# ------------------------------------------------------------------

CHUNKS = [
    # Irrelevant
    "The French Revolution began in 1789 and led to the rise of Napoleon Bonaparte.",
    "Photosynthesis is the process by which plants convert sunlight into glucose.",
    "The Amazon rainforest spans over 5.5 million square kilometres.",
    "Classical music composers include Beethoven, Mozart, and Bach.",
    # Relevant (about Python async)
    "Python's asyncio library allows writing concurrent code using the async/await syntax.",
    "The asyncio.gather() function runs multiple coroutines concurrently and returns their results.",
    # Irrelevant
    "The Eiffel Tower was constructed between 1887 and 1889 in Paris.",
    "Quantum entanglement is a phenomenon where particles become correlated.",
    # Relevant
    "AsyncIO event loops manage and dispatch asynchronous tasks in Python programs.",
    "To run an async function, use asyncio.run() as the entry point in Python 3.7+.",
    # Irrelevant
    "The Great Wall of China stretches over 21,000 kilometres.",
    "Mitochondria are known as the powerhouse of the cell.",
    "Shakespeare wrote approximately 37 plays and 154 sonnets.",
    "The speed of light in a vacuum is approximately 299,792 km/s.",
    "Coffee was first cultivated in Ethiopia around the 9th century.",
    "The periodic table was organized by Dmitri Mendeleev in 1869.",
]

QUERY = "How does Python asyncio work for concurrent programming?"

# ------------------------------------------------------------------
# Config — change these to your provider
# ------------------------------------------------------------------

config = SkipperConfig(
    base_url="https://api.openai.com/v1",  # or http://localhost:11434/v1 for Ollama
    api_key="your-api-key-here",
    filter_model="gpt-4o-mini",            # fast model for YES/NO checks
    answer_model="gpt-4o",                 # capable model for final answer
    leaf_size=1,
)


async def demo_filter_only():
    """Show which chunks survive after filtering."""
    print(f"Query: {QUERY}")
    print(f"Total chunks: {len(CHUNKS)}\n")

    relevant, stats = await filter_chunks(CHUNKS, QUERY, config)

    print("=== Surviving chunks ===")
    for i, chunk in enumerate(relevant, 1):
        print(f"  [{i}] {chunk}")

    print(f"\n=== Stats ===\n  {stats}")


async def demo_full_pipeline():
    """Full pipeline: filter + answer."""
    print(f"Query: {QUERY}\n")

    answer, stats = await ask(CHUNKS, QUERY, config)

    print("=== Answer ===")
    print(answer)
    print(f"\n=== Stats ===\n  {stats}")


if __name__ == "__main__":
    # Run filter-only demo
    asyncio.run(demo_filter_only())

    print("\n" + "=" * 60 + "\n")

    # Run full pipeline
    asyncio.run(demo_full_pipeline())
