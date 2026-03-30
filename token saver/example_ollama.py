"""
Example usage of Prompt Skipper — 使用本地 Ollama gemma3:27b。

兩種過濾策略對比：
  [A] filter_chunks           — 分治法，O(log N) 輪平行呼叫
  [B] filter_chunks_by_marker — Marker Streaming，1次呼叫，流式偵測標記即停止

Ollama 伺服器：http://192.168.32.68:11434
模型：gemma3:27b
"""

import asyncio
from prompt_skipper import (
    ask,
    ask_with_marker,
    filter_chunks,
    filter_chunks_by_marker,
    filter_chunks_by_marker_parallel,
    SkipperConfig,
)

# ------------------------------------------------------------------
# Simulated RAG chunks（相關 vs 不相關混合）
# ------------------------------------------------------------------

CHUNKS = [
    # 不相關
    "法國大革命始於1789年，並導致拿破崙的崛起。",
    "光合作用是植物將陽光轉化為葡萄糖的過程。",
    "亞馬遜雨林面積超過550萬平方公里。",
    "古典音樂作曲家包括貝多芬、莫札特和巴哈。",
    # 相關（Python async）
    "Python 的 asyncio 函式庫允許使用 async/await 語法撰寫並發程式碼。",
    "asyncio.gather() 函式可並發執行多個協程，並返回結果列表。",
    # 不相關
    "艾菲爾鐵塔於1887年至1889年間在巴黎建成。",
    "量子糾纏是粒子之間產生相關性的現象。",
    # 相關
    "AsyncIO 事件迴圈負責管理和調度 Python 程式中的非同步任務。",
    "若要執行 async 函式，在 Python 3.7+ 中可使用 asyncio.run() 作為入口點。",
    # 不相關
    "中國長城全長超過2萬1千公里。",
    "粒線體被稱為細胞的發電廠。",
    "莎士比亞共創作了約37部戲劇和154首十四行詩。",
    "光在真空中的速度約為每秒29萬9792公里。",
    "咖啡最早於9世紀在衣索比亞被種植。",
    "門得列夫於1869年整理創建了元素週期表。",
]

QUERY = "Python asyncio 如何實現並發程式設計？"

# ------------------------------------------------------------------
# 設定 — 指向本地 Ollama (gemma3:27b)
# ------------------------------------------------------------------
config = SkipperConfig(
    base_url="http://192.168.32.68:11434/v1",
    api_key="ollama",
    filter_model="gemma3:27b",
    answer_model="gemma3:27b",
    leaf_size=2,
    max_tokens_filter=16,
)


# ------------------------------------------------------------------
# [A] 分治法過濾
# ------------------------------------------------------------------
async def demo_filter_only():
    """只展示哪些 chunks 存活。"""
    print(f"查詢：{QUERY}")
    print(f"Chunks 總數：{len(CHUNKS)}\n")

    relevant, stats = await filter_chunks(CHUNKS, QUERY, config)

    print("=== 存活的 Chunks ===")
    for i, chunk in enumerate(relevant, 1):
        print(f"  [{i}] {chunk}")

    print(f"\n=== 統計 ===\n  {stats}")


async def demo_full_pipeline():
    """完整流程：過濾 + 回答。"""
    print(f"查詢：{QUERY}\n")

    answer, stats = await ask(CHUNKS, QUERY, config)

    print("=== 回答 ===")
    print(answer)
    print(f"\n=== 統計 ===\n  {stats}")


# ------------------------------------------------------------------
# [B] Marker Streaming 過濾（新策略）
# ------------------------------------------------------------------
async def demo_marker_filter():
    """
    每個 chunk 被標記為 [§N]。
    LLM 在 stream 中輸出不相關 chunk 的標記。
    一旦偵測到 [§N] → 立刻把第 N 個 chunk 剔除。
    偵測到 [DONE] → 立即截斷 stream。
    只需 1 次 API 呼叫，輸出極短（只有標記 ID）。
    """
    print(f"查詢：{QUERY}")
    print(f"Chunks 總數：{len(CHUNKS)}\n")

    relevant, stats = await filter_chunks_by_marker(CHUNKS, QUERY, config)

    print("=== 存活的 Chunks ===")
    for i, chunk in enumerate(relevant, 1):
        print(f"  [{i}] {chunk}")

    print(f"\n=== 統計 ===\n  {stats}")


# ------------------------------------------------------------------
# [C] 平行 Marker Streaming（新策略）
# ------------------------------------------------------------------
async def demo_parallel_marker_filter():
    """
    將 chunks 切成 N 份，每份分配給一個獨立的 marker-streaming LLM 呼叫。
    所有呼叫透過 asyncio.gather() 同時跑，各自偵測到 [DONE] 就立即截斷自己的 stream。
    最後將各工作者的存活 chunks 合併，依原始順序排列。

    num_workers=4 → 16 個 chunks 被分成 4 組各 4 個，4 個 LLM 同時跑。
    """
    print(f"查詢：{QUERY}")
    print(f"Chunks 總數：{len(CHUNKS)}  |  workers=4\n")

    relevant, stats = await filter_chunks_by_marker_parallel(
        CHUNKS, QUERY, config, num_workers=4
    )

    print("=== 存活的 Chunks ===")
    for i, chunk in enumerate(relevant, 1):
        print(f"  [{i}] {chunk}")

    print(f"\n=== 統計 ===\n  {stats}")


async def demo_marker_pipeline():
    """完整流程：Marker Streaming 過濾 + 最終回答。"""
    print(f"查詢：{QUERY}\n")

    answer, stats = await ask_with_marker(CHUNKS, QUERY, config)

    print("=== 回答 (Marker Strategy) ===")
    print(answer)
    print(f"\n=== 統計 ===\n  {stats}")


if __name__ == "__main__":
    print("【策略 A】分治法過濾（O(log N) 輪平行呼叫）")
    print("=" * 60)
    asyncio.run(demo_filter_only())

    print("\n" + "=" * 60 + "\n")

    print("【策略 B】Marker Streaming 過濾（1 次呼叫，流式截斷）")
    print("=" * 60)
    asyncio.run(demo_marker_filter())

    print("\n" + "=" * 60 + "\n")

    print("【策略 C】平行 Marker Streaming（N 次平行呼叫，各自流式截斷）")
    print("=" * 60)
    asyncio.run(demo_parallel_marker_filter())

    print("\n" + "=" * 60 + "\n")

    print("【完整 pipeline - 策略 A】分治過濾 + 最終回答")
    print("=" * 60)
    asyncio.run(demo_full_pipeline())

    print("\n" + "=" * 60 + "\n")

    print("【完整 pipeline - 策略 B】Marker 過濾 + 最終回答")
    print("=" * 60)
    asyncio.run(demo_marker_pipeline())
