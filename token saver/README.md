# Prompt Skipper

分治法 RAG 過濾器：用 O(log N) 輪平行 API 呼叫，從大量 chunks 中快速篩出真正相關的內容。

---

## 問題背景

RAG（Retrieval-Augmented Generation）系統的常見做法：

```
向量搜尋 → 取 top-K chunks → 全部塞進 prompt → 讓 LLM 回答
```

這個流程有三個缺點：

| 問題 | 說明 |
|------|------|
| **Token 浪費** | top-K 裡有大量不相關內容，卻佔用 context window |
| **品質下降** | 無關資訊會干擾模型，導致回答品質降低（Lost in the Middle 問題） |
| **成本增加** | 輸入 token 越多，API 費用越高 |

常見的解法是 **Re-ranking**（重新排序後取前 N 名），但這依然是序列式的 O(N) 操作。

---

## 核心概念：分治篩選

Prompt Skipper 借鑑二元搜尋的思路，把 chunks 視為一棵二元樹：

```
                    [全部 16 個 chunks]
                   相關？YES → 繼續拆分
                  /                    \
        [前 8 個]                    [後 8 個]
       YES ↓        NO ✗           YES ↓     NO ✗
      [前 4 個]  (丟棄)          [後 4 個]  (丟棄)
      YES ↓                      NO ✗
    [前 2 個]                  (丟棄)
    YES↓  NO✗
  [chunk1] (丟棄)
```

**關鍵優勢**：每一層的所有檢查都是**平行**執行的。

---

## 演算法細節

### 遞迴邏輯

```python
async def filter(chunks, query):
    if len(chunks) <= leaf_size:          # 1. 到達葉節點 → 直接檢查
        return chunks if relevant else []

    left, right = split(chunks)            # 2. 對半切分

    left_rel, right_rel = await parallel(  # 3. 平行檢查兩半
        check(left, query),
        check(right, query),
    )

    left_result, right_result = await parallel(  # 4. 平行遞迴相關的部分
        filter(left) if left_rel else [],
        filter(right) if right_rel else [],
    )

    return left_result + right_result      # 5. 合併結果
```

### 複雜度分析

設 N = chunk 數量，R = 實際相關的 chunks 數量。

| 指標 | 傳統逐一檢查 | Prompt Skipper |
|------|-------------|----------------|
| **API 呼叫輪數** | O(N) | O(log N) |
| **最差 API 總呼叫數** | N | 2N − 1 |
| **最佳 API 總呼叫數** | N | 2 |
| **平行度** | 無 | 每輪全部平行 |

> **輪數說明**：有 N 個 chunks 時，樹高為 log₂N，因此只需 log₂N 輪。
> 例如 1024 個 chunks → 最多 10 輪（而非 1024 輪）。

### 早期剪枝

當某個子樹被判定為「不相關」，整棵子樹立即被丟棄，不再往下遞迴。
相關 chunks 越稀疏，節省越多。

---

## 實作架構

```
prompt_skipper.py
├── SkipperConfig          資料類別，存放模型設定
├── FilterStats            統計：輪數、API 呼叫數、chunk 存活率
├── _check_relevance()     對一組 chunks 問「是否有任何相關？」→ YES/NO
├── _filter_recursive()    核心遞迴函式
├── filter_chunks()        公開 API：只做過濾
└── ask()                  公開 API：過濾 + 最終回答
```

### `_check_relevance` 的 prompt 設計

只問「**這組 chunks 裡有沒有任何一個**和 query 相關」，而非「哪個最相關」。
這讓我們可以：
1. 使用便宜快速的小模型（如 `gpt-4o-mini` 或本地 `llama3.2`）
2. 限制輸出 token 數到 10（只需 YES/NO）
3. 批次檢查多個 chunks，降低 API 呼叫總次數

---

## 策略二：Marker Streaming (自定義快速裁切)

除了分治法，本專案現支援 **Marker Streaming** 策略（由 USER 提議優化）：

### 核心概念
1. **唯一標記**：為每個 chunk 加上 `[§0]`, `[§1]` 等特殊標記。
2. **單次呼叫**：一次性將所有 chunks 送入，要求模型輸出「不相關」段落的標記。
3. **流式截斷**：
   - 程式監聽 API stream。
   - 一旦偵測到 `[§N]` 出現在 token 中，立即將該段落從 Context 中剔除。
   - 一旦偵測到 `[DONE]` 標記，**立刻中斷 (Abort) API 連線**，不等待模型輸出完畢。

### 優點
- **極低延遲**：不需要等待模型生成完整解釋，看到標記就裁切。
- **節省輸出 Token**：輸出僅包含短標記，且在任務完成時強行中斷。

---

## 策略三：平行 Marker Streaming

將 Marker Streaming 與平行化結合：
1. 將 chunks 分成 $N$ 個群組。
2. 平行啟動 $N$ 個 marker-streaming 呼叫。
3. 各自偵測到 `[DONE]` 即停止。
4. 取各工作者的聯集，恢復原始順序並合併。

---

## 公開 API

### `filter_chunks` — 只過濾

```python
from prompt_skipper import filter_chunks, SkipperConfig

config = SkipperConfig(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    filter_model="gpt-4o-mini",
    answer_model="gpt-4o",
    leaf_size=1,
)

relevant_chunks, stats = await filter_chunks(chunks, query, config)

print(stats)
# rounds=4 | api_calls=14 | chunks: 16 → 4 (12 pruned)
```

### `ask` — 完整 pipeline

```python
from prompt_skipper import ask, SkipperConfig

answer, stats = await ask(chunks, query, config)
```

內部流程：

```
chunks → filter_chunks() → 相關 chunks → 強力模型 → 回答
                ↑                              ↑
           gpt-4o-mini                      gpt-4o
          （便宜、快速）                  （準確、完整）
```

---

## 設定選項

```python
@dataclass
class SkipperConfig:
    base_url: str        # 任何 OpenAI compatible endpoint
    api_key: str
    filter_model: str    # 用於 YES/NO 判斷的輕量模型
    answer_model: str    # 用於最終回答的強力模型
    leaf_size: int = 1   # 遞迴到幾個 chunk 時停止拆分
```

**`leaf_size` 的影響：**

| leaf_size | 行為 | 適合情境 |
|-----------|------|---------|
| `1` | 精確到單一 chunk | 預設，最精準 |
| `3`–`5` | 以小組為單位保留 | chunks 很短、相關性分散時 |
| 太大 | 退化成單次全量檢查 | 失去分治優勢 |

---

## 接不同服務

因為使用 `openai` 套件的 `base_url` 參數，可以接任何相容的服務：

```python
# OpenAI
config = SkipperConfig(base_url="https://api.openai.com/v1", api_key="sk-...")

# Ollama（本地）
config = SkipperConfig(base_url="http://localhost:11434/v1", api_key="ollama",
                       filter_model="llama3.2", answer_model="llama3.1:70b")

# Azure OpenAI
config = SkipperConfig(base_url="https://<resource>.openai.azure.com/openai/deployments/<model>",
                       api_key="<azure-key>")

# Anthropic（透過相容層）
config = SkipperConfig(base_url="https://api.anthropic.com/v1",
                       api_key="<anthropic-key>",
                       filter_model="claude-haiku-4-5-20251001",
                       answer_model="claude-sonnet-4-6")
```

---

## 使用限制與注意事項

**何時效果最好：**
- Chunks 數量多（> 20）
- 相關 chunks 佔少數（稀疏分布）
- 需要控制最終 prompt 的長度

**何時效果有限：**
- Chunks 數量很少（< 8），分治優勢不明顯
- 相關 chunks 佔多數（大量剪枝機會不存在）
- Chunks 之間關聯性強，拆開後 YES/NO 判斷不準確

**`leaf_size=1` 的邊界情況：**
單一 chunk 判斷可能比「一組相關 chunks 一起判斷」更不準確，
因為單一句子缺乏上下文。可考慮 `leaf_size=2` 或 `3` 來緩解。

---

## 快速開始

```bash
pip install openai
```

```python
import asyncio
from prompt_skipper import ask, SkipperConfig

chunks = [...]   # 你的 RAG chunks
query = "你的問題"

config = SkipperConfig(
    base_url="https://api.openai.com/v1",
    api_key="your-key",
    filter_model="gpt-4o-mini",
    answer_model="gpt-4o",
)

answer, stats = asyncio.run(ask(chunks, query, config))
print(answer)
print(stats)
```

完整範例見 [`example.py`](example.py)。
