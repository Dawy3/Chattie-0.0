# Chattie — Technical Case Study

This document explains the key engineering decisions behind Chattie and why each choice was made.

---

## 1. Server-Sent Events (SSE) for Streaming

**Decision:** Stream LLM responses to the client using SSE instead of waiting for the full response.

**Why it matters:**
- **Time to First Token (TTFT)** drops from seconds to milliseconds — the user sees the answer forming immediately.
- Perceived latency is dramatically lower even when total generation time stays the same.
- SSE is a lightweight, HTTP-native protocol — no WebSocket upgrade, no persistent connection management, and automatic reconnection built into the browser `EventSource` API.

**Trade-off considered:** WebSockets offer bidirectional communication, but Chattie only needs server-to-client streaming. SSE is simpler and sufficient.

---

## 2. Page-Level Chunking

**Decision:** Split documents into chunks at page boundaries rather than using fixed token windows or recursive character splitting.

**Why it matters:**
- Pages are natural semantic units — headings, paragraphs, and tables rarely span page breaks.
- Retrieval accuracy is higher because each chunk carries coherent, self-contained context.
- The LLM receives well-formed input, reducing hallucination from mid-sentence chunk boundaries.

**Trade-off considered:** Smaller fixed-size chunks (e.g., 512 tokens) increase recall for pinpoint questions, but they fragment context and hurt answer quality on broader questions. Page-level chunking prioritizes accuracy over granularity.

---

## 3. HNSW Indexing (via Qdrant)

**Decision:** Use Hierarchical Navigable Small World (HNSW) graphs for vector indexing.

**Why it matters:**
- HNSW delivers approximate nearest-neighbor search in logarithmic time — retrieval stays fast even as the document collection grows.
- Query latency is consistently low (sub-10ms for typical collection sizes), which is critical when retrieval sits on the hot path before every LLM call.

**Trade-off considered:** HNSW indexes are memory-resident, so they consume more RAM than disk-based indexes (e.g., IVF). For Chattie's use case, the speed gain outweighs the memory cost.

---

## 4. Hybrid Search (Dense + Sparse + Recency)

**Decision:** Combine dense vector search, sparse keyword search (BM25), and recency boosting instead of relying on embeddings alone.

**Why it matters:**
- **Dense search** captures semantic similarity — "How do I cancel my subscription?" matches "refund policy."
- **Sparse search** catches exact terms that embeddings sometimes miss — product names, error codes, acronyms.
- **Recency boosting** surfaces the latest version of a document when multiple versions exist.
- Combining all three with reciprocal rank fusion produces more robust retrieval than any single method.

**Trade-off considered:** Hybrid search adds query-time complexity and requires maintaining two index types. The accuracy improvement justifies the cost.

---

## 5. Rule-Based Query Classifier

**Decision:** Classify incoming queries with lightweight rules before deciding whether to invoke the LLM.

**Why it matters:**
- Not every user message needs an LLM call. Greetings, simple commands, and FAQ-style questions can be handled directly.
- Skipping the LLM for these queries **saves cost** (fewer API calls) and **reduces latency** (instant response).
- The classifier acts as a fast-path router: only queries that genuinely require retrieval and generation are sent through the full RAG pipeline.

**Trade-off considered:** A learned classifier (fine-tuned model) would be more flexible, but adds model-serving overhead and training maintenance. Rule-based classification is transparent, zero-cost, and easy to extend.

---

## 6. Semantic Caching & Embedding Cache

**Decision:** Cache both LLM responses (keyed by semantic similarity) and embedding vectors.

**Why it matters:**
- **Semantic cache:** If a new query is semantically similar to a previously answered one, return the cached answer instantly — no embedding, retrieval, or generation needed. This cuts cost and latency to near zero for repeat and paraphrased questions.
- **Embedding cache:** Avoid recomputing embeddings for documents or queries that have already been embedded. This speeds up both ingestion and retrieval.

**Trade-off considered:** Cache invalidation is the classic hard problem. Chattie mitigates this with TTL-based expiry and cache clearing on document re-upload.
