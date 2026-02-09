 Full RAG Pipeline

  ┌─────────────────────────────────────────────────────────────┐
  │                    INGESTION PIPELINE                       │
  │                                                             │
  │  Upload (POST /documents)                                   │
  │    │                                                        │
  │    ▼                                                        │
  │  DocumentProcessor                                          │
  │    │  parse file (PDF, TXT, etc.)                           │
  │    ▼                                                        │
  │  Chunker (recursive, 512 tokens, 50 overlap)                │
  │    │  split into chunks                                     │
  │    ▼                                                        │
  │  EmbeddingGenerator (text-embedding-3-small, 1536d)         │
  │    │  embed each chunk                                      │
  │    ▼                                                        │
  │  QdrantStore                    BM25Search                  │
  │    │  store vectors + metadata    │  index chunks in memory │
  │    ▼                              ▼                         │
  │  [Qdrant DB]                  [BM25 Index]                  │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │                     QUERY PIPELINE                          │
  │                                                             │
  │  User Query (POST /query)                                   │
  │    │                                                        │
  │    ▼                                                        │
  │  ┌──────────────────────┐                                   │
  │  │  QueryClassifier     │  rule-based, <1ms, $0             │
  │  │  (4 routes)          │                                   │
  │  └──────┬───────────────┘                                   │
  │         │                                                   │
  │    ┌────┼──────────┬──────────────┐                         │
  │    ▼    ▼          ▼              ▼                          │
  │  REJECT CLARIFY  GENERATION   RETRIEVAL                     │
  │    │      │        │              │                          │
  │    ▼      ▼        │              │                         │
  │  static  static    │              │                         │
  │  reply   reply     ▼              ▼                         │
  │  (done)  (done)    ┌───────────────────────┐                │
  │                    │  SemanticCache        │                │
  │                    │  Layer 1: exact hash  │                │
  │                    │  Layer 2: embedding   │                │
  │                    │  similarity ≥ 0.9     │                │
  │                    │  Layer 3: cross-      │                │
  │                    │  encoder validate     │                 │
  │                    └─────────┬───────────┬─┘                 │
  │                    │     HIT │       MISS│                   │
  │                    │         ▼           │                   │
  │                    │    return cached    │                   │
  │                    │    response (done)  │                   │
  │                    │                     │                   │
  │                    │                     ▼                   │
  │                    │         EmbeddingGenerator              │
  │                    │              │  embed query             │
  │                    │              ▼                          │
  │                    │         HybridSearch                    │
  │                    │           ┌────┴────┐                   │
  │                    │           ▼         ▼                   │
  │                    │     VectorSearch  BM25Search            │
  │                    │      (Qdrant)    (keyword)              │
  │                    │           └────┬────┘                   │
  │                    │                ▼                        │
  │                    │         RRF Fusion + Recency Boost      │
  │                    │              │  top-k chunks            │
  │                    │              ▼                          │
  │                    │         ContextBuilder                  │
  │                    │              │  format chunks → context │
  │                    │              ▼                          │
  │                    │    ConversationMemory.get(session_id)   │
  │                    │              │  prior messages          │
  │                    │              ▼                          │
  │                    ├───→  PromptManager.build()              │
  │                    │       (query + context + history)       │
  │                    │              │                          │
  │                    │              ▼                          │
  │                    │         LLMClient.generate_stream()     │
  │                    │              │  SSE token-by-token      │
  │                    │              ▼                          │
  │                    │    ConversationMemory.add()             │
  │                    │         (store user + assistant msgs)   │
  │                    │              │                          │
  │                    │              ▼                          │
  │                    │    SemanticCache.set()                  │
  │                    │         (cache query → response)        │
  │                    │              │                          │
  │                    │              ▼                          │
  │                    └────→   SSE "done" event                 │
  └─────────────────────────────────────────────────────────────┘

  Step-by-step breakdown

  Ingestion (POST /documents)

  1. Upload — file saved to data/uploads/
  2. Parse — DocumentProcessor extracts raw text from the file
  3. Chunk — Chunker splits text using recursive strategy (512 tokens, 50 overlap)
  4. Embed — EmbeddingGenerator calls OpenAI text-embedding-3-small → 1536-dim vectors
  5. Store — vectors + metadata → QdrantStore; raw text → BM25Search in-memory index

  Query (POST /query)

  1. Classify — QueryClassifier (regex/keyword rules, no LLM) routes to one of 4 paths:
    - REJECTION → static refusal, done
    - CLARIFICATION → follow-up question, done
    - GENERATION / RETRIEVAL → continue below
  2. Cache check — SemanticCache.get(query):
    - Layer 1: exact hash match (SHA-256 of normalized query)
    - Layer 2: cosine similarity against cached embeddings (threshold ≥ 0.9)
    - Layer 3: CrossEncoder (ms-marco-MiniLM-L-6-v2) validates the match (threshold ≥ 0.7)
    - On hit → return cached response immediately, skip LLM
  3. Embed query — EmbeddingGenerator.embed_query() → 1536-dim vector
  4. Hybrid search — two parallel retrievals fused together:
    - VectorSearch — cosine similarity against Qdrant (top 100)
    - BM25Search — keyword/TF-IDF scoring (top 100)
    - RRF fusion — Reciprocal Rank Fusion merges both ranked lists + recency boost → top-k results
  5. Build context — ContextBuilder formats retrieved chunks into a single context string
  6. Conversation history — ConversationMemory.get(session_id) retrieves prior messages for multi-turn context
  7. Build prompts — PromptManager.build(query, context, history) → system prompt + user prompt
  8. Stream LLM — LLMClient.generate_stream() calls OpenAI-compatible API, yields tokens via SSE
  9. Store memory — ConversationMemory.add() saves both user query and assistant response for the session
  10. Store cache — SemanticCache.set(query, response) caches the pair (hash + embedding) for future hits

  Config-driven settings (from .env)
  ┌───────────┬────────────────────────────────────────────────────────────────────────────────────┐
  │ Component │                                    Key settings                                    │
  ├───────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ Embedding │ model, dimensions (1536), batch size                                               │
  ├───────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ Chunking  │ strategy (recursive), size (512), overlap (50)                                     │
  ├───────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ LLM       │ model (gpt-4o-mini), base_url, temperature, max_tokens                             │
  ├───────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ Cache     │ enabled, threshold (0.9), TTL (3600s), cross-encoder model, rerank threshold (0.7) │
  ├───────────┼────────────────────────────────────────────────────────────────────────────────────┤
  │ Qdrant    │ url, api_key, collection name                                                      │
  └───────────┴────────────────────────────────────────────────────────────────────────────────────┘