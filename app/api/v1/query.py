"""Full RAG pipeline query endpoint (SSE streaming) and classify debug endpoint."""

import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

from src.core.embedding.generator import EmbeddingGenerator
from src.core.retrieval.hybrid_search import HybridSearch
from src.core.generation.llm_client import LLMClient
from src.core.generation.context_builder import ContextBuilder
from src.core.generation.prompt_manager import PromptManager
from src.core.query.classifier import QueryClassifier, QueryRoute
from src.core.memory.conversation import ConversationMemory

from api.v1.schemas import (
    QueryRequest,
    QuerySource,
    ClassifyRequest,
    ClassifyResponse,
)
from api.v1.dependencies import (
    get_embedding_generator,
    get_hybrid_search,
    get_llm_client,
    get_context_builder,
    get_prompt_manager,
    get_query_classifier,
    get_conversation_memory,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/query")
async def query(
    request: QueryRequest,
    embedding_gen: EmbeddingGenerator = Depends(get_embedding_generator),
    hybrid_search: HybridSearch = Depends(get_hybrid_search),
    llm_client: LLMClient = Depends(get_llm_client),
    context_builder: ContextBuilder = Depends(get_context_builder),
    prompt_manager: PromptManager = Depends(get_prompt_manager),
    classifier: QueryClassifier = Depends(get_query_classifier),
    memory: ConversationMemory = Depends(get_conversation_memory),
):
    """Full RAG pipeline with SSE streaming.

    SSE events:
        event: metadata  — {route, session_id, sources}
        event: token     — {content: "..."}
        event: done      — {model, usage, latency_ms}
        event: error     — {detail: "..."}
    """
    start = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    # 1. Classify
    classification = classifier.classify(request.query)
    route = classification.route

    # ---- Non-streaming early returns (rejection / clarification) ----

    if route == QueryRoute.REJECTION:
        answer = "I'm sorry, I can't help with that request."

        async def rejection_stream():
            yield _sse_event("metadata", {"route": route.value, "session_id": session_id, "sources": []})
            yield _sse_event("token", {"content": answer})
            yield _sse_event("done", {"model": "", "usage": {}, "latency_ms": (time.time() - start) * 1000})

        return StreamingResponse(rejection_stream(), media_type="text/event-stream")

    if route == QueryRoute.CLARIFICATION:
        answer = classification.follow_up_question or "Could you please provide more details?"

        async def clarification_stream():
            yield _sse_event("metadata", {"route": route.value, "session_id": session_id, "sources": []})
            yield _sse_event("token", {"content": answer})
            yield _sse_event("done", {"model": "", "usage": {}, "latency_ms": (time.time() - start) * 1000})

        return StreamingResponse(clarification_stream(), media_type="text/event-stream")

    # ---- GENERATION route (LLM-only, no RAG) ----

    if route == QueryRoute.GENERATION:
        system_prompt, user_prompt = prompt_manager.build(query=request.query, context="")

        async def generation_stream():
            collected = []
            yield _sse_event("metadata", {"route": route.value, "session_id": session_id, "sources": []})

            async for token in llm_client.generate_stream(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                collected.append(token)
                yield _sse_event("token", {"content": token})

            full_answer = "".join(collected)
            memory.add(session_id, "user", request.query)
            memory.add(session_id, "assistant", full_answer)

            yield _sse_event("done", {
                "model": llm_client.model,
                "usage": {},
                "latency_ms": (time.time() - start) * 1000,
            })

        return StreamingResponse(generation_stream(), media_type="text/event-stream")

    # ---- RETRIEVAL route — full RAG pipeline ----

    # Embed query
    query_embedding = await embedding_gen.embed_query(request.query)

    # Hybrid search
    search_results = hybrid_search.search(
        query=request.query,
        query_embedding=query_embedding,
        top_k=request.top_k,
        filter_dict=request.filter,
    )

    # Build context
    chunks_for_context = [
        {
            "chunk_id": r.chunk_id,
            "content": r.content,
            "document_id": r.document_id,
            "score": r.score,
            "metadata": r.metadata,
        }
        for r in search_results
    ]
    context = context_builder.build(chunks_for_context)

    # Conversation history
    history = None
    if request.use_history:
        history = memory.get(session_id) or None

    # Build prompt
    system_prompt, user_prompt = prompt_manager.build(
        query=request.query,
        context=context.text,
        history=history,
    )

    # Build sources payload
    sources = [
        QuerySource(
            document_id=r.document_id,
            chunk_id=r.chunk_id,
            content=r.content,
            score=r.score,
            metadata=r.metadata,
        ).model_dump()
        for r in search_results
    ]

    async def retrieval_stream():
        collected = []

        # Send metadata + sources first
        yield _sse_event("metadata", {
            "route": route.value,
            "session_id": session_id,
            "sources": sources,
        })

        # Stream LLM tokens
        async for token in llm_client.generate_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            collected.append(token)
            yield _sse_event("token", {"content": token})

        full_answer = "".join(collected)
        memory.add(session_id, "user", request.query)
        memory.add(session_id, "assistant", full_answer)

        yield _sse_event("done", {
            "model": llm_client.model,
            "usage": {},
            "latency_ms": (time.time() - start) * 1000,
        })

    return StreamingResponse(retrieval_stream(), media_type="text/event-stream")


@router.post("/query/classify", response_model=ClassifyResponse)
async def classify_query(
    request: ClassifyRequest,
    classifier: QueryClassifier = Depends(get_query_classifier),
):
    """Debug endpoint: classify a query without executing the pipeline."""
    result = classifier.classify(request.query)
    return ClassifyResponse(
        query=result.query,
        route=result.route.value,
        reason=result.reason,
        follow_up_question=result.follow_up_question,
    )
