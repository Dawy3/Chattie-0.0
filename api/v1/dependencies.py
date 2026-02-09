"""FastAPI dependency injection accessors.

Each function reads a service from request.app.state, set during lifespan.
"""

from fastapi import Request

from src.services.vector_store.qdrant import QdrantStore
from src.core.embedding.generator import EmbeddingGenerator
from src.core.retrieval.bm25_search import BM25Search
from src.core.retrieval.hybrid_search import HybridSearch
from src.core.generation.llm_client import LLMClient
from src.core.generation.context_builder import ContextBuilder
from src.core.generation.prompt_manager import PromptManager
from src.core.query.classifier import QueryClassifier
# from src.core.query.transformer import MultiQueryTransformer
from src.core.chunking.strategies import Chunker
from src.services.document_processor import DocumentProcessor
from src.core.memory.conversation import ConversationMemory
from src.core.caching.semantic_cache import SemanticCache


def get_qdrant_store(request: Request) -> QdrantStore:
    return request.app.state.qdrant_store


def get_embedding_generator(request: Request) -> EmbeddingGenerator:
    return request.app.state.embedding_generator


def get_bm25_search(request: Request) -> BM25Search:
    return request.app.state.bm25_search


def get_hybrid_search(request: Request) -> HybridSearch:
    return request.app.state.hybrid_search


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_context_builder(request: Request) -> ContextBuilder:
    return request.app.state.context_builder


def get_prompt_manager(request: Request) -> PromptManager:
    return request.app.state.prompt_manager


def get_query_classifier(request: Request) -> QueryClassifier:
    return request.app.state.query_classifier


# def get_query_transformer(request: Request) -> MultiQueryTransformer:
    # return request.app.state.query_transformer


def get_chunker(request: Request) -> Chunker:
    return request.app.state.chunker


def get_document_processor(request: Request) -> DocumentProcessor:
    return request.app.state.document_processor


def get_conversation_memory(request: Request) -> ConversationMemory:
    return request.app.state.conversation_memory


def get_semantic_cache(request: Request) -> SemanticCache | None:
    return request.app.state.semantic_cache


def get_upload_dir(request: Request) -> str:
    return request.app.state.upload_dir
