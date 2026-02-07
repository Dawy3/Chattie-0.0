"""Document upload, listing, detail, and deletion endpoints."""

import logging
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from src.services.vector_store.qdrant import QdrantStore
from src.core.embedding.generator import EmbeddingGenerator
from src.core.retrieval.bm25_search import BM25Search
from src.core.chunking.strategies import Chunker
from src.services.document_processor import DocumentProcessor

from api.v1.schemas import (
    DocumentUploadResponse,
    DocumentInfo,
    DocumentChunk,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
)
from api.v1.dependencies import (
    get_qdrant_store,
    get_embedding_generator,
    get_bm25_search,
    get_chunker,
    get_document_processor,
    get_upload_dir,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xlsx", ".html"}


async def _rebuild_bm25_index(
    qdrant_store: QdrantStore,
    bm25_search: BM25Search,
) -> int:
    """Rebuild the full BM25 index from all Qdrant documents."""
    all_docs = await qdrant_store.get_all_documents()
    chunks = [
        {
            "chunk_id": doc["chunk_id"],
            "content": doc["content"],
            "document_id": doc["document_id"],
            "metadata": doc["metadata"],
        }
        for doc in all_docs
    ]
    if chunks:
        return bm25_search.index(chunks)
    # Reset BM25 if no documents remain
    bm25_search._documents = []
    bm25_search._bm25 = None
    return 0


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
    embedding_gen: EmbeddingGenerator = Depends(get_embedding_generator),
    bm25_search: BM25Search = Depends(get_bm25_search),
    chunker: Chunker = Depends(get_chunker),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    upload_dir: str = Depends(get_upload_dir),
):
    """Upload a document, process, chunk, embed, and index it."""
    # Validate extension
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Save to temp file
    document_id = str(uuid.uuid4())
    temp_path = os.path.join(upload_dir, f"{document_id}{ext}")

    try:
        # Read and validate size
        content_bytes = await file.read()
        if len(content_bytes) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024*1024)} MB.",
            )

        with open(temp_path, "wb") as f:
            f.write(content_bytes)

        # Process document
        processed = doc_processor.process(temp_path)

        # Chunk
        chunks = chunker.chunk(
            text=processed.content,
            document_id=document_id,
            source_file=filename,
            metadata={
                "filename": filename,
                "page_count": processed.page_count,
                **processed.metadata,
            },
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No content could be extracted from the document.")

        # Embed
        texts = [c.content for c in chunks]
        embedding_result = await embedding_gen.embed_documents(texts)

        # Prepare for upsert
        ids = [c.chunk_id for c in chunks]
        metadata_list = [c.to_dict() for c in chunks]

        # Upsert to Qdrant
        await qdrant_store.upsert(
            ids=ids,
            embeddings=embedding_result.embeddings,
            metadata=metadata_list,
        )

        # Rebuild BM25 index
        await _rebuild_bm25_index(qdrant_store, bm25_search)

        logger.info(f"Uploaded document {filename} ({document_id}): {len(chunks)} chunks")

        return DocumentUploadResponse(
            document_id=document_id,
            filename=filename,
            chunks_count=len(chunks),
        )

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
):
    """List all documents with chunk counts."""
    all_docs = await qdrant_store.get_all_documents()

    # Group by document_id
    doc_map: dict[str, dict] = {}
    for doc in all_docs:
        doc_id = doc["document_id"]
        if not doc_id:
            continue
        if doc_id not in doc_map:
            doc_map[doc_id] = {
                "document_id": doc_id,
                "filename": doc["metadata"].get("filename", doc["metadata"].get("source_file", "unknown")),
                "chunks_count": 0,
            }
        doc_map[doc_id]["chunks_count"] += 1

    documents = [DocumentInfo(**info) for info in doc_map.values()]

    return DocumentListResponse(documents=documents, total=len(documents))


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: str,
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
):
    """Get document details with all chunks."""
    all_docs = await qdrant_store.get_all_documents()

    # Filter by document_id
    doc_chunks = [d for d in all_docs if d["document_id"] == document_id]

    if not doc_chunks:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    filename = doc_chunks[0]["metadata"].get(
        "filename", doc_chunks[0]["metadata"].get("source_file", "unknown")
    )

    chunks = [
        DocumentChunk(
            chunk_id=d["chunk_id"],
            content=d["content"],
            metadata=d["metadata"],
        )
        for d in doc_chunks
    ]

    return DocumentDetailResponse(
        document_id=document_id,
        filename=filename,
        chunks=chunks,
    )


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    qdrant_store: QdrantStore = Depends(get_qdrant_store),
    bm25_search: BM25Search = Depends(get_bm25_search),
):
    """Delete a document and all its chunks."""
    await qdrant_store.delete(filter={"document_id": document_id})

    # Rebuild BM25 index
    await _rebuild_bm25_index(qdrant_store, bm25_search)

    logger.info(f"Deleted document {document_id}")

    return DocumentDeleteResponse(document_id=document_id)
