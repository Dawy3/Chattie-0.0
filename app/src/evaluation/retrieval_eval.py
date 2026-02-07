"""
Retrieval Evaluation - RAG Retrieval Benchmark.

Supports TWO evaluation modes:
1. Vector-only: Basic vector search via VectorSearch (baseline)
2. Full Pipeline: Hybrid Search (Vector + BM25 + RRF fusion)

Usage:
    python -m app.src.evaluation.retrieval_eval --evaluate
    python -m app.src.evaluation.retrieval_eval --evaluate --full-pipeline
    python -m app.src.evaluation.retrieval_eval --compare
"""

import argparse
import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.config import settings
from ..core.embedding.generator import EmbeddingGenerator
from ..core.retrieval.vector_search import VectorSearch
from ..core.retrieval.bm25_search import BM25Search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_SETS_DIR = Path(__file__).parent / "test_sets"
BENCHMARK_QUERIES_PATH = TEST_SETS_DIR / "benchmark_queries.json"
BENCHMARK_DOCUMENTS_PATH = TEST_SETS_DIR / "benchmark_documents.json"


@dataclass
class BenchmarkQuery:
    """Benchmark query with ground truth."""
    id: str
    query: str
    category: str
    difficulty: str
    relevant_doc_ids: list[str]
    ground_truth: str
    expected_behavior: Optional[str] = None


@dataclass
class BenchmarkDocument:
    """Document in the benchmark corpus."""
    id: str
    title: str
    category: str
    content: str
    keywords: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Retrieval evaluation results."""
    total_queries: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    hits_at_5: int
    hits_at_10: int
    by_category: dict
    by_difficulty: dict
    failed_queries: list
    mode: str = "vector_only"


def load_benchmark_queries() -> list[BenchmarkQuery]:
    """Load benchmark queries from JSON file."""
    with open(BENCHMARK_QUERIES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = []
    for q in data["queries"]:
        queries.append(BenchmarkQuery(
            id=q["id"],
            query=q["query"],
            category=q["category"],
            difficulty=q.get("difficulty", "medium"),
            relevant_doc_ids=q.get("relevant_doc_ids", []),
            ground_truth=q.get("ground_truth", ""),
            expected_behavior=q.get("expected_behavior"),
        ))

    logger.info(f"Loaded {len(queries)} benchmark queries")
    return queries


def load_benchmark_documents() -> list[BenchmarkDocument]:
    """Load benchmark documents from JSON file."""
    with open(BENCHMARK_DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for doc in data["documents"]:
        documents.append(BenchmarkDocument(
            id=doc["id"],
            title=doc["title"],
            category=doc["category"],
            content=doc["content"],
            keywords=doc.get("keywords", []),
        ))

    logger.info(f"Loaded {len(documents)} benchmark documents")
    return documents


def calculate_ndcg(relevance_scores: list[int], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    def dcg(scores: list[int]) -> float:
        return sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(scores[:k])
        )

    actual_dcg = dcg(relevance_scores)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True))

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def _rrf_fusion(
    vector_ids: list[str],
    bm25_ids: list[str],
    rrf_k: int = 60,
) -> list[str]:
    """
    Reciprocal Rank Fusion — same formula as HybridSearch._rrf_fusion.

    RRF(d) = 1/(k + rank_vector + 1) + 1/(k + rank_bm25 + 1)
    """
    vector_ranks = {doc_id: i for i, doc_id in enumerate(vector_ids)}
    bm25_ranks = {doc_id: i for i, doc_id in enumerate(bm25_ids)}

    all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

    scores = {}
    for doc_id in all_ids:
        rrf_score = 0.0
        if doc_id in vector_ranks:
            rrf_score += 1.0 / (rrf_k + vector_ranks[doc_id] + 1)
        if doc_id in bm25_ranks:
            rrf_score += 1.0 / (rrf_k + bm25_ranks[doc_id] + 1)
        scores[doc_id] = rrf_score

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_results]


class RetrievalEvaluator:
    """
    Retrieval Benchmark Evaluator.

    Uses core/retrieval components directly:
    - VectorSearch for semantic search (Qdrant)
    - BM25Search for keyword search
    - RRF fusion for hybrid (same formula as HybridSearch)
    """

    def __init__(
        self,
        corpus_collection: str = "benchmark_corpus",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        self.corpus_collection = corpus_collection
        self.qdrant_url = qdrant_url or settings.qdrant.qdrant_url
        self.qdrant_api_key = qdrant_api_key or settings.qdrant.qdrant_api_key

        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.vector_search: Optional[VectorSearch] = None
        self.bm25_search: Optional[BM25Search] = None

    def initialize(self):
        """Initialize embedding generator and vector search."""
        logger.info("Initializing retrieval evaluator...")

        self.embedding_generator = EmbeddingGenerator()

        # VectorSearch connects in __init__ (sync)
        self.vector_search = VectorSearch(
            collection_name=self.corpus_collection,
            dimensions=self.embedding_generator.model.dimensions,
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )

        logger.info(f"Initialized with collection: {self.corpus_collection}")

    def initialize_full_pipeline(self):
        """Initialize BM25 for hybrid search."""
        if not self.embedding_generator:
            self.initialize()

        logger.info("Initializing full retrieval pipeline...")

        documents = load_benchmark_documents()

        # Build BM25 index — uses original doc.id as chunk_id
        logger.info("Building BM25 index...")
        self.bm25_search = BM25Search()

        bm25_docs = []
        for doc in documents:
            bm25_docs.append({
                "chunk_id": doc.id,
                "content": f"{doc.title}\n\n{doc.content}",
                "document_id": doc.id,
                "metadata": {
                    "title": doc.title,
                    "category": doc.category,
                    "keywords": doc.keywords,
                },
            })

        self.bm25_search.index(bm25_docs)
        logger.info(f"BM25 index built with {len(bm25_docs)} documents")

    async def ingest_corpus(self) -> dict:
        """Ingest benchmark documents into VectorSearch."""
        if not self.embedding_generator or not self.vector_search:
            self.initialize()

        documents = load_benchmark_documents()

        if not documents:
            logger.warning("No documents to ingest")
            return {"total": 0}

        doc_texts = [f"{doc.title}\n\n{doc.content}" for doc in documents]

        logger.info(f"Generating embeddings for {len(doc_texts)} documents...")
        result = await self.embedding_generator.embed_texts(doc_texts)

        # VectorSearch.index() needs integer chunk_ids for Qdrant.
        # Original doc IDs go in document_id (payload), used for result mapping.
        chunks = []
        for i, doc in enumerate(documents):
            chunks.append({
                "chunk_id": i + 1,
                "content": f"{doc.title}\n\n{doc.content}",
                "document_id": doc.id,
                "metadata": {
                    "title": doc.title,
                    "category": doc.category,
                    "keywords": doc.keywords,
                },
            })

        logger.info(f"Indexing {len(chunks)} documents...")
        count = self.vector_search.index(chunks, result.embeddings)

        logger.info(f"Corpus ingestion complete: {count} documents")

        return {
            "total_documents": count,
            "embedding_model": result.model_id,
            "dimensions": result.dimensions,
            "collection": self.corpus_collection,
        }

    def _search_vector_only(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[str]:
        """Vector-only search (baseline). Returns list of document_ids."""
        results = self.vector_search.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )
        return [r.document_id for r in results]

    def _search_full_pipeline(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[str]:
        """
        Full pipeline — matches HybridSearch logic:
        1. Vector Search (top-100)
        2. BM25 Search (top-100)
        3. RRF fusion: 1/(k + rank + 1)
        4. Return top-k document_ids
        """
        # Step 1: Vector search — extract document_ids from results
        vector_results = self.vector_search.search(
            query_embedding=query_embedding,
            top_k=100,
        )
        vector_doc_ids = [r.document_id for r in vector_results]

        # Step 2: BM25 search — extract document_ids from results
        bm25_results = self.bm25_search.search(query=query, top_k=100)
        bm25_doc_ids = [r.document_id for r in bm25_results]

        # Step 3: RRF fusion on document_ids
        fused_ids = _rrf_fusion(vector_doc_ids, bm25_doc_ids, rrf_k=60)

        return fused_ids[:top_k]

    async def evaluate(
        self,
        top_k: int = 10,
        include_edge_cases: bool = True,
        use_full_pipeline: bool = False,
    ) -> EvaluationResult:
        """
        Run retrieval evaluation.

        Args:
            top_k: Maximum K for retrieval
            include_edge_cases: Include edge case queries
            use_full_pipeline: Use full pipeline (Hybrid RRF) vs vector-only
        """
        if use_full_pipeline:
            self.initialize_full_pipeline()
            mode = "full_pipeline"
            logger.info("Evaluating with FULL PIPELINE (Vector + BM25 + RRF)")
        else:
            if not self.embedding_generator or not self.vector_search:
                self.initialize()
            mode = "vector_only"
            logger.info("Evaluating with VECTOR-ONLY search")

        queries = load_benchmark_queries()

        if not include_edge_cases:
            queries = [q for q in queries if q.category != "edge_case"]

        results = []
        failed_queries = []

        logger.info(f"Evaluating {len(queries)} queries...")

        for i, query in enumerate(queries):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(queries)} queries")

            if not query.relevant_doc_ids:
                continue

            try:
                query_embedding = await self.embedding_generator.embed_query(query.query)

                if use_full_pipeline:
                    retrieved_ids = self._search_full_pipeline(
                        query=query.query,
                        query_embedding=query_embedding,
                        top_k=top_k,
                    )
                else:
                    retrieved_ids = self._search_vector_only(
                        query_embedding=query_embedding,
                        top_k=top_k,
                    )

                # Metrics
                hit_at_5 = any(doc_id in retrieved_ids[:5] for doc_id in query.relevant_doc_ids)
                hit_at_10 = any(doc_id in retrieved_ids[:10] for doc_id in query.relevant_doc_ids)

                rr = 0.0
                for doc_id in query.relevant_doc_ids:
                    if doc_id in retrieved_ids:
                        rank = retrieved_ids.index(doc_id) + 1
                        rr = max(rr, 1.0 / rank)
                        break

                relevance_scores = [
                    1 if doc_id in query.relevant_doc_ids else 0
                    for doc_id in retrieved_ids
                ]

                results.append({
                    "query_id": query.id,
                    "query": query.query,
                    "category": query.category,
                    "difficulty": query.difficulty,
                    "hit_at_5": hit_at_5,
                    "hit_at_10": hit_at_10,
                    "reciprocal_rank": rr,
                    "relevance_scores": relevance_scores,
                })

            except Exception as e:
                logger.warning(f"Failed to evaluate query {query.id}: {e}")
                failed_queries.append({
                    "query_id": query.id,
                    "error": str(e),
                })

        # Aggregate metrics
        n = len(results)
        if n == 0:
            return EvaluationResult(
                total_queries=0,
                recall_at_5=0, recall_at_10=0, mrr=0, ndcg_at_10=0,
                hits_at_5=0, hits_at_10=0,
                by_category={}, by_difficulty={},
                failed_queries=failed_queries,
                mode=mode,
            )

        recall_at_5 = sum(1 for r in results if r["hit_at_5"]) / n
        recall_at_10 = sum(1 for r in results if r["hit_at_10"]) / n
        mrr = sum(r["reciprocal_rank"] for r in results) / n
        ndcg_at_10 = sum(calculate_ndcg(r["relevance_scores"], 10) for r in results) / n

        by_category = _breakdown(results, "category")
        by_difficulty = _breakdown(results, "difficulty")

        return EvaluationResult(
            total_queries=n,
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            mrr=mrr,
            ndcg_at_10=ndcg_at_10,
            hits_at_5=sum(1 for r in results if r["hit_at_5"]),
            hits_at_10=sum(1 for r in results if r["hit_at_10"]),
            by_category=by_category,
            by_difficulty=by_difficulty,
            failed_queries=failed_queries,
            mode=mode,
        )

    async def compare_modes(self, top_k: int = 10) -> dict:
        """Compare vector-only vs full pipeline."""
        logger.info("=" * 60)
        logger.info("COMPARING: Vector-Only vs Full Pipeline")
        logger.info("=" * 60)

        logger.info("\n[1/2] Evaluating Vector-Only...")
        vector_result = await self.evaluate(top_k=top_k, use_full_pipeline=False)

        logger.info("\n[2/2] Evaluating Full Pipeline...")
        pipeline_result = await self.evaluate(top_k=top_k, use_full_pipeline=True)

        improvements = {
            "recall_at_5": pipeline_result.recall_at_5 - vector_result.recall_at_5,
            "recall_at_10": pipeline_result.recall_at_10 - vector_result.recall_at_10,
            "mrr": pipeline_result.mrr - vector_result.mrr,
            "ndcg_at_10": pipeline_result.ndcg_at_10 - vector_result.ndcg_at_10,
        }

        return {
            "vector_only": {
                "recall_at_5": vector_result.recall_at_5,
                "recall_at_10": vector_result.recall_at_10,
                "mrr": vector_result.mrr,
                "ndcg_at_10": vector_result.ndcg_at_10,
            },
            "full_pipeline": {
                "recall_at_5": pipeline_result.recall_at_5,
                "recall_at_10": pipeline_result.recall_at_10,
                "mrr": pipeline_result.mrr,
                "ndcg_at_10": pipeline_result.ndcg_at_10,
            },
            "improvements": improvements,
            "improvement_percentages": {
                k: f"+{v*100:.1f}%" if v > 0 else f"{v*100:.1f}%"
                for k, v in improvements.items()
            },
        }


def _breakdown(results: list[dict], key: str) -> dict:
    """Calculate per-group metrics breakdown."""
    groups = {}
    for r in results:
        group = r[key]
        if group not in groups:
            groups[group] = {"total": 0, "hits_5": 0, "hits_10": 0, "rr_sum": 0.0}
        groups[group]["total"] += 1
        if r["hit_at_5"]:
            groups[group]["hits_5"] += 1
        if r["hit_at_10"]:
            groups[group]["hits_10"] += 1
        groups[group]["rr_sum"] += r["reciprocal_rank"]

    for group in groups:
        t = groups[group]["total"]
        groups[group]["recall_at_5"] = groups[group]["hits_5"] / t
        groups[group]["recall_at_10"] = groups[group]["hits_10"] / t
        groups[group]["mrr"] = groups[group]["rr_sum"] / t

    return groups


def format_evaluation_report(result: EvaluationResult) -> str:
    """Format evaluation results as a readable report."""
    mode_label = "FULL PIPELINE (Vector + BM25 + RRF)" if result.mode == "full_pipeline" else "VECTOR-ONLY"

    lines = [
        "=" * 60,
        f"RETRIEVAL EVALUATION REPORT - {mode_label}",
        "=" * 60,
        "",
        "OVERALL METRICS",
        "-" * 40,
        f"Total Queries:     {result.total_queries}",
        f"Recall@5:          {result.recall_at_5:.2%}",
        f"Recall@10:         {result.recall_at_10:.2%}",
        f"MRR:               {result.mrr:.4f}",
        f"NDCG@10:           {result.ndcg_at_10:.4f}",
        f"Hits@5:            {result.hits_at_5}/{result.total_queries}",
        f"Hits@10:           {result.hits_at_10}/{result.total_queries}",
        "",
        "BY CATEGORY",
        "-" * 40,
    ]

    for cat, metrics in sorted(result.by_category.items()):
        lines.append(
            f"{cat:20} R@5={metrics['recall_at_5']:.2%} "
            f"R@10={metrics['recall_at_10']:.2%} "
            f"MRR={metrics['mrr']:.3f} (n={metrics['total']})"
        )

    lines.extend(["", "BY DIFFICULTY", "-" * 40])

    for diff, metrics in sorted(result.by_difficulty.items()):
        lines.append(
            f"{diff:20} R@5={metrics['recall_at_5']:.2%} "
            f"R@10={metrics['recall_at_10']:.2%} "
            f"MRR={metrics['mrr']:.3f} (n={metrics['total']})"
        )

    if result.failed_queries:
        lines.extend(["", "FAILED QUERIES", "-" * 40])
        for fq in result.failed_queries[:5]:
            lines.append(f"  {fq['query_id']}: {fq['error']}")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


def format_comparison_report(comparison: dict) -> str:
    """Format comparison results."""
    lines = [
        "=" * 60,
        "COMPARISON: Vector-Only vs Full Pipeline (RRF)",
        "=" * 60,
        "",
        f"{'Metric':<20} {'Vector-Only':>15} {'Full Pipeline':>15} {'Improvement':>15}",
        "-" * 65,
    ]

    metrics = ["recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]
    labels = ["Recall@5", "Recall@10", "MRR", "NDCG@10"]

    for metric, label in zip(metrics, labels):
        v = comparison["vector_only"][metric]
        p = comparison["full_pipeline"][metric]
        imp = comparison["improvement_percentages"][metric]
        lines.append(f"{label:<20} {v:>14.2%} {p:>14.2%} {imp:>15}")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG Retrieval Benchmark")
    parser.add_argument("--ingest-corpus", action="store_true", help="Ingest benchmark documents")
    parser.add_argument("--evaluate", action="store_true", help="Run retrieval evaluation")
    parser.add_argument("--full-pipeline", action="store_true", help="Use full pipeline (Vector + BM25 + RRF)")
    parser.add_argument("--compare", action="store_true", help="Compare vector-only vs full pipeline")
    parser.add_argument("--collection", type=str, default="benchmark_corpus", help="Collection name")
    parser.add_argument("--qdrant-url", type=str, help="Qdrant URL")
    parser.add_argument("--top-k", type=int, default=10, help="Top K for evaluation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    evaluator = RetrievalEvaluator(
        corpus_collection=args.collection,
        qdrant_url=args.qdrant_url,
    )

    try:
        if args.ingest_corpus:
            evaluator.initialize()
            stats = await evaluator.ingest_corpus()
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print(f"\nCorpus ingested: {stats['total_documents']} documents")

        if args.compare:
            comparison = await evaluator.compare_modes(top_k=args.top_k)
            if args.json:
                print(json.dumps(comparison, indent=2))
            else:
                print(format_comparison_report(comparison))

        elif args.evaluate:
            result = await evaluator.evaluate(
                top_k=args.top_k,
                use_full_pipeline=args.full_pipeline,
            )
            if args.json:
                print(json.dumps({
                    "mode": result.mode,
                    "total_queries": result.total_queries,
                    "recall_at_5": result.recall_at_5,
                    "recall_at_10": result.recall_at_10,
                    "mrr": result.mrr,
                    "ndcg_at_10": result.ndcg_at_10,
                }, indent=2))
            else:
                print(format_evaluation_report(result))

        if not any([args.ingest_corpus, args.evaluate, args.compare]):
            parser.print_help()

    finally:
        pass  # VectorSearch has no async close

if __name__ == "__main__":
    asyncio.run(main())
