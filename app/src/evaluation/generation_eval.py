"""
Generation Evaluation using Ragas.

FOCUS: Faithfulness, Answer Relevancy, Context Precision/Recall
MUST: Run before production
TARGET: Faithfulness > 0.85, Answer Relevancy > 0.80
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env file

import warnings
warnings.filterwarnings("ignore", message=".*Importing.*from 'ragas.metrics'.*deprecated.*")

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)


def get_ragas_llm(model: str = "gpt-4o-mini") -> ChatOpenAI:
    """
    Get OpenAI LLM for Ragas evaluation.

    Requires OPENAI_API_KEY in .env

    Args:
        model: OpenAI model name (default: gpt-4o-mini)

    Returns:
        ChatOpenAI instance
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")

    logger.info(f"Using OpenAI LLM: {model}")
    return ChatOpenAI(model=model, api_key=api_key)


def get_ragas_embeddings(model: str = "text-embedding-3-small") -> OpenAIEmbeddings:
    """
    Get OpenAI embeddings for Ragas evaluation.

    Requires OPENAI_API_KEY in .env

    Args:
        model: OpenAI embedding model (default: text-embedding-3-small)

    Returns:
        OpenAIEmbeddings instance
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env")

    logger.info(f"Using OpenAI embeddings: {model}")
    return OpenAIEmbeddings(model=model, api_key=api_key)


@dataclass
class GenerationMetrics:
    """Ragas generation metrics."""

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }

    def check_targets(
        self,
        faithfulness_target: float = 0.85,
        relevancy_target: float = 0.80,
    ) -> dict[str, bool]:
        """Check if metrics meet production targets. NaN values are considered failures."""
        import math
        return {
            "faithfulness": not math.isnan(self.faithfulness) and self.faithfulness >= faithfulness_target,
            "answer_relevancy": not math.isnan(self.answer_relevancy) and self.answer_relevancy >= relevancy_target,
        }


@dataclass
class EvalSample:
    """Single evaluation sample for Ragas."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: Optional[str] = None


@dataclass
class GenerationDataset:
    """Dataset for generation evaluation."""

    name: str
    samples: list[EvalSample]

    @classmethod
    def from_json(cls, path: str) -> "GenerationDataset":
        with open(path) as f:
            data = json.load(f)
        samples = [
            EvalSample(
                question=s["question"],
                answer=s["answer"],
                contexts=s["contexts"],
                ground_truth=s.get("ground_truth"),
            )
            for s in data["samples"]
        ]
        return cls(name=data.get("name", Path(path).stem), samples=samples)

    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset for Ragas."""
        data = {
            "question": [s.question for s in self.samples],
            "answer": [s.answer for s in self.samples],
            "contexts": [s.contexts for s in self.samples],
        }
        if any(s.ground_truth for s in self.samples):
            data["ground_truth"] = [s.ground_truth or "" for s in self.samples]
        return Dataset.from_dict(data)


class GenerationEvaluator:
    """Evaluator for generation quality using Ragas with OpenAI."""

    def __init__(self, metrics: Optional[list] = None):
        """
        Initialize evaluator.

        Args:
            metrics: Ragas metrics to evaluate
        """
        self.metrics = metrics or [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def evaluate(
        self,
        dataset: GenerationDataset,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> GenerationMetrics:
        """
        Run Ragas evaluation using OpenAI.

        Args:
            dataset: Evaluation dataset
            llm_model: OpenAI LLM model name
            embedding_model: OpenAI embedding model name
        """
        hf_dataset = dataset.to_hf_dataset()

        llm = get_ragas_llm(llm_model)
        embeddings = get_ragas_embeddings(embedding_model)

        result = evaluate(hf_dataset, metrics=self.metrics, llm=llm, embeddings=embeddings)

        # Convert EvaluationResult to pandas DataFrame and get mean scores
        df = result.to_pandas()
        return GenerationMetrics(
            faithfulness=df["faithfulness"].mean() if "faithfulness" in df.columns else 0.0,
            answer_relevancy=df["answer_relevancy"].mean() if "answer_relevancy" in df.columns else 0.0,
            context_precision=df["context_precision"].mean() if "context_precision" in df.columns else 0.0,
            context_recall=df["context_recall"].mean() if "context_recall" in df.columns else 0.0,
        )

    def evaluate_from_results(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: Optional[list[str]] = None,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ) -> GenerationMetrics:
        """Evaluate from raw results using OpenAI."""
        samples = [
            EvalSample(
                question=q,
                answer=a,
                contexts=c,
                ground_truth=g if ground_truths else None,
            )
            for q, a, c, g in zip(
                questions,
                answers,
                contexts,
                ground_truths or [None] * len(questions),
            )
        ]
        dataset = GenerationDataset(name="inline", samples=samples)
        return self.evaluate(dataset, llm_model, embedding_model)


def create_synthetic_dataset(num_samples: int = 20) -> GenerationDataset:
    """Create synthetic dataset for testing."""
    samples = [
        EvalSample(
            question=f"What is topic {i}?",
            answer=f"Topic {i} is about subject {i}. It covers key aspects of area {i}.",
            contexts=[f"Topic {i} covers subject {i}.", f"Area {i} is related to topic {i}."],
            ground_truth=f"Topic {i} is about subject {i}.",
        )
        for i in range(num_samples)
    ]
    return GenerationDataset(name="synthetic", samples=samples)


def run_evaluation(
    dataset: GenerationDataset,
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    faithfulness_target: float = 0.85,
    relevancy_target: float = 0.80,
) -> tuple[GenerationMetrics, bool]:
    """Run evaluation with OpenAI and check against targets."""
    evaluator = GenerationEvaluator()
    metrics = evaluator.evaluate(dataset, llm_model, embedding_model)

    targets = metrics.check_targets(faithfulness_target, relevancy_target)
    passed = all(targets.values())

    logger.info("=" * 50)
    logger.info(f"Generation Evaluation Results for {dataset.name}")
    logger.info(f"Faithfulness:       {metrics.faithfulness:.4f} (target: {faithfulness_target})")
    logger.info(f"Answer Relevancy:   {metrics.answer_relevancy:.4f} (target: {relevancy_target})")
    logger.info(f"Context Precision:  {metrics.context_precision:.4f}")
    logger.info(f"Context Recall:     {metrics.context_recall:.4f}")
    logger.info(f"PASSED: {passed}")

    return metrics, passed


def _format_metric(value: float) -> str:
    """Format metric value, handling NaN."""
    import math
    if math.isnan(value):
        return "N/A"
    return f"{value:.4f}"


def print_report(metrics: GenerationMetrics, dataset_name: str = "dataset") -> None:
    """Print formatted evaluation report."""
    import math

    print("\n" + "=" * 60)
    print(f"GENERATION EVALUATION REPORT: {dataset_name}")
    print("=" * 60)

    print("\n[OpenAI Ragas Metrics]")
    print(f"  Faithfulness:       {_format_metric(metrics.faithfulness)}")
    print(f"  Answer Relevancy:   {_format_metric(metrics.answer_relevancy)}")
    print(f"  Context Precision:  {_format_metric(metrics.context_precision)}")
    print(f"  Context Recall:     {_format_metric(metrics.context_recall)}")

    print("\n[Target Check]")
    for name, passed in metrics.check_targets().items():
        value = getattr(metrics, name)
        if math.isnan(value):
            print(f"  {name}: N/A")
        else:
            print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Generation Evaluation Module")
    print("Running RAGAS evaluation...\n")

    # Create synthetic dataset
    dataset = create_synthetic_dataset(num_samples=5)
    print(f"Dataset: {len(dataset.samples)} samples")

    # Run actual evaluation
    metrics, passed = run_evaluation(dataset)

    # Print formatted report
    print_report(metrics, dataset.name)
