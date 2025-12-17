import csv
import json
import logging
from pathlib import Path
from statistics import mean
from typing import TypedDict

import valohai
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from rag_doctor.consts import PROMPT_MODEL

log = logging.getLogger(__name__)

RETRIEVAL_K = 3  # Number of top retrieved chunks to evaluate for context coverage
CONTEXT_CHAR_LIMIT = 3000  # Max characters for LLM judge context to manage token costs


class ResponseData(TypedDict):
    question: str
    answer: str
    provider: str
    retrieved_contents: list[str]


class GoldStandard(TypedDict):
    answer: str
    source: str


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def evaluate_retrieval_quality(
    retrieved_contents: list,
    ground_truth: str,
    question: str,
    prompt_model,
) -> float:
    """Use LLM to judge if retrieved docs support the ground truth answer."""
    if not retrieved_contents:
        raise ValueError("retrieved_contents cannot be empty")

    context = "\n\n---\n\n".join(retrieved_contents[:RETRIEVAL_K])[:CONTEXT_CHAR_LIMIT]

    prompt = f"""You are evaluating a RAG system's retrieval quality.

Question: {question}

Retrieved Documents:
{context}

Ground Truth Answer:
{ground_truth}

Score from 0.0 to 1.0: What fraction of the information needed to produce the ground truth answer is present in the retrieved documents?
- 0.0 = Retrieved docs contain none of the needed information
- 0.5 = Retrieved docs contain about half the needed information
- 1.0 = Retrieved docs contain all the needed information

Return only a decimal number:"""

    response = prompt_model.invoke(prompt)
    score = float(response.content.strip())
    return min(1.0, max(0.0, score))


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def evaluate_factuality(
    question: str,
    answer: str,
    prompt_model,
) -> float:
    """Use LLM to judge the factual accuracy of an answer on a scale of 1-5."""
    prompt = f"""Rate the factual accuracy of this answer on a scale of 1-5 where:
1 = Completely inaccurate
2 = Mostly inaccurate
3 = Somewhat accurate
4 = Mostly accurate
5 = Completely accurate

Question: {question}
Answer: {answer}

Rating (just return the number):"""

    message = prompt_model.invoke(prompt)
    score = float(message.content.strip())
    return min(5.0, max(1.0, score))


def evaluate_provider(
    data: list[ResponseData],
    provider: str,
    gold_lookup: dict[str, GoldStandard],
    prompt_model,
) -> dict[str, float]:
    """Evaluate responses from a single provider."""

    print(f"\n=== EVALUATING {provider.upper()} PROVIDER ===")
    print(f"Processing {len(data)} responses...")

    # 1. RETRIEVAL METRICS (LLM-as-judge)
    coverage_scores = []

    for d in data:
        question_lower = d["question"].strip().lower()
        gold = gold_lookup.get(question_lower)

        if not gold:
            log.warning(f"No ground truth for question: {d['question']}")
            continue

        retrieved_contents = d.get("retrieved_contents", [])
        ground_truth = gold["answer"]

        if retrieved_contents:
            coverage = evaluate_retrieval_quality(
                retrieved_contents,
                ground_truth,
                d["question"],
                prompt_model,
            )
            coverage_scores.append(coverage)
            print(f"Question: {d['question'][:50]}... -> Context coverage: {coverage:.2%}")

    context_coverage_score = mean(coverage_scores) if coverage_scores else 0.0

    valid_responses = [d for d in data if d.get("answer", "").strip()]
    response_rate = len(valid_responses) / len(data) if data else 0

    # 2. GENERATION METRICS - Factuality via LLM-as-a-judge
    factuality_scores = []
    substantive = [r for r in valid_responses if len(r.get("answer", "")) > 100]

    for response in substantive:
        score = evaluate_factuality(
            response["question"],
            response["answer"],
            prompt_model,
        )
        if score > 0:
            factuality_scores.append(score)

    factuality_score = mean(factuality_scores) if factuality_scores else 0.0

    avg_length = mean([len(d.get("answer", "")) for d in data]) if data else 0

    substantive_responses = len(substantive)
    substantive_rate = substantive_responses / len(data) if data else 0

    # 3. OPERATIONAL METRICS
    estimated_latency = 1.2 + (avg_length * 0.001)

    total_chars = sum(len(d.get("answer", "")) for d in data)
    estimated_tokens = total_chars // 4
    estimated_cost = (estimated_tokens * 0.000002) + (len(data) * 0.0001)

    # Collect metrics for return
    metrics = {
        "response_rate": response_rate,
        "context_coverage": context_coverage_score,
        "factuality_score": factuality_score,
        "avg_response_length": avg_length,
        "substantive_rate": substantive_rate,
        "total_questions": len(data),
        "estimated_latency_seconds": round(estimated_latency, 3),
        "estimated_cost_usd": round(estimated_cost, 4),
    }

    with valohai.logger() as logger:
        for metric_name, value in metrics.items():
            logger.log(f"{metric_name}_{provider}", value)

    # Print organized summary
    print(f"\n=== {provider.upper()} RESULTS ===")
    print("RETRIEVAL METRICS:")
    print(f"  Context coverage: {context_coverage_score:.2%}")
    print(f"  Response rate: {response_rate:.2%}")
    print()
    print("GENERATION METRICS:")
    print(f"  Factuality score: {factuality_score}/5")
    print(f"  Average response length: {avg_length:.1f} characters")
    print(f"  Substantive response rate: {substantive_rate:.2%}")
    print()
    print("OPERATIONAL METRICS:")
    print(f"  Total questions evaluated: {len(data)}")
    print(f"  Estimated latency: {estimated_latency:.3f} seconds")
    print(f"  Estimated cost: ${estimated_cost:.4f} USD")
    print(f"=== END {provider.upper()} RESULTS ===")

    return metrics


def evaluate_responses(responses_dir: str) -> None:
    """Evaluate RAG responses from multiple providers."""

    log.info("Loading response data for evaluation...")

    responses_path = Path(responses_dir)

    # Find all JSON files
    json_files = list(responses_path.glob("*.json"))

    print(f"Found {len(json_files)} response file(s) to evaluate")

    # Load ground truth once
    gold_standards_file = valohai.inputs("gold_standards").path()
    gold_lookup: dict[str, GoldStandard] = {}
    with open(gold_standards_file) as f:
        for row in csv.DictReader(f):
            question = row["question"].strip().lower()
            gold_lookup[question] = {
                "answer": row["ground_truth_answer"],
                "source": row.get("source", ""),
            }

    # LLM judge
    prompt_model = ChatOpenAI(model=PROMPT_MODEL, temperature=0)

    all_metrics = {}

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        if not data:
            print(f"Skipping empty file: {json_file.name}")
            continue

        provider = data[0].get("provider", json_file.stem)

        metrics = evaluate_provider(data, provider, gold_lookup, prompt_model)
        all_metrics[provider] = metrics

    print(f"\nEvaluation complete! Processed {len(all_metrics)} provider(s)")
    log.info("All evaluation metrics logged to execution metadata")
