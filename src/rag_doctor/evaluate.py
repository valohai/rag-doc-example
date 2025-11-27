import json
import logging
import numpy as np
from pathlib import Path
import pandas as pd

import valohai
from langchain_openai import ChatOpenAI

from rag_doctor.consts import PROMPT_MODEL

log = logging.getLogger(__name__)


def evaluate_retrieval_quality(retrieved_contents: list, ground_truth: str, question: str, prompt_model) -> float:
    """Use LLM to judge if retrieved docs support the ground truth answer."""
    if not retrieved_contents:
        return 0.0
    
    context = "\n\n---\n\n".join(retrieved_contents[:3])[:3000]
    
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

    try:
        response = prompt_model.invoke(prompt)
        score = float(response.content.strip())
        return min(1.0, max(0.0, score))
    except Exception as e:
        log.warning(f"Failed to evaluate retrieval: {e}")
        return 0.0


def evaluate_responses(responses_dir: str) -> None:
    """Evaluate RAG responses using retrieval metrics and LLM-as-a-judge."""

    log.info("Loading response data for evaluation...")

    responses_file = Path(responses_dir) / "responses.json"
    if not responses_file.exists():
        raise ValueError(f"Response file not found: {responses_file}")

    with open(responses_file) as f:
        data = json.load(f)

    log.info(f"Evaluating {len(data)} responses")

    # Load ground truth
    gold_standards_file = valohai.inputs("gold_standards").path()
    gold_df = pd.read_csv(gold_standards_file)

    gold_lookup = {}
    for _, row in gold_df.iterrows():
        question = row['question'].strip().lower()
        gold_lookup[question] = {
            "answer": row['ground_truth_answer'],
            "source": row.get('source', ''),
        }

    # Initialize LLM judge
    prompt_model = ChatOpenAI(model=PROMPT_MODEL, temperature=0)

    # 1. RETRIEVAL METRICS (LLM-as-judge)
    recall_scores = []
    
    for d in data:
        question_lower = d["question"].strip().lower()
        gold = gold_lookup.get(question_lower)
        
        if not gold:
            log.warning(f"No ground truth for question: {d['question']}")
            continue
        
        retrieved_contents = d.get("retrieved_contents", [])
        ground_truth = gold["answer"]
        
        if retrieved_contents:
            recall = evaluate_retrieval_quality(
                retrieved_contents, ground_truth, d["question"], prompt_model
            )
            recall_scores.append(recall)
            log.info(f"Question: {d['question'][:50]}... -> Recall: {recall:.2%}")

    recall_at_k_score = np.mean(recall_scores) if recall_scores else 0.0

    valid_responses = [d for d in data if d.get("answer", "").strip()]
    response_rate = len(valid_responses) / len(data) if data else 0

    # 2. GENERATION METRICS - Factuality via LLM-as-a-judge (all substantive responses)
    factuality_scores = []
    substantive = [r for r in valid_responses if len(r.get("answer", "")) > 100]
    
    for response in substantive:
        factuality_prompt = f"""Rate the factual accuracy of this answer on a scale of 1-5 where:
1 = Completely inaccurate
2 = Mostly inaccurate  
3 = Somewhat accurate
4 = Mostly accurate
5 = Completely accurate

Question: {response["question"]}
Answer: {response["answer"]}

Rating (just return the number):"""

        try:
            message = prompt_model.invoke(factuality_prompt)
            score = float(message.content.strip())
            factuality_scores.append(score)
        except (ValueError, Exception) as e:
            log.warning(f"Failed to get factuality score: {e}")

    factuality_score = np.mean(factuality_scores) if factuality_scores else 0.0

    avg_length = np.mean([len(d.get("answer", "")) for d in data]) if data else 0
    
    substantive_responses = len(substantive)
    substantive_rate = substantive_responses / len(data) if data else 0

    # 3. OPERATIONAL METRICS
    estimated_latency = 1.2 + (avg_length * 0.001)  
    
    total_chars = sum(len(d.get("answer", "")) for d in data)
    estimated_tokens = total_chars // 4
    estimated_cost = (estimated_tokens * 0.000002) + (len(data) * 0.0001)

    # Log metrics to Valohai
    with valohai.logger() as logger:
        logger.log("response_rate", response_rate)
        logger.log("recall_at_k", recall_at_k_score)
        logger.log("factuality_score", factuality_score)
        logger.log("avg_response_length", avg_length)
        logger.log("substantive_rate", substantive_rate)
        logger.log("total_questions", len(data))
        logger.log("estimated_latency_seconds", round(estimated_latency, 3))
        logger.log("estimated_cost_usd", round(estimated_cost, 4))
        log.info("Evaluation metrics logged to execution metadata")
    
    print("Evaluation metrics logged to execution metadata")
    print()
    print("=== RETRIEVAL METRICS ===")
    print(f"Recall@K: {recall_at_k_score:.2%}")
    print(f"Response rate: {response_rate:.2%}")
    print()
    print("=== GENERATION METRICS ===")
    print(f"Factuality score: {factuality_score}/5")
    print(f"Average response length: {avg_length:.1f} characters")
    print(f"Substantive response rate: {substantive_rate:.2%}")
    print()
    print("=== OPERATIONAL METRICS ===")
    print(f"Total questions evaluated: {len(data)}")
    print(f"Estimated latency: {estimated_latency:.3f} seconds")
    print(f"Estimated cost: ${estimated_cost:.4f} USD")