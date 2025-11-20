import json
import logging
import numpy as np
from pathlib import Path

import valohai
from langchain_openai import ChatOpenAI

from rag_doctor.consts import PROMPT_MODEL

log = logging.getLogger(__name__)


def evaluate_responses(responses_dir: str) -> None:
    """Evaluate RAG responses using retrieval metrics and LLM-as-a-judge."""

    log.info("Loading response data for evaluation...")

    # Load the responses from Valohai input
    responses_file = Path(responses_dir) / "responses.json"
    if not responses_file.exists():
        raise ValueError(f"Response file not found: {responses_file}")

    with open(responses_file) as f:
        data = json.load(f)

    log.info(f"Evaluating {len(data)} responses")

    # 1. RETRIEVAL METRICS 
    # (proxy metrics using available data)
    valid_responses = [d for d in data if d.get("answer", "").strip()]
    response_rate = len(valid_responses) / len(data) if data else 0
    
    # Proxy for retrieval quality: responses that show confidence vs uncertainty
    confident_responses = sum(1 for d in data 
                             if not any(phrase in d.get("answer", "").lower() 
                                       for phrase in ["don't know", "not sure", "unclear", "i'm not", "cannot find"]))
    confidence_rate = confident_responses / len(data) if data else 0

    # 2. GENERATION METRICS
    # Factuality via LLM-as-a-judge (sample evaluation)
    if valid_responses:
        prompt_model = ChatOpenAI(model=PROMPT_MODEL, temperature=0)
        sample_response = valid_responses[0]

        factuality_prompt = f"""Rate the factual accuracy of this answer on a scale of 1-5 where:
1 = Completely inaccurate
2 = Mostly inaccurate  
3 = Somewhat accurate
4 = Mostly accurate
5 = Completely accurate

Question: {sample_response["question"]}
Answer: {sample_response["answer"]}

Rating (just return the number):"""

        try:
            message = prompt_model.invoke(factuality_prompt)
            factuality_score = float(message.content.strip())
        except (ValueError, Exception) as e:
            log.warning(f"Failed to get factuality score: {e}")
            factuality_score = 0.0
    else:
        factuality_score = 0.0

    # Response quality metrics
    avg_length = np.mean([len(d.get("answer", "")) for d in data]) if data else 0
    
    # Completeness: responses that provide substantive answers
    substantive_responses = sum(1 for d in data if len(d.get("answer", "")) > 50)
    substantive_rate = substantive_responses / len(data) if data else 0

    # 3. OPERATIONAL METRICS
    # Estimate latency (would be measured in real implementation)
    estimated_latency = 1.2 + (avg_length * 0.001)  # Longer responses take slightly more time
    
    # Estimate cost based on token usage
    total_chars = sum(len(d.get("answer", "")) for d in data)
    estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
    estimated_cost = (estimated_tokens * 0.000002) + (len(data) * 0.0001)  # Token cost + retrieval cost

    # Log metrics to Valohai
    with valohai.logger() as logger:
        logger.log("response_rate", response_rate)
        logger.log("confidence_rate", confidence_rate)
        logger.log("factuality_score", factuality_score)
        logger.log("avg_response_length", avg_length)
        logger.log("substantive_rate", substantive_rate)
        logger.log("total_questions", len(data))
        logger.log("estimated_latency_seconds", round(estimated_latency, 3))
        logger.log("estimated_cost_usd", round(estimated_cost, 4))
        log.info("Evaluation metrics logged to execution metadata")
    
    # Log organized summary
    print("Evaluation metrics logged to execution metadata")
    print()
    print("=== RETRIEVAL METRICS ===")
    print(f"Response rate: {response_rate:.2%}")
    print(f"Confidence rate: {confidence_rate:.2%}")
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