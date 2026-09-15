import sys
import os
import json
import re
from sklearn.metrics import f1_score, cohen_kappa_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import SupportAgent
from src.judge import LLMJudge

def run_evaluation():
    print("Starting Automated Pipeline Benchmark...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    golden_set_path = os.path.join(base_dir, "data", "golden_set.json")

    with open(golden_set_path, "r") as f:
        golden_set = json.load(f)

    # Initialize agent and share generator pipeline with LLMJudge to save RAM/VRAM
    agent = SupportAgent()
    judge = LLMJudge(generator_pipeline=agent.generator)

    true_intents = []
    pred_intents = []
    
    human_grounding_scores = []
    judge_grounding_scores = []

    for item in golden_set:
        tweet = item["customer_tweet"]
        true_intent = item["true_intent"]
        
        # Process through Agent
        result = agent.process_ticket(tweet)
        
        true_intents.append(true_intent)
        pred_intents.append(result.predicted_intent)

        if not result.escalated:
            # Build real retrieved context string from RAG results
            context_text = ""
            if result.retrieved_context:
                for idx, ctx in enumerate(result.retrieved_context, 1):
                    context_text += f"Past Customer: {ctx['historical_customer_query']} | Past Resolution: {ctx['historical_brand_response']}\n"
            else:
                context_text = "No historical context retrieved."

            # Run LLM Judge
            judge_eval = judge.evaluate_response(
                customer_msg=tweet,
                response=result.final_response,
                retrieved_context=context_text.strip()
            )
            human_grounding_scores.append(item["human_grounding_score"])
            judge_grounding_scores.append(judge_eval.grounding_score)

    human_grounding_scores = [int(x) for x in human_grounding_scores]
    judge_grounding_scores = [int(x) for x in judge_grounding_scores]

    # Compute Metrics
    intent_f1 = f1_score(true_intents, pred_intents, average="weighted")
    
    if human_grounding_scores and judge_grounding_scores:
        try:
            kappa = cohen_kappa_score(human_grounding_scores, judge_grounding_scores)
        except Exception:
            kappa = 0.0
    else:
        kappa = 0.0

    print("\n================ BENCHMARK RESULTS ================")
    print(f"Intent Classification F1-Score (Weighted): {intent_f1:.2f}")
    print(f"Human vs. Judge Calibration (Cohen's Kappa): {kappa:.2f}")
    print("===================================================")
    print("Human Ground Truth:", human_grounding_scores)
    print("Judge Predictions:", judge_grounding_scores)

if __name__ == "__main__":
    run_evaluation()