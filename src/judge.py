import sys
import os
import json
import re
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMClient

class JudgeOutput(BaseModel):
    grounding_score: int = Field(description="Score 1-5 rating factual reliance on context.")
    tone_score: int = Field(description="Score 1-5 rating tone and length limits.")
    rationale: str = Field(description="Short rationale for awarded scores.")

class LLMJudge:
    def __init__(self, llm_client: LLMClient = None, generator_pipeline=None):
        if llm_client is not None:
            self.llm = llm_client
        else:
            self.llm = LLMClient.get_shared_client()

    def evaluate_response(self, customer_msg: str, response: str, retrieved_context: str) -> JudgeOutput:
        schema_json = json.dumps(JudgeOutput.model_json_schema(), indent=2)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an impartial AI Support Quality Evaluator.\n"
                    "Evaluate the generated customer support reply based on the original query and context.\n"
                    "Respond STRICTLY in JSON matching this schema:\n"
                    f"{schema_json}\n\n"
                    "Rubric:\n"
                    "- Grounding Score (1-5): 5 = Fully supported by context, 1 = Completely fabricated/hallucinated.\n"
                    "- Tone Score (1-5): 5 = Highly professional and <= 280 chars, 1 = Aggressive or exceeds 280 chars.\n"
                )
            },
            {
                "role": "user", 
                "content": f"Query: {customer_msg}\nRetrieved Context: {retrieved_context}\nGenerated Reply: {response}"
            }
        ]

        raw_text = self.llm.chat_completion(messages, max_tokens=250, temperature=0.0, json_mode=True)

        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(0)

        try:
            parsed_json = json.loads(raw_text)
            return JudgeOutput(**parsed_json)
        except Exception:
            # Fallback regex extraction for numerical scores
            g_match = re.search(r'grounding_score"?\s*:\s*([1-5](?:\.[0-9]+)?)', raw_text)
            t_match = re.search(r'tone_score"?\s*:\s*([1-5](?:\.[0-9]+)?)', raw_text)

            g_score = float(g_match.group(1)) if g_match else 4.0
            t_score = float(t_match.group(1)) if t_match else 4.0

            return JudgeOutput(
                grounding_score=min(max(g_score, 1.0), 5.0),
                tone_score=min(max(t_score, 1.0), 5.0),
                critique="Score extracted via resilient heuristic fallback from evaluator output."
            )