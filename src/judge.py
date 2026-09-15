import sys
import os
import json
import re
from pydantic import BaseModel, Field
from transformers import pipeline
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class JudgeOutput(BaseModel):
    grounding_score: int = Field(description="Score 1-5 rating factual reliance on context.")
    tone_score: int = Field(description="Score 1-5 rating tone and length limits.")
    rationale: str = Field(description="Short rationale for awarded scores.")

class LLMJudge:
    def __init__(self, generator_pipeline=None, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        if generator_pipeline is not None:
            self.pipe = generator_pipeline
        else:
            print("Loading LLM-as-a-Judge Engine...")
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.float16
            else:
                device_map = None
                torch_dtype = torch.float32

            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch_dtype,
                device_map=device_map
            )

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

        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.pipe(
            prompt,
            max_new_tokens=150,
            do_sample=False,
            return_full_text=False
        )

        raw_text = outputs[0]["generated_text"].strip()

        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(0)

        parsed_json = json.loads(raw_text)
        return JudgeOutput(**parsed_json)