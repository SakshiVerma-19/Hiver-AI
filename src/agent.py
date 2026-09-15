import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import json
from pydantic import BaseModel
from transformers import pipeline
import torch

from src.intent import IntentClassifier
from src.rag import RAGRetriever

class ExecutionResult(BaseModel):
    customer_message: str
    predicted_intent: str
    confidence_score: float
    escalated: bool
    escalation_reason: str | None = None
    final_response: str

class SupportAgent:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        print("Initializing Support Agent Pipeline...")
        self.intent_classifier = IntentClassifier(model_id=model_id)
        
        self.rag_retriever = RAGRetriever()
        self.rag_retriever.populate_index("data/raw_sample.csv")

        # Reuse local text generator
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = None
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

        self.generator = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch_dtype,
            device_map=device_map
        )

        # Regex Guardrails
        self.pii_patterns = [
            r'\b(?:\d[ -]*?){13,16}\b',  # Credit Cards
            r'\b\d{3}-\d{2}-\d{4}\b'      # SSN
        ]
        self.legal_patterns = [
            r'\b(lawyer|lawsuit|sue|legal action|attorney)\b'
        ]

    def _check_escalation(self, text: str, confidence_score: float) -> tuple[bool, str | None]:
        """Dual-Trigger Escalation Logic."""
        # 1. PII Regex Check
        for pattern in self.pii_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "PII_EXPOSURE_RISK"

        # 2. Legal Threat Check
        for pattern in self.legal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "LEGAL_RISK_DETECTED"

        # 3. Confidence Threshold Check (< 0.65)
        if confidence_score < 0.65:
            return True, "LOW_INTENT_CONFIDENCE"

        return False, None

    def process_ticket(self, customer_message: str) -> ExecutionResult:
        # Step 1: Classify Intent
        intent_res = self.intent_classifier.classify(customer_message)

        # Step 2: Safety & Guardrail Checks
        should_escalate, reason_code = self._check_escalation(
            customer_message, intent_res.confidence_score
        )

        if should_escalate:
            return ExecutionResult(
                customer_message=customer_message,
                predicted_intent=intent_res.predicted_intent.value,
                confidence_score=intent_res.confidence_score,
                escalated=True,
                escalation_reason=reason_code,
                final_response=f"Ticket escalated to human queue. Reason: {reason_code}"
            )

        # Step 3: Retrieve Context via RAG
        contexts = self.rag_retriever.retrieve_context(customer_message, top_k=2)
        
        context_str = ""
        for i, ctx in enumerate(contexts, 1):
            context_str += f"\nExample {i}:\nCustomer: {ctx['historical_customer_query']}\nBrand: {ctx['historical_brand_response']}\n"

        # Step 4: Grounded Response Generation
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are @AmazonHelp official Twitter support agent.\n"
                    "Draft a helpful, professional tweet response based ONLY on the verified policy history below.\n"
                    "Strict Constraints:\n"
                    "1. Response length MUST NOT exceed 280 characters.\n"
                    "2. Do not invent unverified claims or policies.\n\n"
                    f"Verified Policy History Context:\n{context_str}"
                )
            },
            {"role": "user", "content": customer_message}
        ]

        formatted_prompt = self.generator.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.generator(
            formatted_prompt,
            max_new_tokens=120,
            do_sample=False,
            temperature=0.0
        )

        raw_response = outputs[0]["generated_text"][len(formatted_prompt):].strip()

        # Direct 280-character boundary check
        if len(raw_response) > 280:
            raw_response = raw_response[:277] + "..."

        return ExecutionResult(
            customer_message=customer_message,
            predicted_intent=intent_res.predicted_intent.value,
            confidence_score=intent_res.confidence_score,
            escalated=False,
            escalation_reason=None,
            final_response=raw_response
        )

if __name__ == "__main__":
    agent = SupportAgent()

    # Test 1: Standard customer query
    res1 = agent.process_ticket("@AmazonHelp My package is delayed, tracking number shows no movement")
    print("\n--- Test 1 (Standard Flow Output) ---")
    print(res1.model_dump_json(indent=2))

    # Test 2: Legal threat trigger
    res2 = agent.process_ticket("@AmazonHelp I am hiring a lawyer and will sue if my item is not delivered today!")
    print("\n--- Test 2 (Escalation Trigger Output) ---")
    print(res2.model_dump_json(indent=2))