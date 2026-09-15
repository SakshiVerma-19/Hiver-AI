import sys
import os
import re
import json
from pydantic import BaseModel
from transformers import pipeline
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.intent import IntentClassifier
from src.rag import RAGRetriever

class ExecutionResult(BaseModel):
    customer_message: str
    predicted_intent: str
    confidence_score: float
    escalated: bool
    escalation_reason: str | None = None
    retrieved_context: list[dict] | None = None
    final_response: str

class SupportAgent:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        print("Initializing Support Agent Pipeline...")
        
        # 1. Initialize Single Model Pipeline
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = None
            torch_dtype = torch.float32

        self.generator = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch_dtype,
            device_map=device_map
        )

        # Pass generator to IntentClassifier to share weights
        self.intent_classifier = IntentClassifier(generator_pipeline=self.generator)
        
        # 2. RAG Retriever (resolve paths relative to project root)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(base_dir, "chroma_db")
        sample_path = os.path.join(base_dir, "data", "raw_sample.csv")

        self.rag_retriever = RAGRetriever(db_path=db_path)
        self.rag_retriever.populate_index(sample_path)

        # 3. Guardrails
        self.pii_patterns = [
            r'\b(?:\d[ -]*?){13,16}\b',  # Credit Cards
            r'\b\d{3}-\d{2}-\d{4}\b'      # SSN
        ]
        self.legal_patterns = [
            r'\b(lawyer|lawsuit|sue|legal action|attorney)\b'
        ]

    def _check_escalation(self, text: str, confidence_score: float) -> tuple[bool, str | None]:
        for pattern in self.pii_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "PII_EXPOSURE_RISK"

        for pattern in self.legal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, "LEGAL_RISK_DETECTED"

        if confidence_score < 0.65:
            return True, "LOW_INTENT_CONFIDENCE"

        return False, None

    def process_ticket(self, customer_message: str) -> ExecutionResult:
        intent_res = self.intent_classifier.classify(customer_message)

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
                retrieved_context=None,
                final_response=f"Ticket escalated to human queue. Reason: {reason_code}"
            )

        contexts = self.rag_retriever.retrieve_context(customer_message, top_k=2)
        
        context_str = ""
        for i, ctx in enumerate(contexts, 1):
            context_str += f"\nExample {i}:\nCustomer: {ctx['historical_customer_query']}\nBrand: {ctx['historical_brand_response']}\n"

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
            return_full_text=False,
            clean_up_tokenization_spaces=False
        )

        raw_response = outputs[0]["generated_text"].strip()

        if len(raw_response) > 280:
            raw_response = raw_response[:277] + "..."

        return ExecutionResult(
            customer_message=customer_message,
            predicted_intent=intent_res.predicted_intent.value,
            confidence_score=intent_res.confidence_score,
            escalated=False,
            escalation_reason=None,
            retrieved_context=contexts,
            final_response=raw_response
        )

if __name__ == "__main__":
    agent = SupportAgent()
    sample_queries = [
        "My order has been delayed for 3 days, can you please help check tracking?",
        "I was charged twice on my credit card 4111-2222-3333-4444! Fix this!",
        "Fix this now or I will contact my lawyer and sue you!"
    ]

    for q in sample_queries:
        print(f"\nQuery: {q}")
        res = agent.process_ticket(q)
        print(f"Intent: {res.predicted_intent} (Confidence: {res.confidence_score:.2f})")
        print(f"Escalated: {res.escalated} (Reason: {res.escalation_reason})")
        print(f"Response: {res.final_response}")