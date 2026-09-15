import json
import os
from enum import Enum
from pydantic import BaseModel, Field
import torch
from transformers import pipeline

class IntentCategory(str, Enum):
    ORDER_TRACKING = "Order/Tracking Status"
    CANCELLATION_REFUND = "Cancellation/Refund Request"
    ACCOUNT_AUTH = "Account Access/Authentication"
    BILLING_PAYMENT = "Billing/Payment Issue"
    TECHNICAL_BUG = "Service Outage/Technical Bug"
    GENERAL_INQUIRY = "General Inquiry/Feedback"

class IntentOutput(BaseModel):
    predicted_intent: IntentCategory
    reasoning: str = Field(description="Brief explanation of why this intent category was selected.")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0.")

class IntentClassifier:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        print(f"Loading lightweight local model: {model_id}...")
        
        # Configure device and precision based on hardware
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = None
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch_dtype,
            device_map=device_map
        )

    def classify(self, customer_message: str) -> IntentOutput:
        schema_json = json.dumps(IntentOutput.model_json_schema(), indent=2)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert customer support intent classification engine.\n"
                    "Analyze the customer message and respond STRICTLY in JSON matching this schema:\n"
                    f"{schema_json}\n\n"
                    "Categories:\n"
                    "1. Order/Tracking Status\n"
                    "2. Cancellation/Refund Request\n"
                    "3. Account Access/Authentication\n"
                    "4. Billing/Payment Issue\n"
                    "5. Service Outage/Technical Bug\n"
                    "6. General Inquiry/Feedback\n"
                    "Return ONLY valid JSON."
                )
            },
            {"role": "user", "content": f"Customer Message: '{customer_message}'"}
        ]

        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0
        )

        raw_text = outputs[0]["generated_text"][len(prompt):].strip()
        
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].split("```")[0].strip()

        parsed_json = json.loads(raw_text)
        return IntentOutput(**parsed_json)

if __name__ == "__main__":
    classifier = IntentClassifier(model_id="Qwen/Qwen2.5-1.5B-Instruct")
    sample_text = "@AmazonHelp My package was supposed to arrive yesterday but the tracking number shows delayed."
    
    result = classifier.classify(sample_text)
    
    print("\n--- Intent Classifier Output ---")
    print(f"Predicted Intent: {result.predicted_intent.value}")
    print(f"Confidence Score: {result.confidence_score}")
    print(f"Reasoning:        {result.reasoning}")