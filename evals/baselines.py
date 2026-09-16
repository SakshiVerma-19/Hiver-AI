import re
from pydantic import BaseModel
from src.llm import LLMClient

class BaselineResult(BaseModel):
    customer_message: str
    predicted_intent: str
    escalated: bool
    escalation_reason: str | None = None
    final_response: str

class TrivialBaseline:
    """
    Baseline 1: Trivial Baseline
    - Always predicts the majority class (Order/Tracking Status).
    - Returns a standard canned macro response.
    - Auto-handles all tickets (never escalates).
    """
    def __init__(self):
        self.majority_intent = "Order/Tracking Status"
        self.canned_reply = (
            "Thank you for contacting @AmazonHelp! Please send us a direct message with "
            "your order details and email address so our team can assist you further."
        )

    def process_ticket(self, customer_message: str) -> BaselineResult:
        return BaselineResult(
            customer_message=customer_message,
            predicted_intent=self.majority_intent,
            escalated=False,
            escalation_reason=None,
            final_response=self.canned_reply
        )

class SimpleZeroShotBaseline:
    """
    Baseline 2: Simple Zero-Shot (No RAG) Baseline
    - Simple zero-shot prompt for intent classification.
    - Generates reply directly without historical context grounding (No RAG).
    - Regex-only escalation detection (no confidence threshold or context validation).
    """
    def __init__(self, llm_client: LLMClient = None, generator_pipeline=None):
        self.llm = llm_client or LLMClient.get_shared_client()
        self.pii_regex = [
            r'\b(?:\d[ -]*?){13,16}\b',
            r'\b\d{3}-\d{2}-\d{4}\b'
        ]
        self.legal_regex = [
            r'\b(lawyer|lawsuit|sue|legal action|attorney)\b'
        ]

    def _check_escalation(self, text: str) -> tuple[bool, str | None]:
        for p in self.pii_regex:
            if re.search(p, text, re.IGNORECASE):
                return True, "PII_EXPOSURE_RISK"
        for p in self.legal_regex:
            if re.search(p, text, re.IGNORECASE):
                return True, "LEGAL_RISK_DETECTED"
        return False, None

    def process_ticket(self, customer_message: str) -> BaselineResult:
        # Regex-only escalation check
        should_escalate, reason = self._check_escalation(customer_message)
        if should_escalate:
            return BaselineResult(
                customer_message=customer_message,
                predicted_intent="Order/Tracking Status",
                escalated=True,
                escalation_reason=reason,
                final_response=f"Ticket escalated to human queue. Reason: {reason}"
            )

        # Simple zero-shot prompt without RAG grounding
        messages = [
            {"role": "system", "content": "You are @AmazonHelp official customer support. Reply to this customer query in under 280 characters without external context."},
            {"role": "user", "content": customer_message}
        ]
        try:
            response = self.llm.chat_completion(messages, max_tokens=100, temperature=0.0)
            if len(response) > 280:
                response = response[:277] + "..."
        except Exception:
            response = "We apologize for the inconvenience. Please DM us your details so we can investigate."

        # Default naive keyword intent heuristic
        lower = customer_message.lower()
        if any(w in lower for w in ["refund", "cancel", "return"]):
            intent = "Cancellation/Refund Request"
        elif any(w in lower for w in ["charge", "billed", "payment", "card", "tax"]):
            intent = "Billing/Payment Issue"
        elif any(w in lower for w in ["password", "login", "account", "locked", "otp"]):
            intent = "Account Access/Authentication"
        elif any(w in lower for w in ["down", "error", "crash", "outage", "broken", "bug"]):
            intent = "Service Outage/Technical Bug"
        elif any(w in lower for w in ["order", "tracking", "package", "delivered", "delay"]):
            intent = "Order/Tracking Status"
        else:
            intent = "General Inquiry/Feedback"

        return BaselineResult(
            customer_message=customer_message,
            predicted_intent=intent,
            escalated=False,
            escalation_reason=None,
            final_response=response
        )
