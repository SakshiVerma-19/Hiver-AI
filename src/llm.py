import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """
    Unified LLM Client.
    Automatically prioritizes Groq API (e.g. llama-3.1-8b-instant) if GROQ_API_KEY is found in .env.
    Falls back to local open-source pipeline (Qwen2.5-1.5B) if no key is configured.
    """
    _instance = None

    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.groq_key = os.getenv("GROQ_API_KEY")
        if self.groq_key and self.groq_key.strip():
            from groq import Groq
            self.client = Groq(api_key=self.groq_key.strip())
            self.use_groq = True
            self.model_name = model_name
            self.pipe = None
            print(f"⚡ [LLMClient] Connected to Groq Cloud API ({self.model_name})")
        else:
            self.use_groq = False
            self.client = None
            self.pipe = None
            print("⚙️ [LLMClient] No GROQ_API_KEY detected; using local model.")

    @classmethod
    def get_shared_client(cls, model_name: str = "llama-3.1-8b-instant"):
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    def chat_completion(self, messages: list[dict], max_tokens: int = 150, temperature: float = 0.0, json_mode: bool = False) -> str:
        if self.use_groq:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            for attempt in range(5):
                try:
                    chat_completion = self.client.chat.completions.create(**kwargs)
                    return chat_completion.choices[0].message.content.strip()
                except Exception as e:
                    err_str = str(e).lower()
                    if "429" in err_str or "rate" in err_str or "limit" in err_str:
                        wait_time = (attempt + 1) * 3
                        time.sleep(wait_time)
                    else:
                        raise e
            raise RuntimeError("Exceeded maximum retries on Groq API.")

        else:
            if self.pipe is None:
                from transformers import pipeline
                import torch
                device_map = "auto" if torch.cuda.is_available() else None
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self.pipe = pipeline(
                    "text-generation",
                    model="Qwen/Qwen2.5-1.5B-Instruct",
                    torch_dtype=torch_dtype,
                    device_map=device_map
                )

            prompt = self.pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            outputs = self.pipe(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
                return_full_text=False
            )
            return outputs[0]["generated_text"].strip()
