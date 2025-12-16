import os
from openai import OpenAI
from workspace.utils.logging_utils import get_logger

logger = get_logger("models.openai_client")

class OpenAIChat:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
