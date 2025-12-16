ANSWER_PROMPT = """You are a financial QA assistant.
You MUST answer using ONLY the Evidence text provided.

Evidence:
{context}

Question:
{question}

Instructions:
- If the evidence does not contain enough information to answer, output:
  ANSWER: CANNOT_ANSWER
  CONFIDENCE: 0.0
- If you can answer, compute the requested quantity carefully.
- Output EXACTLY two lines:
  1) ANSWER: <final answer>
  2) CONFIDENCE: <p>  (0 to 1)

No other text.
"""

REWRITE_PROMPT = """You rewrite financial questions to improve document retrieval.

Original question:
{question}

Retrieved evidence snippets (may be incomplete or noisy):
{snippets}

Rules:
- STRICTLY PRESERVE company names, years/dates, numbers, units, and the financial metric asked.
- You may remove fluff (role-play, instructions).
- You may add explicit keywords for the metric (e.g., working capital ratio = current assets / current liabilities).
- Keep it short and retrieval-friendly.
- Do NOT add any new factual constraints (years/dates, numbers, amounts, segment names) that are not explicitly in the original question.
- If the original question does not specify a year, do NOT guess a year.
- Output only the rewritten query, nothing else.

Return ONLY the rewritten question.
"""
