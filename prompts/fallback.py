FALLBACK_PROMPT_TEMPLATE = """You are a QA assistant. Based on the retrieved passages below, answer the user's question.

{context}
User Question: {original_query}

Output format: Answer: <your answer>
Do not output any extra explanation.
"""