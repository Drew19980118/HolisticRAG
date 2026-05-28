GENERATION_PROMPT_TEMPLATE = """You are a QA assistant. Based on the user question and known information below, generate a new question that needs to be further queried.

User Question: {original_query}

{context}
IMPORTANT:
- The new question must be atomic: it should ask for a single piece of information or a simple relationship, not a compound or multi-part question.
- The new question must start with a question word (Who, What, When, Where, Which, How, How many).
- The new question should not be overly long.
- Output format: Query: <new question>

Do not output any extra explanation, only the above format.
"""