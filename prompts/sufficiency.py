SUFFICIENCY_PROMPT_TEMPLATE = """You are a QA assistant. Based on the user question and known information below, determine whether you can answer the user's question.

User Question: {original_query}

{context}
Based on the known information to answer the question, follow the output format strictly:
- If the known information is sufficient to answer the question, output: Yes. Answer: <your answer>
- If the known information is insufficient to answer the question, output format: No.

Do not output any extra explanation, only the above format.
"""