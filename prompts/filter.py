FILTER_PROMPT_TEMPLATE = """You are a relevance filter. Given a seed query and a list of candidate questions (each with a number), select the numbers of those questions that are highly relevant to the seed query.

Seed Query: {seed_query}

Candidate Questions:
{numbered_list}

Instructions:
- Highly relevant means: the candidate question asks for information that directly helps answer the seed query, or asks for a closely related fact/entity.
- If a candidate question is only loosely related or irrelevant, do NOT include its number.
- You MUST output a JSON list of integers, e.g., [1, 3, 5].
- If none are relevant, output an empty list [].
- Do NOT output any extra explanation, only the JSON list.

Output:"""