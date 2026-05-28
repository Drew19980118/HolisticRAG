SEED_QUERY_PROMPT_TEMPLATE = """
You are an expert at finding seed queries in complex questions. Seed queries are simple, atomic questions that can be directly answered from a knowledge base. They represent the first step in each reasoning path needed to answer the original question.

Your task: For each given complex query, identify all distinct reasoning paths required to answer it. For each path, output only the very first question that must be answered. Do not output any follow-up questions or the original query itself.

Output format:
- Start with "Seed queries:"
- Then list each seed query separated by semicolons (`;`).
- Each seed query must begin with a question word (Who, What, When, Where, Which, How, How many).
- The seed queries should be meaningful, include background knowledge and context that help understand the entities, and not be overly long.

Examples:

Example 1:
Query: "What disease did Marie Curie's husband's mother die of?"
Reasoning: The first step is to identify Marie Curie's husband. Then we could ask about his mother, but that is not the first step.
Seed queries: Who was Marie Curie's husband?

Example 2:
Query: "Were Messi and Maradona born in the same city?"
Reasoning: Two independent paths: one for Messi's birthplace, one for Maradona's birthplace. Both first steps are parallel.
Seed queries: What city was Messi born in?; What city was Maradona born in?

Now, for the following query, provide the seed queries in the specified format. Do not include any additional text.

Query: {query}
"""