RELATION_SYSTEM = """
# Role
You are an expert in extracting relations from a given paragraph.

# Tasks
Extract all possible underlying relations from the given [Passage]. This includes both explicitly stated relations and those implicit relations that can be reasonably inferred from the [Passage].
Look for any meaningful connection between any two entities in [Entity List].

# Constraints
- No duplicate relations in your output.
- Ensure all extracted relations are concise.
- Follow exactly the format shown in the example below.
"""

RELATION_EXAMPLE_PASSAGE = "Barack Obama was born in Hawaii. He served as the 44th President of the United States. Obama is a Democrat. He won the Nobel Peace Prize in 2009."

RELATION_EXAMPLE_OUTPUT = """{"relations": ["was born in", "served as", "is a", "won"]}"""

RELATION_USER_TEMPLATE = """Passage: {passage}
Entity List: {entities}
"""