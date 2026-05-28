QUESTION_SYSTEM = "You are an expert in generating natural, context-aware questions from given passages and extracted facts."

QUESTION_USER_TEMPLATE = """
Task: Generate two natural and contextual questions based on the provided passage and fact, such that the answer to the first question is the subject of the fact, and the answer to the second question is the object of the fact.

Rules for QA Generation:
(1) Each question must start with a question word (Who, What, When, Where, Which, How, How many).
(2) The questions should be meaningful, include background knowledge and context that help understand the entities, and not be overly long.
(3) The output format must exactly match the example below, with no extra text.

Example:
Passage:
"The Godfather is a 1972 American crime film directed by Francis Ford Coppola, produced by Albert S. Ruddy, and based on Mario Puzo's bestselling novel of the same name. It stars Marlon Brando and Al Pacino as the leaders of a powerful New York crime family."

Triple: (The Godfather, directed by, Francis Ford Coppola)

Output:
Q1: Which film from 1972, based on Mario Puzo's novel and starring Marlon Brando, was directed by Francis Ford Coppola?
Q2: Who was the director of the 1972 crime film The Godfather, which was produced by Albert S. Ruddy?

Now please generate questions for the following passage and triple. Output only the two questions prefixed with Q1: and Q2: respectively.

Passage:
\"\"\"
{passage}
\"\"\"

Triple: ({subj}, {pred}, {obj})
"""