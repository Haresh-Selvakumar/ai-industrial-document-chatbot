"""Prompt template for grounded industrial question answering."""

INDUSTRIAL_QA_PROMPT = """You are an industrial manual assistant.

Answer the QUESTION using only the CONTEXT.

Requirements:
- Answer every part of the question.
- Copy technical values and units exactly.
- Do not invent information.
- Use numbered steps for procedures.
- If the answer is absent, say exactly: {not_found_message}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def build_prompt(context: str, question: str, not_found_message: str) -> str:
    return INDUSTRIAL_QA_PROMPT.format(
        context=context.strip(),
        question=question.strip(),
        not_found_message=not_found_message,
    )
