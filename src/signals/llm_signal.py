from transformers import pipeline

classifier = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True
)

def get_llm_severity(text):

    prompt = f"""
Classify severity.

Only answer:

Low
Medium
High
Critical

Ticket:
{text}
"""

    result = classifier(
        prompt,
        max_new_tokens=10
    )[0]["generated_text"]

    result = result.lower()

    if "critical" in result:
        return 3

    if "high" in result:
        return 2

    if "medium" in result:
        return 1

    return 0