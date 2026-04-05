TOOL_SCHEMA = """# Tools

You may call one tool at a time to assist with the question.

Available tools:
<tools>
{"type": "function", "function": {"name": "search", "description": "Search the index and return top results with URLs.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "A focused search query."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Open a URL from the search results when snippets are insufficient, ambiguous, or you need an exact fact from the page.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The exact URL to open from prior search results."}}, "required": ["url"]}}}
</tools>

Preferred tool-call format:
<tool_call>
{"name": "search", "arguments": {"query": "your query"}}
</tool_call>

or

<tool_call>
{"name": "visit", "arguments": {"url": "https://example.com"}}
</tool_call>

Legacy formats are also accepted:
<search> your query </search>
<access> https://example.com </access>
"""

SEARCH_ACCESS_PROMPT_TEMPLATE=(
    "A conversation between User and Assistant. The user asks a question, and the Assistant answers it. "
    "The Assistant should not rely on memory alone. For answerable questions, the Assistant should normally search first, "
    "inspect the returned evidence carefully, and visit a webpage when the snippets are insufficient, ambiguous, or when an exact fact is needed. "
    + TOOL_SCHEMA
    + "\nThe top search results and webpage contents will be returned between <information> and </information>. "
    "The reasoning process is enclosed within <think> </think>. When you have enough evidence, provide the answer inside <answer> and </answer>. "
    "Do not answer directly from unsupported prior knowledge. If search results are noisy, search again or visit a promising result instead of guessing. "
    "If the model generated a URL as a search query, that usually means it should use visit/access instead. "
    "Use '<answer> the question is invalid. </answer>' only when the question is truly inconsistent or cannot be satisfied after trying to verify it with the available tools.\n\n"
    "User: \n\n{question}. \n\nThe language of your answer should align with the question. \n\nAssistant: \n<think>\n"
)
SEARCH_ONLY_PROMPT_TEMPLATE=(
    "A conversation between User and Assistant. The user asks a question, and the Assistant answers it. "
    "The Assistant should not rely on memory alone. For answerable questions, the Assistant should normally search first and ground the answer in the returned evidence. "
    + TOOL_SCHEMA
    + "\nOnly use search in this mode; do not visit webpages. "
    "The top search results will be returned between <information> and </information>. "
    "The reasoning process is enclosed within <think> </think>. Finally, provide the answer inside <answer> and </answer>. "
    "Do not answer directly from unsupported prior knowledge when evidence has not been checked. "
    "Use '<answer> the question is invalid. </answer>' only when the question is truly inconsistent or cannot be satisfied after trying to verify it with the available tools.\n\n"
    "User: \n\n{question}. \n\nThe language of your answer should align with the question. \n\nAssistant: \n<think>\n"
)
INVALID_PROMPT="Use '<answer> the question is invalid. </answer>' only after attempting to verify the question with the available tools and finding that the question is genuinely inconsistent or unsatisfiable."
VALID_PROMPT="You should search before answering, avoid unsupported guesses, and visit/access a returned URL when snippets are insufficient for an exact answer. "
