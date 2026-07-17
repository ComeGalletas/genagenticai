from langchain_core.prompts import ChatPromptTemplate

judge_prompt = ChatPromptTemplate.from_template("""
You are a strict research quality evaluator.

User Query: {query}

Search Results:
{results}

Evaluate if the search results are sufficient to answer the query well.

Criteria for "ACCEPTABLE":
- Contains relevant, up-to-date, and factual information
- Covers the main aspects of the query
- Has good sources (not just generic or spammy pages)
- Enough depth (not just titles/snippets)

Respond in JSON only:
{{
  "decision": "ACCEPT" or "REJECT",
  "reason": "brief explanation",
  "confidence": 0-100,
  "suggestion": "If REJECT, suggest a better query or what is missing"
}}
""")