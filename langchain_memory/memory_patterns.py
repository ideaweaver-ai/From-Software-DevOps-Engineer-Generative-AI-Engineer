"""
LangChain 1.x: Memory integration patterns (working code).

1. Retrieval-based memory: store policy snippets, embed with Ollama, retrieve top-k and inject into prompt.
2. Tool-accessible memory: a tool that reads/writes "user preferences"; the model never sees the store, only tool results.

Run: python langchain_examples/memory_patterns.py
Requires Ollama (for embeddings and chat). Set OPENAI_API_KEY to use OpenAI for chat; embeddings still use Ollama for the retrieval example.
"""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

# Chat model
if os.environ.get("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI
    chat_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    from langchain_ollama import ChatOllama
    chat_llm = ChatOllama(model="llama3.2", temperature=0)

# Embeddings (Ollama) for retrieval-based memory
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# ---- 1. Retrieval-based memory ----
# In-memory "policy store": list of (embedding, text). We embed query, find top-k by dot product, inject into prompt.

POLICY_SNIPPETS = [
    "Employees must use the VPN when accessing internal tools from outside the office.",
    "Expense reports over 5000 USD require manager approval before submission.",
    "Leave requests should be submitted at least two weeks in advance for approval.",
    "Default password policy: 12 characters minimum, mix of letters, numbers, and symbols.",
]

# Precompute embeddings for the snippets (in production you would use a vector DB)
_snippet_embeddings: list[tuple[list[float], str]] = []


def _ensure_embeddings():
    global _snippet_embeddings
    if not _snippet_embeddings:
        vecs = embeddings.embed_documents(POLICY_SNIPPETS)
        _snippet_embeddings = list(zip(vecs, POLICY_SNIPPETS))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def retrieve_relevant(query: str, top_k: int = 3) -> list[str]:
    """Return top-k policy snippets most relevant to the query. This is the 'memory' the model sees."""
    _ensure_embeddings()
    q_embed = embeddings.embed_query(query)
    scored = [(_dot(q_embed, vec), text) for vec, text in _snippet_embeddings]
    scored.sort(key=lambda x: -x[0])
    return [text for _, text in scored[:top_k]]


def run_helpdesk_query(question: str) -> str:
    """Answer a policy question by retrieving relevant snippets and injecting them into the prompt."""
    relevant = retrieve_relevant(question)
    context = "\n".join(f"- {s}" for s in relevant)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an internal helpdesk bot. Answer using ONLY the following policy snippets. If the answer is not there, say so.\n\nSnippets:\n{context}"),
        ("human", "{question}"),
    ])
    chain = prompt | chat_llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# ---- 2. Tool-accessible memory ----
# The "memory" is a dict we own. The model never sees it; it calls tools to get/set preferences.

user_prefs: dict[str, str] = {}  # In production this would be a DB or key-value store


@tool
def get_user_preferences() -> str:
    """Returns the current user preferences (e.g. seat, meal, frequent flyer number). Call this when you need to remember what the user likes."""
    if not user_prefs:
        return "No preferences stored yet."
    return " | ".join(f"{k}: {v}" for k, v in user_prefs.items())


@tool
def set_user_preference(key: str, value: str) -> str:
    """Store a user preference. key: e.g. 'seat', 'meal', 'frequent_flyer'. value: the preference."""
    user_prefs[key] = value
    return f"Stored {key}={value}."


def run_travel_agent(user_message: str) -> str:
    """Simple flow: get prefs via tool, inject into prompt, call model. (No full agent loop here; just one turn with tool result in context.)"""
    prefs_text = get_user_preferences.invoke({})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a travel booking assistant. Current user preferences (from memory): {prefs}. Use them when suggesting seats or meals. If the user gives a new preference, say you've noted it (the system will store it)."),
        ("human", "{input}"),
    ])
    chain = prompt | chat_llm | StrOutputParser()
    reply = chain.invoke({"prefs": prefs_text, "input": user_message})
    # If user said something like "I prefer window seats", we could parse and call set_user_preference here (simplified: we don't auto-detect in this demo)
    return reply


if __name__ == "__main__":
    print("=== 1. Retrieval-based memory (helpdesk policy Q&A) ===\n")
    q = "What is the rule for expense reports over 5000?"
    print(f"Q: {q}")
    print(f"A: {run_helpdesk_query(q)}\n")

    print("=== 2. Tool-accessible memory (user preferences) ===\n")
    # Simulate storing a preference via the tool
    set_user_preference.invoke({"key": "seat", "value": "window"})
    set_user_preference.invoke({"key": "meal", "value": "vegetarian"})
    print("Stored preferences: seat=window, meal=vegetarian\n")
    print("User: When I book a flight, what seat and meal should I get?")
    print(f"Assistant: {run_travel_agent('When I book a flight, what seat and meal should I get?')}\n")
