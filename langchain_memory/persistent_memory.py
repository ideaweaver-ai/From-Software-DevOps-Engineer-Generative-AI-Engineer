"""
LangChain 1.x: Persistent conversations.

You own storage. Load history by session_id, append new turn, call model, save back.
LangChain stays stateless; this script uses a JSON file as the store.

Run: python langchain_examples/memory_persistent.py
Uses Ollama by default; set OPENAI_API_KEY to use OpenAI instead.
"""

import json
import os
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, message_to_dict, messages_from_dict

if os.environ.get("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.2", temperature=0)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a billing support agent. Use the conversation history when the user refers to earlier messages."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()

# Storage: JSON file (in production you would use a database)
STORE_PATH = Path(__file__).parent / "persistent_store.json"
MAX_MESSAGES = 50


def load_history(session_id: str) -> list[BaseMessage]:
    """Load conversation history for this session from our store."""
    if not STORE_PATH.exists():
        return []
    data = json.loads(STORE_PATH.read_text())
    raw = data.get(session_id, [])
    if not raw:
        return []
    return list(messages_from_dict(raw))


def save_history(session_id: str, messages: list[BaseMessage]) -> None:
    """Save conversation history for this session."""
    if len(messages) > MAX_MESSAGES:
        messages = messages[-MAX_MESSAGES:]
    raw = [message_to_dict(m) for m in messages]
    data = {}
    if STORE_PATH.exists():
        data = json.loads(STORE_PATH.read_text())
    data[session_id] = raw
    STORE_PATH.write_text(json.dumps(data, indent=2))


def run_turn(session_id: str, user_text: str) -> str:
    """Load history, add user message, call model, append both to history, save. Return assistant reply."""
    messages = load_history(session_id)
    messages.append(HumanMessage(content=user_text))
    reply = chain.invoke({"history": messages, "input": user_text})
    messages.append(AIMessage(content=reply))
    save_history(session_id, messages)
    return reply


if __name__ == "__main__":
    session_id = "user_123_thread_billing"
    print("Persistent conversation demo. History is stored in persistent_store.json by session_id.\n")
    print("Same session_id across runs will see previous messages. Change session_id to start fresh.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            break
        reply = run_turn(session_id, user_input)
        print(f"Assistant: {reply}\n")
