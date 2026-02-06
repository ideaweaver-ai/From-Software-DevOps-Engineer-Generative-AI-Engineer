"""
LangChain 1.x: Short-term memory as explicit state.

Memory = the list of messages you keep and pass into the model each time.
No framework "memory" object; you cap the list (e.g. last 10 messages) and pass it in.

Run: python langchain_examples/memory_short_term.py
Uses Ollama by default (ollama run llama3.2); set OPENAI_API_KEY to use OpenAI instead.
"""

import os
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Prefer Ollama if running locally; otherwise OpenAI
if os.environ.get("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.2", temperature=0)

# System message; then a placeholder for the conversation history we will pass explicitly
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful support assistant. Answer briefly. Use the conversation history if the user refers to something earlier."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# Short-term memory: we keep the last N messages in a list (e.g. 10 turns = 20 messages)
MAX_MESSAGES = 20

def run_chat(messages: list[BaseMessage], user_text: str) -> tuple[str, list[BaseMessage]]:
    """Append user message, invoke chain with full history, return reply and updated messages."""
    messages.append(HumanMessage(content=user_text))
    # Trim to last MAX_MESSAGES so context does not grow forever
    if len(messages) > MAX_MESSAGES:
        messages = messages[-MAX_MESSAGES:]
    reply = chain.invoke({"history": messages, "input": user_text})
    messages.append(AIMessage(content=reply))
    return reply, messages


if __name__ == "__main__":
    # In a real app this would live in your server/graph state or request context
    messages: list[BaseMessage] = []

    print("Short-term memory demo. You pass the last N messages explicitly each time.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            break
        reply, messages = run_chat(messages, user_input)
        print(f"Assistant: {reply}\n")
