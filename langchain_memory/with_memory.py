import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

openai_api_key = os.environ["OPENAI_API_KEY"]
openai_base_url = os.environ["OPENAI_BASE_URL"]

# Create the LLM
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

# Build a prompt that includes prior conversation ("history") + new user input ("input")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer briefly and clearly."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create a chain: prompt -> model
chain = prompt | llm

# In-memory store for chat history per session_id
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap chain with message history
conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Run the conversation (same session_id => memory works)
session_id = "user1"

response1 = conversation.invoke(
    {"input": "My name is Prashant"},
    config={"configurable": {"session_id": session_id}},
)
print("Response 1:", response1.content)

response2 = conversation.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": session_id}},
)
print("Response 2:", response2.content)
