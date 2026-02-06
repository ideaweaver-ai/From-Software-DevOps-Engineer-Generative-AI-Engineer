# pip install -U langchain langchain-openai
# export OPENAI_API_KEY="..."

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from langchain.agents.middleware import (
    before_model,
    wrap_model_call,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime


# ----------------------------
# 1) Define a tool (something the agent can call)
# ----------------------------
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


# ----------------------------
# 2) Define middleware
# ----------------------------
@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    # This runs before EVERY model call inside the agent loop
    print(f"[before_model] sending {len(state['messages'])} messages to the model")
    return None


basic_model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
advanced_model = ChatOpenAI(model="gpt-4.1", temperature=0)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    # This runs right around the model call.
    # We can override the request (model/tools/messages/system prompt, etc.)
    message_count = len(request.state["messages"])

    model = advanced_model if message_count > 10 else basic_model
    return handler(request.override(model=model))


# ----------------------------
# 3) Create agent with middleware
# ----------------------------
agent = create_agent(
    model=basic_model,                 # default model
    tools=[add],
    system_prompt="You are a helpful assistant.",
    middleware=[log_before_model, dynamic_model_selection],
)


# ----------------------------
# 4) Run the agent
# ----------------------------
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is 12 + 30? Use the add tool."}
    ]
})

print("\nFinal output:")
print(result)
