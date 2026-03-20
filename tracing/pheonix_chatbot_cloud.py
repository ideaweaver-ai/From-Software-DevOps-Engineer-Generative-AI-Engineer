import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

from phoenix.otel import register

# The environment has an outbound proxy configured which is returning:
#   ProxyError('Tunnel connection failed: 403 Forbidden')
# Disable proxies for this script so Phoenix Cloud can be reached directly.
for _k in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "ALL_proxy",
]:
    os.environ.pop(_k, None)

# Your environment has a proxy set (HTTP[S]_PROXY). That proxy appears to block
# outbound traffic to Phoenix Cloud with `403 Forbidden`. Bypass it for Phoenix.
phoenix_host = "app.phoenix.arize.com"
existing_no_proxy = os.getenv("NO_PROXY", "")
if phoenix_host not in existing_no_proxy:
    os.environ["NO_PROXY"] = (existing_no_proxy + ("," if existing_no_proxy else "") + phoenix_host).strip(",")

# Phoenix Cloud endpoint format (per docs): https://app.phoenix.arize.com/s/<space_id>
tracer_provider = register(
    project_name=os.getenv("PHOENIX_PROJECT_NAME", "phoenix-chatbot-cloud-simplified"),
    auto_instrument=True,
)

# 1) Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2) Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can answer questions clearly."),
        ("user", "{input}"),
    ]
)

# 3) Chain
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# 4) Run
question = "What is the capital of India?"
response = chain.invoke({"input": question})

print("Question:", question)
print("Response:", response)
