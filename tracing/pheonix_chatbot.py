import os
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load API keys and config from .env (OPENAI_API_KEY, etc.)
# Requires `python-dotenv` (provides the `dotenv` module).
from dotenv import load_dotenv

load_dotenv(override=True)

# Make Phoenix use a writable working directory (avoids ~/.phoenix permission issues).
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("PHOENIX_WORKING_DIR", str(BASE_DIR / ".phoenix"))

# Phoenix tracing setup (exports OpenInference spans to self-hosted Phoenix).
from phoenix.otel import register

register(
    project_name=os.getenv("PHOENIX_PROJECT_NAME", "phoenix-chatbot-demo"),
    auto_instrument=True,
    # Phoenix OTLP HTTP receiver endpoint:
    endpoint=os.getenv("PHOENIX_HTTP_TRACES_ENDPOINT", "http://localhost:6006/v1/traces"),
    protocol="http/protobuf",
    verbose=False,
)

# 1) Create the model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2) Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can answer questions clearly."),
        ("user", "{input}"),
    ]
)

# 3) Output parser
output_parser = StrOutputParser()

# 4) Build the chain
chain = prompt | llm | output_parser

# 5) Hardcoded question
question = "What is the capital of India?"

# 6) Run the chain
response = chain.invoke({"input": question})

# 7) Print response
print("Question:", question)
print("Response:", response)
