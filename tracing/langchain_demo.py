"""
LangChain demo: simple question-answering chain using OpenAI.
Loads OPENAI_API_KEY / OPENAI_BASE_URL from .env via python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load .env from the same directory as this script (so it works regardless of cwd)
load_dotenv(Path(__file__).resolve().parent / ".env")

# Fail fast with a clear message if the key is missing or still the placeholder
_api_key = os.getenv("OPENAI_API_KEY", "").strip().strip("'\"")
if not _api_key or "your-openai-api-key" in _api_key.lower():
    print(
        "ERROR: Set your real OpenAI API key in the .env file.\n"
        "  - Edit .env and replace OPENAI_API_KEY=your-openai-api-key-here\n"
        "  - Get a key at: https://platform.openai.com/account/api-keys"
    )
    raise SystemExit(1)
os.environ["OPENAI_API_KEY"] = _api_key  # use cleaned key (no surrounding quotes)




def main():
    # Create the model (reads OPENAI_API_KEY / OPENAI_BASE_URL from environment)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answer questions clearly."),
        ("user", "{input}")
    ])

    # Output parser
    output_parser = StrOutputParser()

    # Build the chain (prompt → model → string output)
    chain = prompt | llm | output_parser

    print("LangChain Demo - Type a question and press Enter (or empty line to exit).\n")
    while True:
        user_input = input("Enter your prompt: ").strip()
        if not user_input:
            break
        response = chain.invoke({"input": user_input})
        print(response)
        print()


if __name__ == "__main__":
    main()
