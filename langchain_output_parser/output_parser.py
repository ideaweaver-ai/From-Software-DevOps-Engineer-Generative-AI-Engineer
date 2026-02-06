"""
LangChain 1.x: Output parser examples.

Demonstrates StrOutputParser, JsonOutputParser, PydanticOutputParser, and a custom regex parser.

Run: python langchain_examples/output_parsers_example.py
Uses Ollama by default; set OPENAI_API_KEY to use OpenAI instead.
"""

import os
import re
from typing import Optional

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
# Pydantic for PydanticOutputParser
try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = None
    Field = None

if os.environ.get("OPENAI_API_KEY"):
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.2", temperature=0)


# ----- 1. StrOutputParser -----
# Returns the model output as a single string. No structure.

def example_str_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer in one short paragraph."),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"input": "What is the capital of France?"})
    assert isinstance(result, str)
    print("1. StrOutputParser (returns str):")
    print(f"   {result[:80]}...\n")
    return result


# ----- 2. JsonOutputParser -----
# Ask the model for JSON; parser returns a dict (or list).

def example_json_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You answer with valid JSON only, no other text. Use keys: 'answer', 'confidence' (0-100)."),
        ("human", "{input}"),
    ])
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"input": "Is Paris the capital of France? Reply with answer (yes/no) and confidence."})
    assert isinstance(result, dict)
    print("2. JsonOutputParser (returns dict):")
    print(f"   {result}\n")
    return result


# ----- 3. PydanticOutputParser -----
# Define a Pydantic model; parser injects format instructions and returns an instance.

def example_pydantic_parser():
    if BaseModel is None:
        print("3. PydanticOutputParser: skipped (install pydantic)\n")
        return None

    from langchain_core.output_parsers import PydanticOutputParser

    class QuizQuestion(BaseModel):
        question: str = Field(description="A quiz question")
        options: list[str] = Field(description="Four possible answers A-D")
        correct: str = Field(description="The correct option letter, e.g. A")

    parser = PydanticOutputParser(pydantic_object=QuizQuestion)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate one quiz question. Output format:\n{format_instructions}"),
        ("human", "Topic: {topic}"),
    ])
    chain = prompt | llm | parser
    result = chain.invoke({
        "topic": "geography",
        "format_instructions": format_instructions,
    })
    assert isinstance(result, QuizQuestion)
    print("3. PydanticOutputParser (returns Pydantic model):")
    print(f"   question={result.question[:50]}...")
    print(f"   options={result.options}")
    print(f"   correct={result.correct}\n")
    return result


# ----- 4. Custom / regex parser -----
# Extract specific parts of the model output (e.g. a code block or an ID).

class CodeBlockParser:
    """Custom parser: extract the first markdown code block from the model output."""

    def invoke(self, message) -> dict[str, Optional[str]]:
        if hasattr(message, "content"):
            text = message.content
        else:
            text = str(message)
        match = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return {"code": match.group(1).strip(), "language": "unknown"}
        return {"code": None, "language": None}

    def parse(self, text: str) -> dict[str, Optional[str]]:
        return self.invoke(text)


def example_custom_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Reply with a single Python code block (markdown). Only the block, no explanation."),
        ("human", "Write a one line function that doubles a number."),
    ])
    chain_no_parse = prompt | llm
    raw = chain_no_parse.invoke({})
    parser = CodeBlockParser()
    result = parser.invoke(raw)
    print("4. Custom parser (regex: extract code block):")
    print(f"   {result}\n")
    return result


if __name__ == "__main__":
    print("Output parser examples (LangChain 1.x)\n")
    example_str_parser()
    example_json_parser()
    example_pydantic_parser()
    example_custom_parser()
