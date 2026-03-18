import os

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)

os.environ.pop("ANTHROPIC_BASE_URL", None)
os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

os.environ["LANGCHAIN_PROJECT"] = "code-gen-review"

prompt1 = PromptTemplate(
    template=(
        "Write Python code that solves the following task: {topic}\n"
        "Return only the code (no markdown fences)."
    ),
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template=(
        "Evaluate the following Python code.\n"
        "Check: (1) correctness for the task, (2) any bugs/edge cases, "
        "(3) readability/style, and (4) how to improve it.\n\n"
        "Code:\n{text}\n\n"
        "Return your evaluation as a short checklist."
    ),
    input_variables=['text']
)

openai_kwargs = {}
if os.getenv("OPENAI_MODEL"):
    openai_kwargs["model"] = os.getenv("OPENAI_MODEL")

if "model" not in openai_kwargs:
    openai_kwargs["model"] = "gpt-5.4-mini"

openai_model = ChatOpenAI(**openai_kwargs)

anthropic_model = ChatAnthropic(
    model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
)

parser = StrOutputParser()

chain1 = prompt1 | openai_model | parser
chain2 = prompt2 | anthropic_model | parser

generated_code = chain1.invoke({'topic': 'Write a Python function that returns True if a string is a palindrome, otherwise False.'})
print("=== Generated Code (OpenAI) ===")
print(generated_code)

evaluation = chain2.invoke({'text': generated_code})
print("\n=== Evaluation (Anthropic) ===")
print(evaluation)
