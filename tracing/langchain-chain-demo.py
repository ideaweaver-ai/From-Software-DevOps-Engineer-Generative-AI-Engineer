"""Two-step chain: generate content on a subject, then summarize it in 5 points."""

from pathlib import Path
from dotenv import load_dotenv

os.environ['LANGCHAIN_PROJECT'] = 'langsmith-tracing-sequential'

load_dotenv(Path(__file__).resolve().parent / ".env")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

first_prompt = PromptTemplate(
    template="Write a short report on {subject}.",
    input_variables=["subject"],
)

second_prompt = PromptTemplate(
    template="Give a 5-point summary of this:\n{content}",
    input_variables=["content"],
)

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = first_prompt | llm | parser | second_prompt | llm | parser

result = chain.invoke({"subject": "remote work and productivity"})
print(result)
