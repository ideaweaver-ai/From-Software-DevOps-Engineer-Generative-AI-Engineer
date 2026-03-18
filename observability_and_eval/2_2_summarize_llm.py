import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv(override=True)

os.environ["LANGCHAIN_PROJECT"] = "sequential-summariser"

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
)

writer_prompt = PromptTemplate(
    template=(
        "You are a research analyst. Write a comprehensive analysis "
        "covering the key causes, current trends, and potential solutions "
        "for the following subject:\n\n{subject}"
    ),
    input_variables=["subject"],
)

summariser_prompt = PromptTemplate(
    template=(
        "Read the analysis below and distil it into exactly 5 concise bullet points. "
        "Each bullet should capture one distinct insight.\n\n"
        "Analysis:\n{text}"
    ),
    input_variables=["text"],
)

output_parser = StrOutputParser()

pipeline = writer_prompt | llm | output_parser | summariser_prompt | llm | output_parser

subject = "The impact of AI on the global job market"

response = pipeline.invoke({"subject": subject})

print("--- Key Takeaways ---")
print(response)
