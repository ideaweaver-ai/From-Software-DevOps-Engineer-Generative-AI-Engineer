from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

from dotenv import load_dotenv

load_dotenv(override=True)

os.environ["LANGCHAIN_PROJECT"] = "chatbot-demo"

# 1) Create the model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2) Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions clearly."),
    ("user", "{input}")
])

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
