from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os

openai_api_key = os.environ["OPENAI_API_KEY"]
openai_base_url = os.environ["OPENAI_BASE_URL"]

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

# Define a prompt template without memory
prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}")
])

# Create a simple chain using LangChain Expression Language (LCEL)
chain = prompt | llm

# First interaction (without memory)
response1 = chain.invoke({"input": "My name is Prashant"})
print(f"Response 1: {response1.content}")

# Second interaction - the model will not 'remember' the previous input
response2 = chain.invoke({"input": "What's my name?"})
print(f"Response 2: {response2.content}")
