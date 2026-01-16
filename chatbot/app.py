from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# 1) Title and description for the app UI
st.title("LangChain + Streamlit Demo")
st.write("Type a question below and get an answer from the model.")

# 2) Create the model (reads OPENAI_API_KEY / OPENAI_BASE_URL from environment)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3) Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions clearly."),
    ("user", "{input}")
])

# 4) Output parser
output_parser = StrOutputParser()

# 5) Build the chain (prompt → model → string output)
chain = prompt | llm | output_parser

# 6) Streamlit input
user_input = st.text_input("Enter your prompt")

# 7) Run the chain when input is provided
if user_input:
    response = chain.invoke({"input": user_input})
    st.write(response)
