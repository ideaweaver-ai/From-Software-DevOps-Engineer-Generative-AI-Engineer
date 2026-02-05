from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# 1) Title and description for the app UI
st.title("LangChain + Ollama Demo")
st.write("Type a question below and get an answer from the local model.")

# 2) Create the model using Ollama (connects to localhost:11434 by default)
llm = ChatOllama(model="gpt-oss:20b", temperature=0)

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
    with st.spinner("Thinking..."):
        response = chain.invoke({"input": user_input})
    st.write(response)
