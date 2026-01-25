from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from guardrails import Guard
from guardrails.hub import DetectJailbreak

# Title and description
st.title("LangChain + Streamlit Demo with Guardrails")
st.write("Type a question below and get a safe answer from the model.")

# Create the model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions clearly."),
    ("user", "{input}")
])

# Output parser
output_parser = StrOutputParser()

# Build the chain
chain = prompt | llm | output_parser

# Create input guard - checks user input before sending to LLM
input_guard = Guard().use(DetectJailbreak(threshold=0.9))

# Streamlit input
user_input = st.text_input("Enter your prompt")

# Run the chain with guardrails
if user_input:
    try:
        # Validate input first
        input_guard.validate(user_input)
        
        # Get LLM response
        response = chain.invoke({"input": user_input})
        st.write(response)
        
    except Exception as e:
        st.error(f"Input validation failed: {str(e)}")
        st.warning("Please rephrase your question.")
