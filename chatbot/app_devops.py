from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# 1) Title and description for the app UI
st.title("DevOps Assistant (LangChain + Streamlit)")
st.write("Ask DevOps / SRE related questions (incident, deployment, logs, etc.)")

# 2) Create the model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3) DevOps System Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a DevOps / SRE assistant.

Rules:
- Do not guess root cause without evidence.
- Provide structured operational answers.

Response format:
1. Summary
2. Possible Causes
3. What to Check Next
4. Suggested Mitigation (low risk first)
5. Slack Update Draft
"""),
    ("user", "Question:\n{input}")
])

# 5) Output parser
output_parser = StrOutputParser()

# 6) Build chain
chain = prompt | llm | output_parser

# 7) User Question
user_input = st.text_input("Enter your DevOps question")

# 8) Run the chain
if user_input:
    response = chain.invoke({
        "input": user_input
    })
    st.write(response)
