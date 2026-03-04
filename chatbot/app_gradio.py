from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

# 1) Title and description for the app UI
title = "DevOps Assistant (LangChain + Streamlit)"
description = "Ask DevOps / SRE related questions (incident, deployment, logs, etc.)"

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

def run_chain(user_input):
    if user_input:
        response = chain.invoke({
            "input": user_input
        })
        return response
    return ""

with gr.Blocks() as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)

    # 7) User Question
    user_input = gr.Textbox(label="Enter your DevOps question")

    # 8) Run the chain
    output = gr.Markdown()
    user_input.change(fn=run_chain, inputs=user_input, outputs=output)

demo.launch()
