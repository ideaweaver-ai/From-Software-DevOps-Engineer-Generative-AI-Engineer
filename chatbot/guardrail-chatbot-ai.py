"""
Streamlit + LangChain + Guardrails — input and output validation.

Guards:
  - Input:  DetectJailbreak (user prompt checked before sending to LLM)
  - Output: DetectJailbreak (model response checked before showing to user)

Install: pip install langchain-openai streamlit guardrails-ai
         guardrails hub install hub://guardrails/detect_jailbreak

Run: streamlit run app_guardrails_review.py
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from guardrails import Guard
from guardrails.hub import DetectJailbreak

# -----------------------------------------------------------------------------
# Config & API key
# -----------------------------------------------------------------------------
if not os.environ.get("OPENAI_API_KEY"):
    st.error("Set **OPENAI_API_KEY** in your environment to use gpt-4o-mini.")
    st.stop()

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("LangChain + Streamlit Demo with Guardrails")
st.write("Type a question below and get a safe answer. Input and output are both checked for safety.")

# -----------------------------------------------------------------------------
# Model and chain (could wrap in st.cache_resource for fewer rebuilds)
# -----------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions clearly."),
    ("user", "{input}"),
])
chain = prompt | llm | StrOutputParser()

# Input guard: jailbreak detection before sending to LLM
input_guard = Guard().use(DetectJailbreak(threshold=0.9))
# Output guard: validate model response before displaying
output_guard = Guard().use(DetectJailbreak(threshold=0.9))

# -----------------------------------------------------------------------------
# User input and run
# -----------------------------------------------------------------------------
user_input = st.text_input("Enter your prompt")

if user_input:
    try:
        # 1) Validate input (raises if jailbreak detected)
        input_guard.validate(user_input)
    except Exception as e:
        err_msg = str(e).lower()
        if "jailbreak" in err_msg or "validation" in err_msg or "guard" in err_msg:
            st.error("Input was flagged as unsafe. Please rephrase your question.")
        else:
            st.error(f"Validation error: {e}")
        st.stop()

    try:
        with st.spinner("Thinking..."):
            response = chain.invoke({"input": user_input})
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.caption("This may be an API or network error. Check OPENAI_API_KEY and connectivity.")
        st.stop()

    # Validate output before showing to user
    try:
        output_guard.validate(response)
        st.write(response)
    except Exception as e:
        err_msg = str(e).lower()
        if "jailbreak" in err_msg or "validation" in err_msg or "guard" in err_msg:
            st.warning("Response was filtered: the model output did not pass safety checks and was not displayed.")
        else:
            st.error(f"Output validation error: {e}")
