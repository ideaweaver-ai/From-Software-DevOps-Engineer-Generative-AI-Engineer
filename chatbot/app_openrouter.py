"""
Streamlit app using the cloud model gpt-oss-20b (free) via OpenRouter and LangChain.

Model: openai/gpt-oss-20b:free

Setup:
  1. Sign up at https://openrouter.ai/ and get an API key.
  2. Set: export OPENROUTER_API_KEY=your_key
  3. Enable free models in data policy: https://openrouter.ai/settings/privacy
  4. Run: streamlit run app_openrouter.py

If you get 429 (rate limit): wait and retry, or add your own provider key at
  https://openrouter.ai/settings/integrations (BYOK) for higher limits.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
api_key = os.environ.get("OPENROUTER_API_KEY")

st.title("LangChain + OpenRouter: gpt-oss-20b")
st.write(
    "Chat using the cloud model **gpt-oss-20b (free)** via OpenRouter. "
    "Set `OPENROUTER_API_KEY`; allow free models in [Privacy](https://openrouter.ai/settings/privacy) if you get 404. "
    "On 429 (rate limit), retry later or add a provider key in [Integrations](https://openrouter.ai/settings/integrations)."
)

if not api_key:
    st.error(
        "Missing **OPENROUTER_API_KEY**. Get a key at [OpenRouter](https://openrouter.ai/), "
        "then run: `export OPENROUTER_API_KEY=your_key`"
    )
    st.stop()

# LangChain ChatOpenAI pointed at OpenRouter (OpenAI-compatible API)
llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    temperature=0,
    openai_api_key=api_key,
    openai_api_base=OPENROUTER_BASE,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions clearly."),
    ("user", "{input}"),
])
chain = prompt | llm | StrOutputParser()

user_input = st.text_input("Enter your prompt")

if user_input:
    with st.spinner("Thinking..."):
        try:
            response = chain.invoke({"input": user_input})
            st.write(response)
        except Exception as e:
            err = str(e)
            st.error(f"Request failed: {err}")
            if "429" in err or "rate" in err.lower():
                st.info(
                    "**Rate limited.** Wait a minute and try again, or add your own provider key (BYOK) at "
                    "[OpenRouter Integrations](https://openrouter.ai/settings/integrations) for higher limits."
                )
