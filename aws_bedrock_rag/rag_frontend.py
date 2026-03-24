import json
import streamlit as st
import requests

API_ENDPOINT = st.secrets.get("API_GATEWAY_URL", "http://localhost:8000/query")

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="centered",
)

st.markdown(
    """
    <style>
        .main-heading  {font-family: 'Segoe UI', sans-serif; color: #2563eb; font-size: 2.4rem; font-weight: 700;}
        .sub-heading   {color: #71717a; font-size: 1.05rem; margin-top: -0.6rem;}
        .citation-box  {background: #f5f0ff; border-left: 4px solid #2563eb; padding: 0.8rem 1rem;
                        border-radius: 0.4rem; margin-bottom: 0.6rem; font-size: 0.92rem;}
        .source-label  {color: #6d28d9; font-weight: 600; font-size: 0.82rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-heading">📋 HR Policy Assistant</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-heading">Ask anything about company HR policies — powered by Amazon Bedrock Knowledge Base</p>',
    unsafe_allow_html=True,
)

st.divider()

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

user_question = st.text_area(
    "Your question",
    placeholder="e.g. What is the leave policy at IdeaWeaver AI?",
    label_visibility="collapsed",
    height=100,
)

submit_btn = st.button("Ask the HR Assistant", type="primary", use_container_width=True)

if submit_btn and user_question.strip():
    with st.spinner("Searching the knowledge base …"):
        try:
            api_result = requests.post(
                API_ENDPOINT,
                json={"question": user_question.strip()},
                timeout=120,
            )
            api_result.raise_for_status()

            payload = api_result.json()
            if isinstance(payload, str):
                payload = json.loads(payload)

            generated_answer = payload.get("answer", "No answer was returned.")
            source_references = payload.get("citations", [])

            st.session_state.chat_log.append(
                {"q": user_question.strip(), "a": generated_answer, "refs": source_references}
            )

        except requests.exceptions.ConnectionError:
            st.error("Could not reach the API. Verify that API_GATEWAY_URL is correct in your Streamlit secrets.")
        except requests.exceptions.Timeout:
            st.error("The request timed out. The knowledge base may need more time — try again shortly.")
        except Exception as err:
            st.error(f"Something went wrong: {err}")

elif submit_btn:
    st.warning("Please type a question before submitting.")

for entry in reversed(st.session_state.chat_log):
    st.markdown(f"**You:** {entry['q']}")
    st.markdown(f"{entry['a']}")

    if entry.get("refs"):
        with st.expander("View sources", expanded=False):
            for ref in entry["refs"]:
                source_uri = ref.get("source", "")
                snippet = ref.get("text", "")
                st.markdown(
                    f'<div class="citation-box">'
                    f"{snippet}"
                    f'<br/><span class="source-label">{source_uri}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
    st.divider()
