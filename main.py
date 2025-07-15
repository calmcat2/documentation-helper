import streamlit as st
import re
from streamlit_chat import message
from backend.core import docsearch
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

st.header("Langchain Document search")
st.write(
    "This app allows you to search documents using Langchain and Google Generative AI."
)

prompt = st.text_input("Enter your query here")

if (
    "user-query-history" not in st.session_state
    and "bot-response-history" not in st.session_state
    and "chat-history" not in st.session_state
):
    st.session_state["user-query-history"] = []
    st.session_state["bot-response-history"] = []
    st.session_state["chat-history"] = []


def format_sources(context):
    if not context:
        return "No sources found."
    sources = []
    for doc in context:
        url = doc.metadata["source"]
        formatted_url = re.sub(r"\\", r'/', url)
        final_url = re.sub(r"https:///", r'https://', formatted_url)
        sources.append(final_url)
    source_set = set(sources)
    return "\n".join(f"- {src}" for src in source_set)


if prompt:
    with st.spinner("Searching..."):
        answer, context = docsearch(prompt, st.session_state["chat-history"])
        formatted_answer = (
            answer + "\n\nSources:\n" + format_sources(context)
        )
        st.session_state["user-query-history"].append(prompt)
        st.session_state["bot-response-history"].append(formatted_answer)
        st.session_state["chat-history"].append(HumanMessage(content=prompt))
        st.session_state["chat-history"].append(
            AIMessage(content=formatted_answer)
        )

if st.session_state["bot-response-history"]:
    for i, (user_query, bot_response) in enumerate(zip(
        st.session_state["user-query-history"],
        st.session_state["bot-response-history"]
    )):
        message(user_query, is_user=True, key=f"user_message_{i}")
        message(bot_response, is_user=False, key=f"bot_message_{i}")
