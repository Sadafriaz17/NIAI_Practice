import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

st.title("SynapseX")

model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
st.caption(f"Model: {model_name} | .env found: {env_path.exists()}")

llm = ChatGroq(model=model_name)

# Keep the chat history between reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant. Keep answers short and clear.")
    ]

# Show past messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Handle new input
user_input = st.chat_input("Type your message")
if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        try:
            response = llm.invoke(st.session_state.messages)
            st.write(response.content)
            st.session_state.messages.append(AIMessage(content=response.content))
        except Exception as e:
            st.error(f"Something went wrong: {e}")