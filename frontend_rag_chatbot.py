import streamlit as st
import requests
import json

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# Backend URL
backend_url = "http://localhost:8000/chat"

st.title("Production Support RAG Chatbot💬")
st.header("Ask your queries related to production support.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I assist you with production support today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.write(content)

if prompt := st.chat_input("Your question:"):
    add_message("user", prompt)

    try:
        response = requests.post(backend_url, json={"prompt": prompt}, stream=True)

        assistant_message = ""
        message_placeholder = st.empty()
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').strip()
                if decoded_line == "data: [DONE]":
                    break
                if decoded_line.startswith("data: "):
                    message_data = json.loads(decoded_line[6:])  # Strip 'data: ' prefix
                    token = message_data["response"]
                    assistant_message += token
                    message_placeholder.markdown(assistant_message)

        # Update final message content
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    except requests.exceptions.ChunkedEncodingError:
        st.error("Error: Response ended prematurely. Please try again.")
