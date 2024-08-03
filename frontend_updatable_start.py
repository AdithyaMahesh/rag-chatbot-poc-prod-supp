import streamlit as st
import requests
import json

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# Backend URLs
backend_url = "http://localhost:8000/chat"
update_url = "http://localhost:8000/update_index"

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.loading = False

# Function to update knowledge base
def update_knowledge_base():
    st.session_state.loading = True
    response = requests.post(update_url)
    if response.status_code == 200:
        st.success("Knowledge base updated successfully!")
    else:
        st.error("Failed to update knowledge base.")
    st.session_state.loading = False
    st.session_state.initialized = True

# Display initial choice buttons
if not st.session_state.initialized:
    st.title("Production Support RAG Chatbot💬")
    st.header("Ask your queries related to production support.")

    st.subheader("Choose an option:")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Update Knowledge Base"):
            update_knowledge_base()

    with col2:
        if st.button("Continue with Existing Knowledge Base"):
            st.session_state.initialized = True

# Chat interface for user interaction
if st.session_state.initialized:
    if not st.session_state.messages:
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
