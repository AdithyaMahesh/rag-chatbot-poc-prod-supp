import streamlit as st
import requests

# Set Wide for page title and layout
st.set_page_config(page_title="Production Support RAG Chatbot", layout="wide")

st.title("Production Support RAG Chatbot")
st.header("Ask your queries related to product support.")

# Centered heading Chat
st.markdown(
    """
    <h2 style='text-align: center; margin-top: 20px; margin-bottom: 20px;'>Chat</h2>
    """,
    unsafe_allow_html=True
)

# Initialize session state for conversation history
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Display conversation history
def display_conversation():
    for message in st.session_state.conversation_history:
        if message['role'] == 'user':
            st.markdown(f"<div style='background-color: #f5f5f5; color: #333333; padding: 10px; margin-bottom: 10px; border-radius: 10px; text-align: right; width: fit-content; max-width: 70%; margin-left: auto;'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #ffffff; color: #333333; font-size: 18px; padding: 12px; margin-bottom: 10px; border-radius: 10px; text-align: left; width: 70%; max-width: 800px; margin-right: auto;'><b>Bot:</b> {message['content']}</div>", unsafe_allow_html=True)

# Display conversation history - Calling
display_conversation()

# Input and submit button section
st.markdown("---")  # Horizontal rule for separation

# Create columns for input and button within the same line
col1, col2, col3 = st.columns([1, 5, 1])  # Width Adjustments

# Left margin for spacing
with col1:
    st.write("")

# Input user query
with col2:
    query = st.text_input("Enter your query:", key="input_query")
    st.write("") 

# Submit button
with col3:
    st.write("") 
    if st.button("Ask", key="submit_button"):
        if query:
            # Append user query to conversation history [Hope to display]
            st.session_state.conversation_history.append({"role": "user", "content": query})

            # Make API request to backend
            response = requests.post(
                "http://127.0.0.1:8000/query",
                json={"query": query}
            )
            
            if response.status_code == 200:
                bot_response = response.json()['response']
                # Append bot response to conversation history
                st.session_state.conversation_history.append({"role": "assistant", "content": bot_response})
            else:
                st.session_state.conversation_history.append({"role": "assistant", "content": "Sorry, couldn't process your request at the moment."})
            
            # Clear the input box for new query
            st.rerun()

        else:
            st.write("Please enter a query.")