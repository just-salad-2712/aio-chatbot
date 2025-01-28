import streamlit as st
import requests
import os

st.title("📝 Resume Improver Chatbot")

API_URL = "http://localhost:8000"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "resume_uploaded" not in st.session_state:
    st.session_state.resume_uploaded = False

# File uploader section
if not st.session_state.resume_uploaded:
    uploaded_file = st.file_uploader("Upload your resume", type=["docx"])
    if uploaded_file:
        # Upload the resume
        files = {"file": uploaded_file}
        try:
            response = requests.post(
                f"{API_URL}/upload-resume",
                files=files
            )
            response.raise_for_status()  # Raise exception for error status codes
            
            st.session_state.resume_uploaded = True
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I've analyzed your resume. What specific aspects would you like me to help you improve?"
            })
            st.rerun()
        except requests.RequestException as e:
            st.error(f"Error uploading resume: {str(e)}")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.resume_uploaded:
    # Get user input
    if prompt := st.chat_input("What would you like to improve in your resume?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"query": prompt}
                )
                response.raise_for_status()
                
                bot_response = response.json()["response"]
                response_placeholder.write(bot_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_response
                })
            except requests.RequestException as e:
                response_placeholder.write(f"Error: {str(e)}")

# Add a reset button
if st.session_state.resume_uploaded:
    if st.button("Start Over"):
        try:
            # Delete the resume file
            response = requests.post(f"{API_URL}/delete-resume")
            response.raise_for_status()
            
            # Reset session state
            st.session_state.resume_uploaded = False
            st.session_state.messages = []
            st.rerun()
        except requests.RequestException as e:
            st.error(f"Error deleting resume: {str(e)}")

