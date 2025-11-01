import streamlit as st
import requests
from shared.config import Config
import shared.logger as logger

# Headers for API requests
HEADERS = {
    "x-api-key": Config.ACCESS_KEY,
    "Content-Type": "application/json"
}

# Initialize session state
for key, default in {
    "generating": False,
    "generation_complete": False,
    "generation_success": False,
    "error_message": "",
    "download_url": "",
    "filename": "",
    "user_prompt": "",
    "topic": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def call_generate_api(query, topic, doc_type):
    """Call the FastAPI generate endpoint (blocking until finished)."""
    try:
        payload = {
            "query": query,
            "topic": topic,
            "type": doc_type.lower()
        }

        response = requests.post(
            "http://app:6789/generate",
            json=payload,
            headers=HEADERS,
            timeout=600   # allow longer time for big jobs
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"API Error: {response.status_code} - {response.text}"}

    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)}"}

def download_file(download_url):
    """Download file from the API"""
    try:
        response = requests.get(
            f"http://app:6789{download_url}",
            headers=HEADERS,
            timeout=60
        )
        if response.status_code == 200:
            return response.content
        return None
    except Exception:
        return None

# STREAMLIT UI PAGE
st.title("PPTGen-AI - AI PowerPoint and PDF Generator")

# Option to choose between PPT and PDF
doc_type = st.radio("Choose the type of document to generate:", ["PPT", "PDF"], key="doc_type")

# Prompt input (fixed height, stable key)
st.session_state.user_prompt = st.text_area(
    "Enter your prompt (max 200 characters):",
    value=st.session_state.user_prompt,
    max_chars=200,
    height=150,
    placeholder="Summarize the evolution of AI in education...",
    key="prompt_input"
)

# Topic input
st.session_state.topic = st.text_input(
    "Enter the topic:",
    value=st.session_state.topic,
    placeholder="AI in education",
    key="topic_input"
)

# Generate button
if st.button("Generate", disabled=st.session_state.generating):
    if not st.session_state.user_prompt.strip():
        st.warning("Please enter a prompt before generating.")
    elif not st.session_state.topic.strip():
        st.warning("Please enter a topic before generating.")
    else:
        # Reset states
        st.session_state.generating = True
        st.session_state.generation_complete = False
        st.session_state.generation_success = False
        st.session_state.error_message = ""
        st.session_state.download_url = ""
        st.session_state.filename = ""

        # Show static waiting message
        with st.spinner("Generating. Please wait... This may take a while..."):
            result = call_generate_api(st.session_state.user_prompt, st.session_state.topic, doc_type)

        # Handle API response
        st.session_state.generating = False
        st.session_state.generation_complete = True

        if result.get("success", False):
            st.session_state.generation_success = True
            file_info = result.get("file_info", {})
            st.session_state.filename = file_info.get("filename", "document")
            st.session_state.download_url = result.get("download_url", "")
        else:
            st.session_state.generation_success = False
            st.session_state.error_message = result.get("error", "Unknown error occurred")

        st.rerun()


# Handle generation results
if st.session_state.generation_complete:
    if st.session_state.generation_success:
        st.success("Yes! File generated successfully!")

        if st.session_state.download_url and st.session_state.filename:
            file_data = download_file(st.session_state.download_url)
            if file_data:
                mime_type = "application/pdf" if doc_type == "PDF" else \
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                st.download_button(
                    label=f"📥 Download {st.session_state.filename}",
                    data=file_data,
                    file_name=st.session_state.filename,
                    mime=mime_type
                )
    else:
        st.error(st.session_state.error_message)
