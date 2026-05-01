import streamlit as st
import requests

# ==========================================
# STREAMLIT UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Local RAG Agent", 
    page_icon="🤖", 
    layout="centered"
)

st.title("🤖 Self-Correcting RAG Agent")
st.caption("Powered by LLaMA-3, ChromaDB, and LangGraph")
st.markdown("---")

# Initialize chat history AND session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to handle the "Bad Response" feedback loop
def trigger_feedback_loop():
    """Forces the backend to re-search the web and regenerate the last answer."""
    with st.spinner("🧠 Forcing Web Search for a better answer..."):
        payload = {
            "session_id": st.session_state.session_id,
            "message": "dummy", # Ignored by backend when feedback is negative
            "feedback": "negative"
        }
        try:
            res = requests.post("http://localhost:8000/chat", json=payload, timeout=120)
            if res.status_code == 200:
                data = res.json()
                # Replace entire history with the newly corrected history from backend
                st.session_state.session_id = data["session_id"]
                st.session_state.messages = data["history"]
                st.rerun() # Force UI to refresh instantly
            else:
                st.error(f"Backend error: {res.text}")
        except Exception as e:
            st.error(f"Failed to connect: {e}")

# Display previous chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add a "Bad Response" button under every AI message
        if message["role"] == "assistant":
            if st.button("👎 Bad Response / Force Web Search", key=f"btn_{i}"):
                trigger_feedback_loop()

# React to user input
if prompt := st.chat_input("Ask me about AI agents, prompt engineering, or current events..."):
    
    # 1. Display user message immediately
    st.chat_message("user").markdown(prompt)
    # Note: We don't append to st.session_state.messages here anymore. 
    # We let the backend return the full, official history so they never get out of sync.

    # 2. Send request to FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("🧠 Thinking (Routing -> Retrieval -> Self-Correction)..."):
            try:
                # Payload MUST match backend/main.py expectations
                payload = {
                    "session_id": st.session_state.session_id,
                    "message": prompt,
                    "feedback": None
                }
                
                # Call the FastAPI backend
                response = requests.post(
                    "http://localhost:8000/chat", 
                    json=payload,
                    timeout=120 # LLaMA-3 can take a moment locally!
                )
                response.raise_for_status() # Check for HTTP errors (like 422)
                
                # Extract the data from the JSON response
                data = response.json()
                
                # Sync Session ID and Full Chat History
                st.session_state.session_id = data["session_id"]
                st.session_state.messages = data["history"]
                
                # Display the latest response to the user
                st.markdown(data["response"])
                
            except requests.exceptions.HTTPError as e:
                    try:
                        error_msg = e.response.json()['detail'][0]['msg']
                        st.error(f"❌ **Backend Error:** {error_msg}")
                    except Exception:
                        st.error(f"❌ **Backend Error:** {e.response.text or str(e)}")
            except requests.exceptions.ConnectionError:
                st.error("❌ **Connection Error:** Make sure the FastAPI backend is running! \n\nDid you run `uvicorn backend.main:app --reload` in your terminal?")
            except Exception as e:
                st.error(f"❌ **An error occurred:** {e}")