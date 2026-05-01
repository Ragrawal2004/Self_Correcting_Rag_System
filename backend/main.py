import sys
from pathlib import Path
# Add the parent directory to Python's path so it can find agent.py
sys.path.append(str(Path(__file__).resolve().parent.parent))


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
import logging

from backend.graph import run_agent

# --- PROFESSIONAL LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FASTAPI SETUP ---
app = FastAPI(title="Self-Correcting RAG API")

# Allow Streamlit frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (safe for local/portfolio projects)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (In production, you'd use Redis or a database)
sessions: Dict[str, List[Dict]] = {}

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    feedback: Optional[str] = None  # Triggers the corrective feedback loop

# --- API ENDPOINT ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Receives a question from the Streamlit UI,
    runs it through the Self-Correcting RAG graph,
    handles chat history & feedback loops, and returns the result.
    """
    # 1. Handle Session ID
    if not request.session_id:
        request.session_id = str(uuid.uuid4())
    
    if request.session_id not in sessions:
        sessions[request.session_id] = []

    logger.info(f"🔔 Received: '{request.message}' | Session: {request.session_id} | Feedback: {request.feedback}")

    # 2. Handle Feedback Loop (Force web search if user clicks "Bad Response")
    actual_question = request.message
    if request.feedback == "negative":
        if not sessions[request.session_id]:
            return {"response": "No history to correct yet.", "session_id": request.session_id, "history": [], "metrics": ""}
        
        # Grab the last user question to re-ask
        actual_question = sessions[request.session_id][-1]["content"] 
        if sessions[request.session_id][-1]["role"] == "assistant" and len(sessions[request.session_id]) >= 2:
            actual_question = sessions[request.session_id][-2]["content"]

    # 3. Convert JSON history back to LangChain Message objects
    from langchain_core.messages import HumanMessage, AIMessage
    lc_history = []
    for msg in sessions[request.session_id]:
        if msg["role"] == "user": 
            lc_history.append(HumanMessage(content=msg["content"]))
        else: 
            lc_history.append(AIMessage(content=msg["content"]))

    # 4. Run the LangGraph Agent
    result = run_agent(
        question=actual_question, 
        history=lc_history, 
        feedback=request.feedback
    )

    # 5. Update session store with the newly returned history
    sessions[request.session_id] = result["history"]

    logger.info("✅ Final Answer generated.")
    
    # 6. Return formatted response
    return {
        "response": result["response"],
        "session_id": request.session_id,
        "history": result["history"],
        "metrics": result["metrics"]
    }