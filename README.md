## **Self-Correcting RAG System**
A resilient AI agent built to answer questions over your documents with high accuracy.

Standard RAG systems often guess or hallucinate when they fail to retrieve the right context. This system addresses that limitation. Inspired by the Self-RAG approach, it evaluates its own outputs at every step—grading documents, detecting hallucinations, and automatically correcting itself before responding.

---

## ✨ Key Features

**Self-Correcting Logic:**  
Uses LangGraph to build an agentic loop. If the system determines that the initial retrieval is insufficient, it re-queries the knowledge base before generating a response.

**Document Grading:**  
Retrieves multiple document chunks, evaluates their semantic relevance to the query, and filters out low-quality or irrelevant context.

**Hallucination Check:**  
After generating an answer, the system validates it against retrieved documents. Unsupported or fabricated responses are discarded and regenerated.

**Session Memory:**  
Maintains conversational context across multiple interactions, enabling coherent multi-turn dialogue.

**Production-Ready:**  
Fully containerized using Docker. Deploys a FastAPI backend and Streamlit frontend with minimal setup.

---

## 🧠 Under the Hood: The Evaluation Flow

Unlike traditional pipelines that retrieve context and immediately generate answers, this system introduces a structured evaluation pipeline.

**Self-RAG Evaluation and Correction Flow**

### How the loop works:

**Routing:**  
Determines whether the query requires retrieval from the document store or can be answered directly.

**Grade Documents:**  
Evaluates retrieved documents and discards those that do not meet relevance thresholds.

**Hallucination Check:**  
Validates whether the generated answer is grounded in retrieved evidence.

**Correction Loop:**  
If the response is incomplete or hallucinated, the system re-enters the retrieval phase or triggers fallback strategies.

---

## 🛠️ Tech Stack

**Logic & Framework:** LangChain, LangGraph  
**LLM & Embeddings:** OpenAI (GPT-4 / GPT-3.5)  
**Vector Store:** ChromaDB  
**Backend:** FastAPI, Uvicorn  
**Frontend:** Streamlit  
**Deployment:** Docker, Docker Compose  

---

## 🗂️ The File Map

If you are exploring the codebase, here is a clear breakdown of each component:

### **frontend/** (The User Interface)
This is the user-facing layer of the application.

- **app.py:** Handles UI rendering, user input, backend communication, and streaming responses.  
- **Dockerfile:** Defines the container environment for the Streamlit app.  
- **requirements.txt:** Contains minimal dependencies (Streamlit, requests).

---

### **backend/** (The Brain)
This is where the core AI logic resides.

- **main.py:** Entry point for the FastAPI server; routes requests to the processing pipeline.  
- **graph.py:** Implements LangGraph workflows including retrieval, grading, hallucination checks, and correction loops.  
- **.env:** Stores sensitive credentials such as `OPENAI_API_KEY` (excluded from version control).  
- **Dockerfile:** Defines the backend container with required AI dependencies.  
- **requirements.txt:** Lists all backend dependencies (FastAPI, LangChain, OpenAI, ChromaDB, etc.).

---

### **Root Files (The Glue)**

- **docker-compose.yml:** Orchestrates frontend and backend containers, enabling seamless communication.  
- **.gitignore:** Excludes sensitive and unnecessary files (e.g., `.env`, vector databases).  
- **images/:** Contains architecture diagrams and README assets.

---

## 🚀 Quick Start (Local Deployment)

No need to install Python or manage environments—Docker handles everything.

### 1. Clone the repository

```bash
git clone https://github.com/Ragrawal2004/Self_Correcting_Rag_System.git
cd Self_Correcting_Rag_System
