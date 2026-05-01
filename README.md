Self-Correcting RAG System
A resilient AI agent built to answer questions over your documents with high accuracy.

Standard RAG systems often guess or hallucinate when they can't find the right context. This system fixes that. Inspired by the Self-RAG paper, it evaluates its own work at every step—grading documents, checking for hallucinations, and automatically correcting itself before it ever responds to you.

✨ Key Features
Self-Correcting Logic: Uses LangGraph to build an agentic loop. If the AI realizes its first attempt didn't find the right information, it searches again before answering.
Document Grading: Retrieves multiple document chunks, evaluates their relevance to the question, and throws out the useless ones.
Hallucination Check: After generating an answer, the AI cross-checks its response against the documents. If it made something up, it discards the answer and tries again.
Session Memory: Maintains the context of your conversation across multiple questions seamlessly.
Production-Ready: Fully Dockerized. Spins up a FastAPI backend and a Streamlit frontend in minutes without wrestling with local Python environments.
🧠 Under the Hood: The Evaluation Flow
Unlike standard pipelines that fetch a document and immediately generate an answer, this system uses a strict grading pipeline.

Self-RAG Evaluation and Correction Flow

How the loop works:

Routing: We check if the question even requires looking at the document database.
Grade Documents: If a retrieved document isn't highly relevant to the specific question, it is thrown out.
Hallucination Check: After drafting an answer, the system asks itself: "Is this actually supported by the documents, or did I just make it up?"
Correction Loop: If it hallucinated or failed to answer the prompt, it loops back to retrieve better context or triggers a fallback search.
🛠️ Tech Stack
Logic & Framework: LangChain, LangGraph
LLM & Embeddings: OpenAI (GPT-4 / GPT-3.5)
Vector Store: ChromaDB
Backend: FastAPI & Uvicorn
Frontend: Streamlit
Deployment: Docker & Docker Compose
🗂️ The File Map
If you are looking at the codebase, here is exactly what each piece does in plain English:

frontend/ (The User Interface)
This is what you see in the browser.

app.py: The face of the application. It renders the chat window, takes your input, sends it to the backend, and beautifully streams the AI's response.
Dockerfile: Tells Docker how to package the Streamlit app into its own isolated container.
requirements.txt: Only two things are needed here: streamlit (for the UI) and requests (to talk to the backend).
backend/ (The Brain)
This is where the actual AI work happens.

main.py: The receptionist. It sets up the FastAPI server, listens for incoming chat messages from the frontend, and passes them to the AI brain.
graph.py: The engine. This contains the LangGraph logic, the document grading, the hallucination checks, and the self-correction loops.
.env: The vault. This is where your OPENAI_API_KEY lives. It is strictly hidden from GitHub so your secrets stay safe.
Dockerfile: Tells Docker how to package the FastAPI server and install all the heavy AI libraries.
requirements.txt: The shopping list of Python packages the AI needs to think (FastAPI, LangChain, OpenAI, ChromaDB, etc.).
Root Files (The Glue)
docker-compose.yml: The conductor. Instead of starting the frontend and backend separately, this file starts both containers at the same time and connects them securely.
.gitignore: The bouncer. Prevents heavy, useless, or secret files (like .env keys or vector databases) from being uploaded to GitHub.
images/: Contains diagrams and screenshots for the README.
🚀 Quick Start (Local Deployment)
You don't need to install Python or mess with virtual environments. Docker handles everything.

1. Clone the code

git clone https://github.com/Ragrawal2004/Self_Correcting_Rag_System.gitcd Self_Correcting_Rag_System
2. Add your OpenAI Key
Create a .env file inside the backend/ folder and add your key:

text

OPENAI_API_KEY=sk-your-key-here
3. Start the app

bash

docker-compose up --build
Give it a minute to download libraries and build. Once it's done:

Open the Chat UI: http://localhost:8501
View API Docs: http://localhost:8000/docs
