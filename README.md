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

```mermaid
graph TD
    %% Styling for a clean, professional look
    classDef process fill:#f8fafc,stroke:#64748b,stroke-width:1px,color:#0f172a;
    classDef decision fill:#fff7ed,stroke:#f97316,stroke-width:1px,color:#0f172a;
    classDef eval fill:#eff6ff,stroke:#3b82f6,stroke-width:2px,color:#0f172a;
    classDef output fill:#f0fdf4,stroke:#22c55e,stroke-width:2px,color:#0f172a;

    %% Flow Nodes
    Q([👤 Question]):::process --> Route{🔀 Routing}:::decision
    
    Route -- "Related to Index" --> Retrieve[📥 Retrieve Documents]:::process
    Route -- "Unrelated to Index" --> WebSearch[🌐 Web Search]:::process
    
    WebSearch --> Rethink[🧠 Rethink Context]:::process
    Rethink --> Retrieve
    
    Retrieve --> Grade[📝 Grade Documents]:::process
    Grade --> CheckRelevance{❌ Any doc irrelevant?}:::decision
    
    CheckRelevance -- "Yes" --> WebSearch
    CheckRelevance -- "No" --> Generate[🤖 Generate Answer]:::process
    
    Generate --> RAGAS
    
    subgraph RAGAS [RAGAS Evaluation Module]
        direction LR
        M1[📊 Context Metrics: Precision / Recall]:::eval
        M2[🛡️ Falsifiability Metric]:::eval
    end
    
    RAGAS --> CheckHall{🤥 Hallucinations? / Answers Question?}:::decision
    
    CheckHall -- "Yes / Failed" --> Generate
    CheckHall -- "No / Passed" --> Out([✅ Final Output]):::output
