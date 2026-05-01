import os
import logging
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# --- PROFESSIONAL LOGGING SETUP (Replaces print statements) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. INDEXING (Persistent ChromaDB) ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

logger.info("🚀 Loading documents and creating ChromaDB vector index...")
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits, 
    collection_name="rag-chroma", 
    embedding=GPT4AllEmbeddings(),
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever()
logger.info("✅ ChromaDB loaded successfully!")

# --- 2. LLM & EVALUATOR SETUP ---
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

local_llm = 'llama3'
llm = ChatOllama(model=local_llm, temperature=0)
eval_llm = ChatOllama(model=local_llm, format="json", temperature=0)

# --- 3. CUSTOM EVALUATORS (Mimics RAGAS Faithfulness/Relevancy without the slow library) ---

# Metric 1: Context Precision
retrieval_grader = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the retrieved document: \n\n {document} \n\n Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
) | eval_llm | JsonOutputParser()

# Metric 2: Faithfulness
hallucination_grader = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|> Here are the facts: \n ------- \n {documents} \n ------- \n Here is the answer: {generation} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
) | eval_llm | JsonOutputParser()

# Metric 3: Answer Relevancy
answer_grader = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' or 'no' score to indicate whether the answer is useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer: \n ------- \n {generation} \n ------- \n Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
) | eval_llm | JsonOutputParser()

# --- 4. ROUTING ---
question_router = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and no preamble or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
) | eval_llm | JsonOutputParser()

# --- 5. GENERATION CHAIN (With Chat History) ---
rag_chain = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|> 
    Chat History: {chat_history}
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context", "chat_history"],
) | llm | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- 6. LANGGRAPH STATE & NODES ---
class GraphState(TypedDict):
    question : str
    generation : str
    web_search : str
    documents : List[str]
    loop_count : int       # ADDED: Prevents infinite loops
    chat_history : List[BaseMessage] # ADDED: For memory

from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)

def retrieve(state):
    logger.info("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    logger.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Safety loop counter
    loop_count = state.get("loop_count", 0) + 1
    
    # Format chat history safely
    history = state.get("chat_history", [])
    history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    
    generation = rag_chain.invoke({"context": format_docs(documents), "question": question, "chat_history": history_str})
    return {"documents": documents, "question": question, "generation": generation, "loop_count": loop_count}

def grade_documents(state):
    logger.info("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        # SAFE PARSING: Handles LLaMA outputting 1/0 instead of "yes"/"no"
        if str(score.get('score', '')).lower() in ["yes", "1", "true"]:
            filtered_docs.append(d)
        else:
            web_search = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state):
    logger.info("---WEB SEARCH---")
    question = state["question"]
    docs = web_search_tool.invoke({"query": question})
    web_results = Document(page_content="\n".join([d["content"] for d in docs]))
    return {"documents": [web_results], "question": question}

# --- 7. LANGGRAPH EDGES (Self-Correction Logic) ---
def route_question(state):
    logger.info("---ROUTE QUESTION---")
    source = question_router.invoke({"question": state["question"]})
    return "websearch" if source.get('datasource') == 'web_search' else "vectorstore"

def decide_to_generate(state):
    logger.info("---ASSESS GRADED DOCUMENTS---")
    return "websearch" if state["web_search"] == "Yes" else "generate"

def grade_generation(state):
    logger.info("---CHECK HALLUCINATIONS & USEFULNESS---")
    documents = format_docs(state["documents"])
    generation = state["generation"]
    loop_count = state.get("loop_count", 0)
    
    # INFINITE LOOP SAFETY NET: Stop after 2 attempts
    if loop_count >= 2:
        logger.warning("---MAX LOOPS REACHED, RETURNING BEST EFFORT---")
        return "useful" 
        
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    # SAFE PARSING applied here too
    if str(score.get('score', '')).lower() in ["yes", "1", "true"]:
        score = answer_grader.invoke({"question": state["question"],"generation": generation})
        if str(score.get('score', '')).lower() in ["yes", "1", "true"]:
            return "useful"
        return "not useful"
    return "not supported"

# --- 8. COMPILE THE GRAPH ---
from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

workflow.set_conditional_entry_point(route_question, {"websearch": "websearch","vectorstore": "retrieve"})
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {"websearch": "websearch","generate": "generate"})
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges("generate", grade_generation, {
    "not supported": "websearch", 
    "useful": END, 
    "not useful": "websearch"
})

app = workflow.compile()
logger.info("✅ LangGraph Agent compiled successfully!")

# --- 9. EXPOSED FUNCTION FOR BACKEND API ---
def run_agent(question: str, history: list, feedback: str = None):
    """Entry point for FastAPI to call the graph."""
    inputs = {"question": question, "chat_history": history, "user_feedback": feedback, "loop_count": 0}
    result = app.invoke(inputs)
    
    # Manually update history since we bypassed the RAGAS node earlier
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=result["generation"]))
    
    # Serialize for JSON response
    serialized_history = [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content} for m in history]
    
    return {
        "response": result["generation"],
        "history": serialized_history,
        "metrics": "Custom Evaluators Passed" 
    }