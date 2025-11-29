# from fastapi import FastAPI
# from pydantic import BaseModel
# from src.vectorstore import FaissVectorStore
# from src.search import RAGSearch

# app = FastAPI(title="RAG Chatbot API")
# from fastapi.middleware.cors import CORSMiddleware

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # or ["http://localhost:5173"] for safety
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # Load FAISS index once when server starts
# store = FaissVectorStore("faiss_store")
# store.load()

# rag_search = RAGSearch()

# # Request body model
# class QueryRequest(BaseModel):
#     question: str
#     top_k: int = 3  # default

# # API endpoint for frontend
# @app.post("/ask")
# def ask_rag(query: QueryRequest):
#     summary = rag_search.search_and_summarize(query.question, top_k=query.top_k)
#     return {"summary": summary}

# # Optional root endpoint
# @app.get("/")
# def home():
#     return {"message": "✅ RAG API is running!"}
  
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import UploadFile, File, BackgroundTasks
import shutil

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # can restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
FAISS_DIR = "faiss_store"

# ---------------------------------------------------------
# AUTO-BUILD FAISS INDEX ON STARTUP
# ---------------------------------------------------------
print("\n📄 Checking documents in:", DATA_DIR)
documents = load_all_documents(DATA_DIR)

store = FaissVectorStore(FAISS_DIR)

# Condition 1: FAISS folder doesn't exist
# Condition 2: Folder exists but empty
if not os.path.exists(FAISS_DIR) or len(os.listdir(FAISS_DIR)) == 0:
    print("⚡ No FAISS index found. Building new index...")
    store.build_from_documents(documents)
    store.save()
else:
    print("ℹ️ FAISS index found. Loading existing index...")
    store.load()

# ---------------------------------------------------------
# RAG SEARCH USES **THE SAME STORE INSTANCE**
# ---------------------------------------------------------
rag_search = RAGSearch(store=store)   # IMPORTANT

# ---------------------------------------------------------
# API MODELS
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    history: list[str] = []     # <-- NEW: conversation history (list of strings)
    top_k: int = 3

# ---------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------
@app.post("/ask")
def ask_rag(query: QueryRequest):
    summary = rag_search.search_and_summarize(
        query=query.question,
        history=query.history,
        top_k=query.top_k
    )
    return {"summary": summary}

# ------------------------------
# Upload new PDF + rebuild FAISS
# ------------------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # 1) Validate file type
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are allowed."}

    # 2) Save file to data/ directory
    save_path = os.path.join(DATA_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3) Rebuild FAISS in background
    def rebuild():
        global store, rag_search
        print("⚡ Rebuilding FAISS index after new upload...")
        docs = load_all_documents(DATA_DIR)

        store = FaissVectorStore(FAISS_DIR)
        store.build_from_documents(docs)
        store.save()
        store.load()

        rag_search = RAGSearch(store=store)
        print("✅ Rebuild complete.")

    # background_tasks will be provided by FastAPI when request is handled
    if background_tasks is not None:
        background_tasks.add_task(rebuild)
    else:
        # fallback: run synchronously (rare)
        rebuild()

    return {"message": "PDF uploaded successfully. Index rebuilding in background."}


@app.get("/")
def home():
    return {"message": "✅ RAG API is running!"}
