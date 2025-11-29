# IntelliRAG
A fast, accurate, Retrieval-Augmented Generation (RAG) chatbot that answers questions from your uploaded documents and uses LLM fallback when needed.

IntelliRAG allows users to:
✔ Upload PDFs
✔ Ask questions directly based on document content
✔ Get citation-based answers grounded in real text
✔ Maintain conversation context across messages
✔ Use fallback LLM knowledge when docs don’t contain the answer
✔ Enjoy a simple, clean, responsive UI



IntelliRAG uses a classic RAG architecture:

1️⃣ Document Loading

All files inside /data are scanned automatically.

2️⃣ Chunking

Each document is split into manageable chunks using RecursiveCharacterTextSplitter.

3️⃣ Embeddings

Chunks → embeddings using the model:

sentence-transformers/all-MiniLM-L6-v2

4️⃣ FAISS Index

Embeddings are stored in a FAISS index for similarity search.

5️⃣ Retrieval

User query → embedding → FAISS → top-k chunks returned.

6️⃣ LLM Response With Context

LLM answers using ONLY retrieved text.
If no relevant text found → fallback to model knowledge, clearly labeled:

(BEST-EFFORT, NOT IN DOCUMENTS)

7️⃣ Conversation Memory

Past messages included for deeper contextual understanding.

🛠 Tech Stack
Backend

FastAPI

FAISS

Sentence Transformers

LangChain + Groq LLM

Pydantic

PyPDFLoader / CSVLoader / DocxLoader / ExcelLoader

Frontend

HTML

CSS

Vanilla JavaScript
