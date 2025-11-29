# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from src.vectorstore import FaissVectorStore

# load_dotenv()

# class RAGSearch:
#     def __init__(
#         self,
#         store: FaissVectorStore,      # <--- injected from main.py
#         llm_model: str = "llama-3.1-8b-instant"
#     ):
#         self.vectorstore = store

#         groq_api_key = os.getenv("GROQ_API_KEY")
#         if not groq_api_key:
#             raise ValueError("GROQ_API_KEY not set in .env")

#         self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
#         print(f"[INFO] Groq LLM initialized: {llm_model}")

#     def search_and_summarize(self, query: str, top_k: int = 5,
#                              allow_fallback: bool = True,
#                              temperature: float = 0.0) -> str:
#         """
#         Retrieval-first answer. If no relevant docs found and allow_fallback=True,
#         return a best-effort answer from LLM labelled as '(BEST-EFFORT, NOT IN DOCUMENTS)'.
#         temperature controls creativity (0.0 deterministic, 0.7 creative).
#         """
#         # Retrieval tuning
#         CANDIDATES = 30
#         DISTANCE_THRESHOLD = 1.1

#         # 1) retrieve filtered results
#         results = self.vectorstore.query(
#             query,
#             top_k=top_k,
#             candidates=CANDIDATES,
#             distance_threshold=DISTANCE_THRESHOLD
#         )

#         # 2) collect snippets + sources
#         snippets = []
#         sources = []
#         for r in results:
#             meta = r.get("metadata") or {}
#             text = meta.get("text", "").strip()
#             if text:
#                 snippets.append(text)
#                 # prefer human-friendly metadata if available (e.g., 'source' filename)
#                 sources.append(meta.get("source", str(r.get("index"))))

#         # 3) if we have snippets, ask LLM to answer using only them
#         if snippets:
#             context = "\n\n---\n\n".join(snippets)
#             prompt = f"""
# You are an assistant. Use ONLY the CONTEXT to answer the USER QUESTION.
# If the CONTEXT contains an answer, give a concise, correct answer and (optionally) a one-line explanation.
# If the CONTEXT does NOT contain information to answer the question, reply exactly with: NO_RELEVANT_DOCUMENTS
# Do NOT hallucinate.

# USER QUESTION:
# {query}

# CONTEXT:
# {context}

# Answer:
# """
#             # call LLM (include temperature if supported)
#             try:
#                 response = self.llm.invoke([prompt], temperature=temperature)
#             except TypeError:
#                 response = self.llm.invoke([prompt])
#             text_out = getattr(response, "content", str(response)).strip()

#             if text_out.startswith("NO_RELEVANT_DOCUMENTS"):
#                 # fall back path below if allowed
#                 snippets = []
#             else:
#                 # return answer with sources
#                 if sources:
#                     return f"{text_out}\n\nSources: {', '.join(sources)}"
#                 return text_out

#         # 4) No relevant documents (or LLM said NO_RELEVANT_DOCUMENTS)
#         if not allow_fallback:
#             return "No relevant documents found."

#         # 5) Fallback: let LLM answer from its own knowledge, but label it clearly
#         fallback_prompt = f"""
# The user asked: {query}

# There is no supporting CONTEXT available in the documents. Provide a concise, best-effort answer from general knowledge.
# Start the answer with the phrase "(BEST-EFFORT, NOT IN DOCUMENTS) " to clearly indicate this is not sourced from the corpus.
# Be cautious: avoid inventing precise facts that can't be verified.
# Answer:
# """
#         try:
#             response = self.llm.invoke([fallback_prompt], temperature=temperature)
#         except TypeError:
#             response = self.llm.invoke([fallback_prompt])
#         text_out = getattr(response, "content", str(response)).strip()
#         if not text_out.startswith("(BEST-EFFORT"):
#             text_out = "(BEST-EFFORT, NOT IN DOCUMENTS) " + text_out
#         return text_out


import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.vectorstore import FaissVectorStore

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        store: FaissVectorStore,      # injected from main.py
        llm_model: str = "llama-3.1-8b-instant"
    ):
        self.vectorstore = store

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not set in .env")

        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(
        self,
        query: str,
        history: list[str] = None,
        top_k: int = 5,
        allow_fallback: bool = True,
        temperature: float = 0.0
    ) -> str:
        """
        Retrieval-first RAG.
        If no relevant doc chunks found:
            -> optionally fall back to model knowledge (clearly labeled).
        history: list of previous messages (strings). We include last 5 turns.
        """

        # Build conversation-aware query
        if history is None:
            history = []
        # combine latest history turns (max 5) with current question
        full_query = "\n".join(history[-5:] + [query])

        # Retrieval tuning
        CANDIDATES = 100
        DISTANCE_THRESHOLD = 1.5  # tuned from your debug output

        # ---------------------------
        # 1) RAG Retrieval (use full_query)
        # ---------------------------
        results = self.vectorstore.query(
            full_query,
            top_k=top_k,
            candidates=CANDIDATES,
            distance_threshold=DISTANCE_THRESHOLD
        )

        snippets = []
        sources = []

        for r in results:
            meta = r.get("metadata") or {}
            text = meta.get("text", "").strip()
            if text:
                snippets.append(text)
                sources.append(meta.get("source", str(r.get("index"))))

        # ---------------------------
        # 2) If we found relevant snippets → answer from documents
        # ---------------------------
        if snippets:
            context = "\n\n---\n\n".join(snippets)

            prompt = f"""
You are an assistant. Use ONLY the CONTEXT to answer the USER QUESTION.
If the CONTEXT contains an answer, give a concise, correct answer.
If the CONTEXT does NOT contain information to answer the question, reply exactly with:
NO_RELEVANT_DOCUMENTS
Do NOT hallucinate.

USER QUESTION (conversation-aware):
{full_query}

CONTEXT:
{context}

Answer:
"""

            # Groq sometimes doesn't support temperature, so wrap safely
            try:
                response = self.llm.invoke([prompt], temperature=temperature)
            except:
                response = self.llm.invoke([prompt])

            text_out = getattr(response, "content", str(response)).strip()

            # LLM says no info → fallback allowed?
            if text_out.startswith("NO_RELEVANT_DOCUMENTS"):
                snippets = []   # force fallback
            else:
                # return good answer + sources
                if sources:
                    return f"{text_out}\n\nSources: {', '.join(sources)}"
                return text_out

        # ---------------------------
        # 3) No snippets found → fallback?
        # ---------------------------
        if not allow_fallback:
            return "No relevant documents found."

        # ---------------------------
        # 4) BEST-EFFORT fallback from LLM knowledge
        # ---------------------------
        fallback_prompt = f"""
The user asked (conversation-aware): {full_query}

No relevant document context is available.
Provide a concise, best-effort answer from general knowledge.
Start the answer with:
(BEST-EFFORT, NOT IN DOCUMENTS)

Avoid hallucinations or unverifiable details.

Answer:
"""

        try:
            response = self.llm.invoke([fallback_prompt], temperature=temperature)
        except:
            response = self.llm.invoke([fallback_prompt])

        text_out = getattr(response, "content", str(response)).strip()

        if not text_out.startswith("(BEST-EFFORT"):
            text_out = "(BEST-EFFORT, NOT IN DOCUMENTS) " + text_out

        return text_out
# print("[DEBUG] full_query:", full_query)
# print("[DEBUG] retrieved (idx, dist, source):", [(r['index'], r['distance'], r.get('metadata',{}).get('source')) for r in results])
