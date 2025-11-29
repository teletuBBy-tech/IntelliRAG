from src.vectorstore import FaissVectorStore

store = FaissVectorStore("faiss_store")
store.load()

print(store.query(
    "what are logarithmic properties",
    debug=True,
    distance_threshold=None,
    top_k=5,
    candidates=30
))
