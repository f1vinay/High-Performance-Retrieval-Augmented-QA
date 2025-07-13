import streamlit as st
import time
import statistics
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Paths
CHROMA_PATH = "chroma"
COLLECTION_NAME = "rag"

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("📚 Ask Your PDFs")

@st.cache_resource
def load_qa_chain():
    # Load ChromaDB client
    client = chromadb.Client(Settings(persist_directory=CHROMA_PATH))

    # 🛠 Check if 'rag' collection exists — create it if not
    existing = [col.name for col in client.list_collections()]
    if COLLECTION_NAME not in existing:
        st.error(f"❌ Collection '{COLLECTION_NAME}' not found in ChromaDB.\n"
                 f"Please run `ingest_chroma.py` first.")
        st.stop()

    collection = client.get_collection(name=COLLECTION_NAME)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Manual retriever function
    def retrieve_docs(query, k=4):
        query_vector = model.encode(query).tolist()
        results = collection.query(query_embeddings=[query_vector], n_results=k)
        return [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    # LLM
    llm = ChatOpenAI(
        temperature=0,
        model="Gpt4o",
        openai_api_base="https://aiunifier.wonderfulrock-83cb33fd.australiaeast.azurecontainerapps.io"
    )

    # QA Chain logic
    def run_qa(query):
        docs = retrieve_docs(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"Answer the question based on the following context:\n\n{context}\n\nQ: {query}\nA:"
        return llm.predict(prompt)

    return run_qa

qa_chain = load_qa_chain()

if "queries" not in st.session_state:
    st.session_state["queries"] = []

query = st.text_input("Ask a question about the PDFs:")
if query:
    start = time.time()
    result = qa_chain(query)
    duration = time.time() - start

    st.session_state["queries"].append({"query": query, "time": duration})
    st.markdown(f"### ✅ Answer:\n{result}")
    st.markdown(f"⏱️ Response Time: {duration:.2f} seconds")

if st.session_state["queries"]:
    st.markdown("---")
    st.markdown("### 📊 Query Performance")
    query_data = st.session_state["queries"]

    median_time = statistics.median([q["time"] for q in query_data])
    st.markdown(f"**Median Response Time:** {median_time:.2f} seconds")

    order = st.radio("Sort by", ["Ascending", "Descending"], horizontal=True)
    sorted_queries = sorted(
        query_data, key=lambda x: x["time"], reverse=(order == "Descending")
    )[:5]

    st.markdown("**Top-5 Queries by Response Time:**")
    for i, q in enumerate(sorted_queries, 1):
        st.write(f"{i}. {q['query']} — {q['time']:.2f} sec")
