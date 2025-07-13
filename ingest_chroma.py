import os
import fitz  # PyMuPDF
import chromadb
from tqdm import tqdm
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma"
PDF_DIR = "data/corpus"
COLLECTION_NAME = "rag"

# Use local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_text_from_pdfs(folder):
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith(".pdf"):
            with fitz.open(os.path.join(folder, fname)) as doc:
                text = "\n".join(page.get_text() for page in doc)
                docs.append((fname, text))
    return docs

def chunk_text(text, chunk_size=500, overlap=50):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

def embed_and_store(docs):
    client = chromadb.Client(Settings(persist_directory=CHROMA_PATH))
    
    # Debug: Show current collections
    print("📦 Collections in DB before:", client.list_collections())

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    for doc_id, (fname, text) in enumerate(tqdm(docs)):
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{doc_id}_{i}"],
                metadatas=[{"source": fname}]
            )

    print("✅ Ingestion complete. Chroma collection created.")
    print("📦 Collections in DB after:", client.list_collections())

if __name__ == "__main__":
    os.makedirs(PDF_DIR, exist_ok=True)
    docs = load_text_from_pdfs(PDF_DIR)
    if not docs:
        raise RuntimeError("No PDFs found in data/corpus/")
    embed_and_store(docs)
