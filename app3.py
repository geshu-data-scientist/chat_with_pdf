import streamlit as st
import os
import faiss
import pickle
import numpy as np
import logging
import time
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

# --------------------------
# CONFIG
# --------------------------
HF_GGUF_URL = "https://huggingface.co/ggml-org/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_0.gguf"
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_DIR = "./models"
VECTOR_STORE_PATH = "./vector_store"
MODEL_NAME = "Meta-Llama-3-8B-Instruct-Q4_0.gguf"

TOP_K = 3
CHUNK_SIZE = 180
CHUNK_OVERLAP = 40
MAX_CONTEXT = 4096

# --------------------------
# Logging
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger("RAGApp")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)


# --------------------------
# Helper Functions
# --------------------------
def download_if_missing(url, filepath):
    """Download a file from URL ONLY if missing."""
    if os.path.exists(filepath):
        return filepath

    import requests

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return filepath


def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    text = []
    for page in reader.pages:
        if page.extract_text():
            text.append(page.extract_text())
    return "\n".join(text)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def build_faiss_index(emb):
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


# --------------------------
# Load models (run-time download)
# --------------------------
@st.cache_resource
def load_models():
    st.write("Downloading models at runtime from HuggingFace...")

    # 1) Download GGUF model from HuggingFace
    local_gguf = os.path.join(MODEL_DIR, MODEL_NAME)
    download_if_missing(HF_GGUF_URL, local_gguf)

    # 2) Download embedding model
    embed_model = SentenceTransformer(HF_EMBED_MODEL)

    # 3) Load GGUF using GPT4All (no further downloads)
    llm = GPT4All(
        model_name=MODEL_NAME,
        model_path=MODEL_DIR,
        allow_download=False,   # prevents GPT4All from trying to download
        n_ctx=MAX_CONTEXT,
        device="cpu",
    )

    return llm, embed_model


# --------------------------
# Streamlit App
# --------------------------
st.title("📄 Offline PDF Chat with Runtime Model Download")

llm, embedder = load_models()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None

# Sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing PDF..."):
            text = extract_pdf_text(uploaded_file)
            chunks = chunk_text(text)
            emb = embedder.encode(chunks)
            emb = np.array(emb).astype("float32")

            index = build_faiss_index(emb)

            st.session_state.vector_store = index
            st.session_state.chunks = chunks

            st.success("Document indexed successfully!")


# Chat
user_query = st.chat_input("Ask something about the PDF...")

if user_query:
    if st.session_state.vector_store is None:
        st.error("Upload and process PDF first.")
    else:
        q_emb = embedder.encode([user_query])
        q_emb = np.array(q_emb).astype("float32")
        faiss.normalize_L2(q_emb)

        D, I = st.session_state.vector_store.search(q_emb, TOP_K)
        retrieved_chunks = [st.session_state.chunks[i] for i in I[0]]

        context = "\n\n".join(retrieved_chunks)
        prompt = f"""
Answer using ONLY the context below:

{context}

QUESTION: {user_query}

ANSWER:
"""

        response = llm.generate(prompt, max_tokens=512)
        st.chat_message("assistant").markdown(response)
