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

# ----------------------
# Configuration
# ----------------------
MODEL_NAME = "Meta-Llama-3-8B-Instruct-Q4_0.gguf"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_BASE_PATH = "./models"
VECTOR_STORE_PATH = "./vector_store"
MAX_CONTEXT = 4096
TOP_K = 3
CHUNK_SIZE = 180
CHUNK_OVERLAP = 40
EMBEDDING_DIM = None

# ----------------------
# Logging
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("OfflineRAG")

if not os.path.exists(VECTOR_STORE_PATH):
    os.makedirs(VECTOR_STORE_PATH)


def file_id_from_name(file_name: str) -> str:
    return file_name.replace(".pdf", "").replace(" ", "_")


def vector_paths_for(file_name: str):
    fid = file_id_from_name(file_name)
    return os.path.join(VECTOR_STORE_PATH, f"{fid}.index"), os.path.join(VECTOR_STORE_PATH, f"{fid}.pkl")

@st.cache_resource
def load_models():
    logger.info("Loading models (offline)...")

    llm_path = os.path.join(MODEL_BASE_PATH, MODEL_NAME)
    if not os.path.exists(llm_path):
        st.error(f"LLM not found at: {llm_path} (offline only).")
        st.stop()

    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to load embedder '{EMBEDDING_MODEL_NAME}': {e}")
        st.error(f"Failed to load embedder: {e}")
        st.stop()

    try:
        llm = GPT4All(
            MODEL_NAME,
            model_path=MODEL_BASE_PATH,
            allow_download=False,
            n_ctx=MAX_CONTEXT,
            device='cpu'
        )
    except Exception as e:
        logger.error(f"Failed to load GPT4All model: {e}")
        st.error(f"Failed to load GPT4All model: {e}")
        st.stop()

    logger.info("Models loaded successfully.")
    return llm, embedder


def text_from_pdf(file_obj) -> str:
    reader = PdfReader(file_obj)
    text_parts = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            text_parts.append(txt)
    return "\n".join(text_parts)


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks


def build_faiss_index(embeddings: np.ndarray):
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def save_vector_store(index, chunks, file_name: str):
    index_path, chunks_path = vector_paths_for(file_name)
    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved vector store for {file_name} -> {index_path}")


def load_vector_store(file_name: str):
    index_path, chunks_path = vector_paths_for(file_name)
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return None, None


st.set_page_config(page_title="Offline PDF Chat (RAG)", layout="wide")

llm, embedder = load_models()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "file_name" not in st.session_state:
    st.session_state.file_name = None

with st.sidebar:
    st.header("📄 Document (offline)")
    uploaded_file = st.file_uploader("Upload PDF (offline)", type="pdf")

    if uploaded_file:
        file_name = uploaded_file.name
        st.write("File: ", file_name)
        if st.button("Process & Index"):            
            with st.spinner("Parsing PDF and creating vector store..."):
                text = text_from_pdf(uploaded_file)
                if not text.strip():
                    st.error("No text could be extracted from the PDF.")
                else:
                    chunks = chunk_text(text)
                    if not chunks:
                        st.error("No chunks created from the document.")
                    else:
                        emb = embedder.encode(chunks, show_progress_bar=True)
                        emb = np.array(emb).astype('float32')
                        index = build_faiss_index(emb)
                        save_vector_store(index, chunks, file_name)
                        st.success("Vector store created & saved (offline).")
                        st.session_state.vector_store = index
                        st.session_state.chunks = chunks
                        st.session_state.file_name = file_name

    if st.button("Clear Chat"):
        st.session_state.messages = []
        logger.info("Chat cleared by user.")

    if st.session_state.file_name:
        st.markdown(f"**Loaded:** {st.session_state.file_name}")

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


def construct_rag_prompt(chunks_for_prompt: list[str], user_question: str) -> str:
    context = "\n\n".join([f"--- Chunk {i+1} ---\n{c}" for i, c in enumerate(chunks_for_prompt)])
    prompt = (
        "You are an AI assistant. Answer the user question based ONLY on the context below.\n"
        "If the answer is not in the context, say 'I don't know based on this document'.\n"
        "Be concise and reference which chunk(s) your answer came from when possible.\n\n"
        f"Context:\n{context}\n\n"
        f"User Question: {user_question}\n\nAnswer:")
    return prompt


if user_input := st.chat_input("Ask about the uploaded PDF..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if not st.session_state.vector_store or not st.session_state.chunks:
            reply = "⚠️ Please upload and index a PDF first (use the sidebar)."
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            start = time.time()
            q_emb = embedder.encode([user_input])
            q_emb = np.array(q_emb).astype('float32')
            faiss.normalize_L2(q_emb)

            D, I = st.session_state.vector_store.search(q_emb, TOP_K)
            retrieved_chunks = [st.session_state.chunks[idx] for idx in I[0] if idx < len(st.session_state.chunks)]
            safe_chunks = [c[:1000] for c in retrieved_chunks]
            rag_prompt = construct_rag_prompt(safe_chunks, user_input)

            msg_box = st.empty()
            msg_box.markdown("🧠 **Generating answer...**")
            response_text = ""

            # Streamed generation using chunked output
            try:
                if hasattr(llm, 'stream_chat_completion'):
                    # hypothetical streaming API
                    for partial in llm.stream_chat_completion([
                        {"role": "user", "content": rag_prompt}
                    ], max_tokens=512, temperature=0.3, top_p=0.95):
                        response_text += partial
                        msg_box.markdown(response_text)
                elif hasattr(llm, 'chat_completion'):
                    res = llm.chat_completion([
                        {"role": "user", "content": rag_prompt}
                    ], max_tokens=512, temperature=0.3, top_p=0.95)
                    response_text = res['choices'][0]['message']['content']
                    msg_box.markdown(response_text)
                else:
                    # fallback generate
                    response_text = llm.generate(rag_prompt, max_tokens=512,  top_p=0.95)
                    msg_box.markdown(response_text)
            except Exception as e:
                logger.exception("LLM streaming failed")
                response_text = f"LLM error: {e}"
                msg_box.markdown(response_text)

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            elapsed = time.time() - start

            with st.expander("🕵️ Debug Info (retrieved chunks & scores)"):
                st.write(f"Time: {elapsed:.2f}s | Top K: {TOP_K}")
                for i, idx in enumerate(I[0]):
                    if idx < len(st.session_state.chunks):
                        st.markdown(f"**Rank {i+1} — idx {idx} — score {float(D[0][i]):.4f}**")
                        st.write(st.session_state.chunks[idx][:300])
                st.write("--- Prompt sent to LLM (truncated) ---")
                st.code(rag_prompt[:4000] + ("..." if len(rag_prompt) > 4000 else ""))

st.sidebar.caption("All models & vector stores are loaded/created offline. Streaming logs appear in the chat.")
