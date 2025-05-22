import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
from uuid import uuid4
from dotenv import load_dotenv
import os
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─── CONFIG & CLIENT SETUP ─────────────────────────────────────────────

load_dotenv()               # loads OPENAI_API_KEY from .env  
client = OpenAI()           # uses that key  

# ─── FAISS INDEX ───────────────────────────────────────────────────────

dimension = 1536
index = faiss.IndexFlatL2(dimension)

# ─── STORAGE FOR CHUNKS & PAGES ────────────────────────────────────────

chunks = []
source_pages = []

# ─── EMBEDDING HELPER ──────────────────────────────────────────────────

def get_embedding(text, model="text-embedding-3-small"):
    resp = client.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding

# ─── STREAMLIT UI ──────────────────────────────────────────────────────

st.set_page_config(page_title="Protocol Pilot AI", layout="centered")
st.title("📋 Protocol Pilot AI")
st.caption("Upload your protocol and ask questions — no external DB needed.")

# Option 1: add a clear instruction above the uploader
st.markdown("**👇 Drop your protocol here to get started**")
pdf = st.file_uploader("", type=["pdf"])

if pdf is not None:
    # Reset previous data
    chunks.clear()
    source_pages.clear()

    # 1) Read PDF & split into chunks, tracking page numbers
    st.info("Reading PDF & splitting into chunks…")
    with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            page_chunks = splitter.split_text(text)
            chunks.extend(page_chunks)
            source_pages.extend([page_num] * len(page_chunks))

    # 2) Batch embed all chunks in one API call
    st.info("Generating embeddings for all chunks in one go…")
    batch_resp = client.embeddings.create(input=chunks, model="text-embedding-3-small")
    vectors = np.array([d.embedding for d in batch_resp.data], dtype="float32")
    index.add(vectors)
    st.success(f"✅ Embedded {len(chunks)} chunks in one batch!")

    # 3) Ask a question
    question = st.text_input("🧠 Ask your protocol a question")
    if question:
        q_vec = np.array([get_embedding(question)], dtype="float32")
        D, I = index.search(q_vec, k=5)
        matches = [(chunks[idx], source_pages[idx]) for idx in I[0]]

        # Build prompt with page tags
        context = "\n\n".join(f"[Page {pg}]\n{chunk}" for chunk, pg in matches)
        prompt = (
            "Use the context below to answer the clinical protocol question.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )

        chat_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert clinical research assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = chat_resp.choices[0].message.content

        # 4) Display answer and source references
        st.markdown("### ✅ Answer")
        st.write(answer)

        with st.expander("🔍 Source references"):
            for chunk, pg in matches:
                st.markdown(f"**Page {pg}**")
                st.write(chunk)
