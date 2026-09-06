"""
Membangun embeddings (Sentence Transformers) dan vector store (FAISS)
untuk satu sesi chat. Vector store disimpan in-memory saja (per session_state),
sesuai kebutuhan: reset kalau aplikasi di-restart.
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


@st.cache_resource(show_spinner=False)
def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Model embedding di-cache secara global (bukan per session) karena:
    - Modelnya sama untuk semua sesi chat.
    - Loading model Sentence Transformers cukup berat, tidak perlu diulang
      setiap kali user membuat sesi baru.
    """
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_vectorstore(documents: List[Document]) -> FAISS:
    """Membuat FAISS vector store baru dari daftar Document."""
    embeddings = get_embeddings_model()
    return FAISS.from_documents(documents, embeddings)


def add_documents_to_vectorstore(
    vectorstore: Optional[FAISS], documents: List[Document]
) -> FAISS:
    """
    Menambahkan dokumen baru ke vector store yang sudah ada (misalnya saat
    user upload PDF tambahan di tengah percakapan). Jika belum ada vector
    store, buat baru.
    """
    if vectorstore is None:
        return build_vectorstore(documents)

    vectorstore.add_documents(documents)
    return vectorstore
