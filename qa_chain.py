"""
Conversational RAG:
FAISS retriever + Gemini LLM + manual chat history.
"""

from __future__ import annotations

import time
from typing import List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from config import settings


# ============================================================
# PROMPT
# ============================================================

QA_PROMPT_TEMPLATE = """Kamu adalah asisten yang menjawab pertanyaan HANYA berdasarkan
konteks dokumen PDF berikut.

ATURAN:
1. Jawab hanya berdasarkan informasi yang terdapat dalam konteks dokumen.
2. Jika jawabannya tidak ditemukan dalam konteks, katakan dengan jujur:
   "Informasi tersebut tidak ditemukan dalam dokumen."
3. Jangan mengarang atau menggunakan informasi dari luar dokumen.
4. Gunakan bahasa yang sama dengan bahasa pertanyaan pengguna.
5. Jika pertanyaan membutuhkan beberapa informasi dari dokumen, gabungkan
   informasi tersebut dengan jelas.
6. Jawab secara ringkas tetapi tetap informatif.

KONTEKS DOKUMEN:
{context}

RIWAYAT PERCAKAPAN:
{chat_history}

PERTANYAAN:
{question}

JAWABAN:"""


QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=[
        "context",
        "chat_history",
        "question",
    ],
)


# ============================================================
# LLM
# ============================================================

def get_llm() -> ChatGoogleGenerativeAI:
    """
    Membuat instance Gemini LLM.
    """

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.gemini_temperature,
    )


# ============================================================
# DOCUMENT FORMATTER
# ============================================================

def format_docs(docs: List[Document]) -> str:
    """
    Menggabungkan isi dokumen hasil retrieval menjadi satu string.
    """

    if not docs:
        return "Tidak ada konteks dokumen yang ditemukan."

    return "\n\n---\n\n".join(
        doc.page_content
        for doc in docs
        if doc.page_content
    )


# ============================================================
# CHAT HISTORY FORMATTER
# ============================================================

def format_chat_history(
    chat_history: List[Tuple[str, str]]
) -> str:
    """
    Mengubah history percakapan menjadi teks.

    Format input:
        [
            ("pertanyaan user", "jawaban assistant"),
            ("pertanyaan user", "jawaban assistant"),
        ]
    """

    if not chat_history:
        return "Belum ada riwayat percakapan."

    formatted_history = []

    for human, ai in chat_history:
        formatted_history.append(
            f"User: {human}"
        )
        formatted_history.append(
            f"Assistant: {ai}"
        )

    return "\n".join(formatted_history)


# ============================================================
# BUILD RAG CHAIN
# ============================================================

def build_chain(vectorstore: FAISS):
    """
    Membuat RAG chain menggunakan LCEL.

    Chain:
        User Question
              ↓
        FAISS Retriever
              ↓
        Relevant Documents
              ↓
        Prompt
              ↓
        Gemini
              ↓
        Answer
    """

    # --------------------------------------------------------
    # Retriever
    # --------------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": settings.top_k
        }
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = get_llm()

    # --------------------------------------------------------
    # Chain
    # --------------------------------------------------------

    def retrieve_documents(inputs):
        question = inputs["question"]

        documents = retriever.invoke(question)

        return format_docs(documents)

    def prepare_chat_history(inputs):
        history = inputs.get(
            "chat_history",
            []
        )

        return format_chat_history(history)

    chain = (
        {
            "context": retrieve_documents,
            "chat_history": prepare_chat_history,
            "question": lambda inputs: inputs["question"],
        }
        | QA_PROMPT
        | llm
    )

    # Simpan retriever pada chain agar ask_question()
    # bisa mengembalikan source documents.
    chain._retriever = retriever

    return chain


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(
    chain,
    question: str,
    chat_history: List[Tuple[str, str]],
    max_retries: int = 2,
) -> dict:
    """
    Menjalankan pertanyaan melalui RAG chain.

    Parameters
    ----------
    chain:
        Chain yang dibuat oleh build_chain()

    question:
        Pertanyaan dari user.

    chat_history:
        List tuple:
        [
            ("pertanyaan", "jawaban"),
            ...
        ]

    max_retries:
        Jumlah retry jika terjadi error sementara.
    """

    last_error = None

    # --------------------------------------------------------
    # Retry loop
    # --------------------------------------------------------

    for attempt in range(max_retries + 1):

        try:

            # ------------------------------------------------
            # Retrieve source documents
            # ------------------------------------------------

            source_documents = []

            retriever = getattr(
                chain,
                "_retriever",
                None
            )

            if retriever is not None:
                source_documents = retriever.invoke(
                    question
                )

            # ------------------------------------------------
            # Invoke chain
            # ------------------------------------------------

            result = chain.invoke(
                {
                    "question": question,
                    "chat_history": chat_history,
                }
            )

            # ------------------------------------------------
            # Extract answer
            # ------------------------------------------------

            if hasattr(result, "content"):
                answer = result.content

            elif isinstance(result, dict):
                answer = result.get(
                    "answer",
                    result.get(
                        "result",
                        str(result)
                    )
                )

            else:
                answer = str(result)

            # ------------------------------------------------
            # Return result
            # ------------------------------------------------

            return {
                "answer": answer,
                "source_documents": source_documents,
            }

        # ----------------------------------------------------
        # Error handling
        # ----------------------------------------------------

        except Exception as exc:

            last_error = exc

            if attempt < max_retries:

                # Retry delay:
                # attempt 0 -> 1.5 sec
                # attempt 1 -> 3 sec
                time.sleep(
                    1.5 * (attempt + 1)
                )

                continue

            break

    # --------------------------------------------------------
    # All retries failed
    # --------------------------------------------------------

    raise RuntimeError(
        "Gagal mendapatkan jawaban dari Gemini "
        f"setelah {max_retries + 1} percobaan: "
        f"{last_error}"
    )
