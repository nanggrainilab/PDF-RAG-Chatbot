"""
Utility untuk ekstraksi teks dari PDF dan pemecahan teks menjadi chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import settings


@dataclass
class ExtractedFile:
    filename: str
    num_pages: int
    text: str


def extract_text_from_pdfs(uploaded_files) -> List[ExtractedFile]:
    """
    Menerima list file (dari st.file_uploader) dan mengembalikan teks per file.
    File yang gagal dibaca akan dilewati (skip), tidak menghentikan proses lain.
    """
    results: List[ExtractedFile] = []

    for file in uploaded_files:
        try:
            reader = PdfReader(file)
            pages_text = []
            for page in reader.pages:
                extracted = page.extract_text() or ""
                pages_text.append(extracted)
            full_text = "\n".join(pages_text).strip()

            if not full_text:
                # PDF kemungkinan hasil scan/gambar tanpa layer teks
                continue

            results.append(
                ExtractedFile(
                    filename=getattr(file, "name", "document.pdf"),
                    num_pages=len(reader.pages),
                    text=full_text,
                )
            )
        except Exception as exc:  # noqa: BLE001
            # Jangan sampai satu file corrupt menghentikan semua proses upload
            print(f"[pdf_utils] Gagal membaca {getattr(file, 'name', '?')}: {exc}")
            continue

    return results


def build_documents(extracted_files: List[ExtractedFile]) -> List[Document]:
    """
    Memecah teks tiap file menjadi chunk dan membungkusnya sebagai
    LangChain Document, lengkap dengan metadata nama file & nomor chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: List[Document] = []
    for f in extracted_files:
        chunks = splitter.split_text(f.text)
        for idx, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": f.filename, "chunk": idx},
                )
            )
    return documents
