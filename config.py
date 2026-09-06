"""
Konfigurasi aplikasi.
Semua parameter penting diambil dari environment variable (.env),
supaya gampang diubah tanpa mengubah kode.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load variabel dari file .env yang ada di root project
load_dotenv()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- Google Gemini ---
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_temperature: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))

    # --- Embedding model (Sentence Transformers, jalan lokal / gratis) ---
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Text splitting ---
    chunk_size: int = _get_int("CHUNK_SIZE", 1000)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 150)

    # --- Retrieval ---
    top_k: int = _get_int("TOP_K", 4)

    # --- App ---
    app_title: str = os.getenv("APP_TITLE", "PDF Chat")


settings = Settings()


def validate_settings() -> list[str]:
    """Mengembalikan daftar pesan error jika ada konfigurasi wajib yang kosong."""
    errors = []
    if not settings.google_api_key:
        errors.append(
            "GOOGLE_API_KEY belum diisi. Buat file .env (lihat .env.example) "
            "dan isi API key dari Google AI Studio."
        )
    return errors
