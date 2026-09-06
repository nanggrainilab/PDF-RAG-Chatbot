# PDF RAG Chatbot — Multi-Session, WhatsApp-style UI

Chatbot RAG (Retrieval Augmented Generation) berbasis PDF dengan tampilan
mirip WhatsApp: sidebar berisi daftar **sesi chat**, di mana **tiap sesi
punya knowledge base PDF sendiri-sendiri** (vector store terpisah, tidak
tercampur antar sesi).

Dibangun dengan **LangChain + Google Gemini (`gemini-2.5-flash`) + FAISS +
Sentence Transformers + Streamlit**.

## Fitur

- **Multi-session chat**: setiap "kontak" di sidebar = 1 topik/sesi dengan PDF sendiri.
- **Multi-PDF per sesi**: upload beberapa PDF sekaligus, otomatis digabung jadi satu knowledge base.
- **Tambah PDF belakangan**: bisa upload PDF tambahan kapan saja lewat panel "📎 Tambah dokumen PDF".
- **Ekstraksi PDF** dengan `pypdf`.
- **Chunking** dengan `RecursiveCharacterTextSplitter` (ukuran & overlap bisa diatur lewat `.env`).
- **Embedding** lokal & gratis dengan Sentence Transformers (`all-MiniLM-L6-v2` default).
- **Vector store** FAISS in-memory per sesi.
- **Jawaban** dari Google Gemini via `ChatGoogleGenerativeAI`, dengan riwayat percakapan (multi-turn) per sesi.
- **Retry otomatis** kalau panggilan ke Gemini gagal sementara.
- **Semua data in-memory** — akan reset kalau aplikasi Streamlit di-restart (sesuai desain, tidak ada database).

## Struktur Project

```
pdf-rag-chatbot/
├── app.py                 # Entry point Streamlit (UI WhatsApp-style + orchestrasi)
│── config.py          # Load & validasi konfigurasi dari .env
├── pdf_utils.py        # Ekstraksi teks PDF + chunking
├── vectorstore.py      # Embedding (Sentence Transformers) + FAISS
└── qa_chain.py         # Conversational RAG chain (Gemini)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Prasyarat

- Python 3.10 atau lebih baru
- API key Gemini gratis dari **Google AI Studio**: https://aistudio.google.com/app/apikey

## Cara Menjalankan (Terminal)

**1. Clone / masuk ke folder project**

```bash
cd pdf-rag-chatbot
```

**2. Buat virtual environment (disarankan)**

```bash
python -m venv venv

# Aktifkan:
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> Catatan: instalasi `sentence-transformers` akan otomatis menarik PyTorch,
> jadi proses ini bisa memakan waktu beberapa menit dan ruang disk (~1-2GB)
> pada instalasi pertama.

**4. Siapkan file konfigurasi**

```bash
cp .env.example .env      # macOS/Linux
copy .env.example .env    # Windows
```

Lalu buka `.env` dan isi:

```
GOOGLE_API_KEY=api_key_kamu_disini
```

**5. Jalankan aplikasi**

```bash
streamlit run app.py
```

Buka browser ke `http://localhost:8501` (biasanya terbuka otomatis).

## Cara Pakai di UI

1. Klik **"➕ Chat baru"** di sidebar.
2. Isi nama chat (misal: `Laporan Keuangan Q3`) dan upload satu atau
   beberapa PDF sekaligus.
3. Klik **"Buat chat"** — aplikasi akan mengekstrak teks, memecah jadi
   chunk, membangun embedding, dan menyimpannya ke FAISS index khusus
   sesi ini.
4. Ketik pertanyaan di kolom bawah dan klik **"Kirim ➤"**.
5. Untuk topik/dokumen lain, buat sesi baru lagi — knowledge base-nya
   akan terpisah total dari sesi sebelumnya.
6. Ingin menambah PDF ke sesi yang sedang aktif? Buka panel
   **"📎 Tambah dokumen PDF ke chat ini"**.

## Konfigurasi (`.env`)

| Variabel | Default | Keterangan |
|---|---|---|
| `GOOGLE_API_KEY` | *(wajib diisi)* | API key dari Google AI Studio |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model Gemini yang dipakai |
| `GEMINI_TEMPERATURE` | `0.2` | Kreativitas jawaban (0 = paling faktual) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model embedding lokal |
| `CHUNK_SIZE` | `1000` | Ukuran tiap potongan teks (karakter) |
| `CHUNK_OVERLAP` | `150` | Overlap antar potongan |
| `TOP_K` | `4` | Jumlah chunk relevan yang diambil per pertanyaan |
| `APP_TITLE` | `PDF Chat` | Judul aplikasi di sidebar |

## Batasan yang Perlu Diketahui

- **Tidak persisten**: semua sesi chat, riwayat, dan index PDF hilang
  saat proses `streamlit run` dihentikan/di-restart (sesuai kebutuhan
  awal — tidak pakai database).
- **PDF hasil scan (gambar)** tanpa layer teks tidak akan terbaca (perlu
  OCR tambahan, di luar cakupan versi ini).
- Karena vector store in-memory, aplikasi ini cocok untuk pemakaian
  personal/demo, bukan multi-user production dengan banyak sesi berat
  sekaligus (butuh RAM lebih besar seiring banyaknya sesi + PDF aktif).

## Pengembangan Lanjutan (opsional)

Kalau nanti butuh persistensi (chat & index tidak hilang saat restart),
tinggal tambahkan:
- SQLite / file JSON untuk menyimpan metadata sesi & riwayat chat.
- `vectorstore.save_local()` / `FAISS.load_local()` untuk menyimpan index ke disk per sesi.
