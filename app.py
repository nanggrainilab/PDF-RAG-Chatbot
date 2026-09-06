"""
PDF RAG Chatbot - WhatsApp-style multi-session UI
==================================================

Setiap "kontak" di sidebar adalah SATU sesi chat terpisah, masing-masing
punya knowledge base PDF sendiri (vector store terpisah). Semua data
disimpan in-memory (st.session_state) dan akan reset saat aplikasi
di-restart.

Jalankan dengan:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

# Pastikan folder project (yang berisi app.py + src/) selalu ada di sys.path,
# apa pun current working directory saat "streamlit run" dijalankan.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings, validate_settings
from pdf_utils import extract_text_from_pdfs, build_documents
from vectorstore import build_vectorstore, add_documents_to_vectorstore
from qa_chain import build_chain, ask_question

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title=settings.app_title,
    page_icon="asset\bubble-chat.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

AVATAR_COLORS = [
    "#F58B8B", "#7AC7A6", "#7AA7E9", "#E9C46A",
    "#B39DDB", "#F4A261", "#4FC3C0", "#E07A9B",
]

# --------------------------------------------------------------------------
# CSS - WhatsApp-ish look
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {padding-top: 1rem; padding-bottom: 0rem; max-width: 100%;}

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #f0f2f5;
            border-right: 1px solid #d1d7db;
        }
        section[data-testid="stSidebar"] .block-container {padding-top: 0.5rem;}

        .sidebar-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #111b21;
            padding: 0.3rem 0 0.6rem 0.2rem;
        }

        .contact-row {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 6px;
            border-radius: 10px;
            margin-bottom: 2px;
        }
        .contact-row:hover {background-color: #e9edef;}
        .contact-row.active {background-color: #d9fdd3;}

        .avatar-circle {
            min-width: 42px; height: 42px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            color: white; font-weight: 700; font-size: 1rem; flex-shrink: 0;
        }
        .contact-meta {overflow: hidden;}
        .contact-name {font-weight: 600; color: #111b21; font-size: 0.92rem;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
        .contact-sub {color: #667781; font-size: 0.78rem;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}

        /* Chat header */
        .chat-header {
            background-color: #075E54; color: white; padding: 14px 20px;
            border-radius: 10px 10px 0 0; display: flex; align-items: center; gap: 12px;
        }
        .chat-header .name {font-weight: 700; font-size: 1.05rem;}
        .chat-header .sub {font-size: 0.78rem; opacity: 0.85;}

        /* Chat body */
        .chat-body {
            background-color: #e5ddd5;
            background-image: linear-gradient(rgba(255,255,255,0.35), rgba(255,255,255,0.35));
            padding: 20px; min-height: 55vh; max-height: 62vh; overflow-y: auto;
            border-radius: 0 0 10px 10px;
        }

        .bubble-row {display: flex; margin-bottom: 10px;}
        .bubble-row.sent {justify-content: flex-end;}
        .bubble-row.received {justify-content: flex-start;}

        .bubble {
            max-width: 65%; padding: 8px 12px; border-radius: 10px;
            font-size: 0.92rem; line-height: 1.4; box-shadow: 0 1px 0.5px rgba(0,0,0,0.13);
        }
        .bubble.sent {background-color: #d9fdd3; color: #111b21; border-top-right-radius: 2px;}
        .bubble.received {background-color: #ffffff; color: #111b21; border-top-left-radius: 2px;}

        .bubble-time {font-size: 0.68rem; color: #667781; text-align: right; margin-top: 3px;}

        .file-card {
            display: flex; align-items: center; gap: 8px; background: #f0f2f5;
            border-radius: 8px; padding: 8px 10px; margin-bottom: 4px; font-size: 0.85rem;
        }

        .empty-state {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            height: 60vh; color: #667781; text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session-state init
# --------------------------------------------------------------------------
def init_state():
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "active_id" not in st.session_state:
        st.session_state.active_id = None
    if "show_new_chat" not in st.session_state:
        st.session_state.show_new_chat = len(st.session_state.sessions) == 0


init_state()


def new_avatar(name: str):
    idx = abs(hash(name)) % len(AVATAR_COLORS)
    letter = name.strip()[:1].upper() if name.strip() else "?"
    return letter, AVATAR_COLORS[idx]


def create_session(name: str, uploaded_files) -> str:
    session_id = str(uuid.uuid4())
    letter, color = new_avatar(name)

    session = {
        "id": session_id,
        "name": name.strip() or "Chat baru",
        "avatar_letter": letter,
        "avatar_color": color,
        "created_at": datetime.now().strftime("%H:%M"),
        "messages": [],
        "vectorstore": None,
        "chain": None,
        "pdf_names": [],
        "history_pairs": [],
    }
    st.session_state.sessions[session_id] = session

    if uploaded_files:
        process_pdfs_for_session(session_id, uploaded_files)

    st.session_state.active_id = session_id
    st.session_state.show_new_chat = False
    return session_id


def process_pdfs_for_session(session_id: str, uploaded_files) -> tuple[int, int]:
    """Ekstrak, chunk, dan masukkan PDF ke vector store sesi. Return (berhasil, gagal)."""
    session = st.session_state.sessions[session_id]

    extracted = extract_text_from_pdfs(uploaded_files)
    failed = len(uploaded_files) - len(extracted)

    if not extracted:
        return 0, failed

    documents = build_documents(extracted)
    session["vectorstore"] = add_documents_to_vectorstore(session["vectorstore"], documents)
    session["chain"] = build_chain(session["vectorstore"])

    new_names = [f.filename for f in extracted]
    session["pdf_names"].extend(new_names)

    session["messages"].append({
        "role": "system",
        "content": f"📎 {len(new_names)} dokumen ditambahkan: {', '.join(new_names)}",
        "time": datetime.now().strftime("%H:%M"),
    })

    return len(extracted), failed


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<div class='sidebar-title'> {settings.app_title}</div>", unsafe_allow_html=True)

    search_query = st.text_input("Cari", placeholder="Cari chat...", label_visibility="collapsed")

    if st.button("Chat baru", use_container_width=True):
        st.session_state.show_new_chat = True

    st.divider()

    sessions = list(st.session_state.sessions.values())
    if search_query:
        sessions = [s for s in sessions if search_query.lower() in s["name"].lower()]

    for s in sessions:
        last_msg = s["messages"][-1]["content"] if s["messages"] else "Belum ada pesan"
        last_msg = (last_msg[:32] + "…") if len(last_msg) > 32 else last_msg
        is_active = s["id"] == st.session_state.active_id

        row = st.container()
        with row:
            cols = st.columns([1, 5])
            with cols[0]:
                st.markdown(
                    f"<div class='avatar-circle' style='background:{s['avatar_color']}'>"
                    f"{s['avatar_letter']}</div>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button(
                    f"{s['name']}\n{last_msg}",
                    key=f"select_{s['id']}",
                    use_container_width=True,
                ):
                    st.session_state.active_id = s["id"]
                    st.session_state.show_new_chat = False
                    st.rerun()

    if not sessions and not st.session_state.show_new_chat:
        st.caption("Belum ada chat. Klik 'Chat baru' untuk mulai.")

# --------------------------------------------------------------------------
# New chat form (modal-ish, ditampilkan di area utama)
# --------------------------------------------------------------------------
config_errors = validate_settings()

if st.session_state.show_new_chat:
    st.subheader("Buat sesi chat baru")

    if config_errors:
        for e in config_errors:
            st.error(e)

    with st.form("new_chat_form", clear_on_submit=True):
        chat_name = st.text_input("Nama chat / materi", placeholder="Contoh: Laporan Keuangan Q3")
        pdf_files = st.file_uploader(
            "Upload satu atau lebih PDF (akan digabung jadi satu knowledge base)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Buat chat", use_container_width=True)

    if submitted:
        if not chat_name.strip():
            st.warning("Nama chat tidak boleh kosong.")
        elif config_errors:
            st.warning("Lengkapi konfigurasi GOOGLE_API_KEY dulu di file .env sebelum membuat chat.")
        else:
            with st.spinner("Memproses PDF & membangun index..."):
                sid = create_session(chat_name, pdf_files or [])
            st.rerun()

    if st.session_state.sessions and st.button("Batal"):
        st.session_state.show_new_chat = False
        st.rerun()

    st.stop()

# --------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------
active_id = st.session_state.active_id
active = st.session_state.sessions.get(active_id) if active_id else None

if not active:
    st.markdown(
        """
        <div class="empty-state">
            <h2>PDF Chat</h2>
            <p>Pilih chat di sidebar, atau buat chat baru untuk mulai bertanya
            seputar dokumen PDF-mu.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ---- Header ----
pdf_count = len(active["pdf_names"])
st.markdown(
    f"""
    <div class="chat-header">
        <div class="avatar-circle" style="background:{active['avatar_color']}">{active['avatar_letter']}</div>
        <div>
            <div class="name">{active['name']}</div>
            <div class="sub">{pdf_count} dokumen · {'siap dijawab' if pdf_count else 'belum ada PDF'}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Message list ----
body_html = ["<div class='chat-body'>"]
for msg in active["messages"]:
    role = msg["role"]
    if role == "system":
        body_html.append(
            f"<div class='bubble-row received'><div class='bubble received'>"
            f"{msg['content']}<div class='bubble-time'>{msg['time']}</div></div></div>"
        )
    else:
        cls = "sent" if role == "user" else "received"
        content = msg["content"].replace("\n", "<br>")
        body_html.append(
            f"<div class='bubble-row {cls}'><div class='bubble {cls}'>"
            f"{content}<div class='bubble-time'>{msg['time']}</div></div></div>"
        )
if not active["messages"]:
    body_html.append(
        "<div class='bubble-row received'><div class='bubble received'>"
        "Halo! Upload PDF lalu tanya apa saja seputar isinya"
        "<div class='bubble-time'></div></div></div>"
    )
body_html.append("</div>")
st.markdown("".join(body_html), unsafe_allow_html=True)

# ---- Add more PDFs (attach) ----
with st.expander("📎 Tambah dokumen PDF ke chat ini"):
    more_files = st.file_uploader(
        "Upload PDF tambahan (digabung ke knowledge base yang sudah ada)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"upload_{active_id}",
    )
    if more_files and st.button("Proses & tambahkan", key=f"process_{active_id}"):
        with st.spinner("Memproses PDF..."):
            ok, failed = process_pdfs_for_session(active_id, more_files)
        if failed:
            st.warning(f"{failed} file gagal diproses (mungkin PDF hasil scan tanpa teks).")
        st.rerun()

# ---- Input box ----
if config_errors:
    for e in config_errors:
        st.error(e)

with st.form(f"chat_input_{active_id}", clear_on_submit=True):
    cols = st.columns([6, 1])
    with cols[0]:
        user_input = st.text_input(
            "Pesan", placeholder="Ketik pertanyaan tentang dokumen...",
            label_visibility="collapsed",
        )
    with cols[1]:
        send = st.form_submit_button("Kirim..", use_container_width=True)

if send and user_input.strip():
    now = datetime.now().strftime("%H:%M")
    active["messages"].append({"role": "user", "content": user_input.strip(), "time": now})

    if active["chain"] is None:
        active["messages"].append({
            "role": "assistant",
            "content": "Belum ada PDF di chat ini. Silakan upload dokumen dulu lewat panel "
                       "'Tambah dokumen PDF' di atas ya.",
            "time": datetime.now().strftime("%H:%M"),
        })
    else:
        try:
            with st.spinner("Sedang mencari jawaban di dokumen..."):
                result = ask_question(
                    active["chain"], user_input.strip(), active["history_pairs"]
                )
            answer = result["answer"]
            active["history_pairs"].append((user_input.strip(), answer))
            active["messages"].append({
                "role": "assistant", "content": answer,
                "time": datetime.now().strftime("%H:%M"),
            })
        except Exception as exc:  # noqa: BLE001
            active["messages"].append({
                "role": "assistant",
                "content": f"Terjadi error saat memanggil Gemini: {exc}",
                "time": datetime.now().strftime("%H:%M"),
            })

    st.rerun()