# groq_streamlit_chat.py
"""
Streamlit chat application that
• stores multi-chat "sessions" on disk               (sessions/ folder)
• accepts PDFs / images / text / DOCX as context
• streams answers from Groq’s chat-completion API

Cloud-friendly tweaks:
──────────────────────
1. Page config is set *first*, before any Streamlit calls.
2. Uses st.cache_resource for the heavy EasyOCR model (fast cold-start on Cloud).
3. Looks for the GROQ key in either env-vars *or* st.secrets (Streamlit Cloud’s UI).
4. No Streamlit calls during import-time try/except (safer on Cloud).
"""

# ───────────────────────── Imports ───────────────────────── #
from __future__ import annotations

import json, os, uuid, time
from datetime import datetime
from pathlib import Path

import streamlit as st                              # 1️⃣ Page config first!
st.set_page_config(page_title="Groq Chat", page_icon="💬", layout="centered")

import fitz                                          # PyMuPDF
import numpy as np
import easyocr
from PIL import Image
from dotenv import load_dotenv
from groq import Groq

# Optional DOCX support
try:
    import docx                                      # python-docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ─────────────────────── Configuration ───────────────────── #
load_dotenv()

# Streamlit Cloud: pick up the key from either env-var or secrets
GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY")                       # local .env
    or st.secrets.get("GROQ_API_KEY", "")           # cloud secret
)

AVAILABLE_MODELS = [
    "llama3-70b-8192",
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen-qwq-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]
DEFAULT_MODEL = "llama3-70b-8192"
SESSIONS_DIR  = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

# ──────────────────────  OCR Reader  ─────────────────────── #
@st.cache_resource(show_spinner="🔍 Loading OCR…")
def get_easy_reader():
    # add/remove languages here if needed
    return easyocr.Reader(["en"])

easy_reader = get_easy_reader()

# ────────────────── Session helpers ────────────────── #
def load_sessions() -> dict[str, dict]:
    sessions: dict[str, dict] = {}
    for fn in SESSIONS_DIR.glob("*.json"):
        try:
            sessions[fn.stem] = json.loads(fn.read_text())
        except Exception as e:
            st.error(f"Could not load session {fn}: {e}")
    return sessions


def save_session(sess: dict) -> None:
    (SESSIONS_DIR / f"{sess['id']}.json").write_text(
        json.dumps(sess, ensure_ascii=False, indent=2)
    )


def create_session(first_msg: str | None = None) -> dict:
    sid   = str(uuid.uuid4())
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = (first_msg[:30] + "…") if first_msg and len(first_msg) > 30 else (first_msg or "New session")

    sess = {
        "id": sid,
        "name": label,
        "created": now,
        "model": DEFAULT_MODEL,
        "messages": [],                         # list[{"role": "...", "content": "..."}]
        "context": {
            x: {"text": "", "source": None} for x in ("pdf", "image", "txt", "docx")
        },
    }
    save_session(sess)
    return sess


def delete_session(sid: str) -> None:
    st.session_state.sessions.pop(sid, None)
    try:
        (SESSIONS_DIR / f"{sid}.json").unlink(missing_ok=True)
    except Exception:
        pass

# ─────────────────────── File helpers ───────────────────── #
def pdf_to_text(uploaded) -> str:
    try:
        buf = uploaded.read()
        uploaded.seek(0)
        doc = fitz.open(stream=buf, filetype="pdf")
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"PDF error: {e}")
        return ""


def image_to_text_easyocr(uploaded) -> str:
    try:
        img_bytes = uploaded.read()
        uploaded.seek(0)
        result = easy_reader.readtext(img_bytes, detail=0)
        return "\n".join(result)
    except Exception as e:
        st.error(f"Image OCR error: {e}")
        return ""


def txt_file_to_text(uploaded) -> str:
    try:
        data = uploaded.read().decode("utf-8", errors="ignore")
        uploaded.seek(0)
        return data
    except Exception as e:
        st.error(f"Text-file error: {e}")
        return ""


def docx_to_text(uploaded) -> str:
    if not DOCX_AVAILABLE:
        return ""
    try:
        document = docx.Document(uploaded)
        uploaded.seek(0)
        return "\n".join(p.text for p in document.paragraphs)
    except Exception as e:
        st.error(f"DOCX error: {e}")
        return ""

# ──────────────────── Groq client ──────────────────── #
client = Groq(api_key=GROQ_API_KEY)


def chat_completion_stream(history: list[dict], ctx: dict, model_name: str):
    system_prompt = (
        "You are a helpful, accurate, friendly AI assistant.\n"
        "Use the supplied *context* if it answers the question; "
        "otherwise rely on your own knowledge. "
        "If you truly don't know, say so."
    )

    # last two full turns = 4 msgs
    conversation_history = history[-4:]

    # build context string
    context_parts = [
        f"context from {typ}:\n{item['text']}"
        for typ, item in ctx.items()
        if item["text"]
    ]

    prompt = system_prompt
    if context_parts:
        prompt += "\n\nHere is extra context:\n" + "\n".join(context_parts) + "\n"

    msgs = [{"role": "system", "content": prompt}, *conversation_history]

    stream = client.chat.completions.create(
        model=model_name,
        messages=msgs,
        temperature=0.7,
        max_tokens=2048,
        stream=True,
    )

    answer = ""
    for chunk in stream:
        answer += chunk.choices[0].delta.content or ""
        yield answer  # incremental


# ═══════════════════════ UI ═══════════════════════ #
# One-time session-state initialisation
if "sessions"           not in st.session_state: st.session_state.sessions          = load_sessions()
if "current_session_id" not in st.session_state: st.session_state.current_session_id = None

# ─────────────── Sidebar ─────────────── #
with st.sidebar:
    st.header("Chat Sessions")
    if st.button("➕  New session"):
        s = create_session()
        st.session_state.sessions[s["id"]] = s
        st.session_state.current_session_id = s["id"]
        st.rerun()

    # Model selector (when a session is active)
    if st.session_state.current_session_id:
        sess = st.session_state.sessions[st.session_state.current_session_id]
        current_model = sess.get("model", DEFAULT_MODEL)
        new_model = st.selectbox(
            "Model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(current_model),
        )
        if new_model != current_model:
            sess["model"] = new_model
            save_session(sess)
            st.experimental_rerun()

    # Upload files
    exts = ["pdf", "png", "jpg", "jpeg", "txt"] + (["docx"] if DOCX_AVAILABLE else [])
    upl = st.file_uploader("Upload context file", type=exts)

    if upl and st.session_state.current_session_id:
        sess = st.session_state.sessions[st.session_state.current_session_id]

        if upl.type.endswith("pdf"):
            txt, src = pdf_to_text(upl), f"PDF: {upl.name}"
            key = "pdf"
        elif upl.type.startswith("image/"):
            txt, src = image_to_text_easyocr(upl), f"Image: {upl.name}"
            key = "image"
        elif upl.name.endswith(".docx"):
            txt, src = docx_to_text(upl), f"DOCX: {upl.name}"
            key = "docx"
        else:
            txt, src = txt_file_to_text(upl), f"Text: {upl.name}"
            key = "txt"

        if txt:
            sess["context"][key] = {"text": txt, "source": src}
            save_session(sess)
            st.success(f"Added context from {src}")

    # Existing sessions list
    st.markdown("### Existing")
    for sid, s in sorted(
        st.session_state.sessions.items(),
        key=lambda t: t[1]["created"],
        reverse=True,
    ):
        cols = st.columns([4, 1])
        if cols[0].button(s["name"], key=f"sel_{sid}"):
            st.session_state.current_session_id = sid
            st.rerun()
        if cols[1].button("🗑️", key=f"del_{sid}"):
            delete_session(sid)
            st.rerun()

    # Missing-key notice
    if not GROQ_API_KEY:
        st.warning("Add **GROQ_API_KEY** in the app’s secrets to chat with Groq.")

    if not DOCX_AVAILABLE:
        st.info("📝 DOCX support not installed (`pip install python-docx`).")

# ─────────────── Main pane ─────────────── #
st.title("💬 AI Assistant")

if not st.session_state.current_session_id:
    st.info("Create a session to start chatting.")
    st.stop()

sess = st.session_state.sessions[st.session_state.current_session_id]
st.subheader(f"Session: {sess['name']}")
st.caption(f"🤖 Model: {sess.get('model', DEFAULT_MODEL)}")

# show context sources
sources = [
    ctx["source"] for ctx in sess["context"].values() if ctx["source"]
]
if sources:
    st.caption("🔗 Context from: " + ", ".join(sources))

# clear button
if st.button("Clear chat & context"):
    sess["messages"].clear()
    for item in sess["context"].values():
        item.update(text="", source=None)
    save_session(sess)
    st.experimental_rerun()

# render chat history
for m in sess["messages"]:
    if m["role"] != "system":
        with st.chat_message(m["role"]):
            st.write(m["content"])

# chat input
prompt = st.chat_input("Ask me something…")
if prompt:
    if not sess["messages"]:
        sess["name"] = (prompt[:30] + "…") if len(prompt) > 30 else prompt

    sess["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        container = st.empty()
        answer = ""
        try:
            for partial in chat_completion_stream(
                sess["messages"],
                sess["context"],
                sess.get("model", DEFAULT_MODEL),
            ):
                answer = partial
                container.markdown(answer + "▌")
            container.markdown(answer)
        except Exception as e:
            answer = f"⚠️ {e}"
            container.error(answer)

    sess["messages"].append({"role": "assistant", "content": answer})
    save_session(sess)
