# groq_streamlit_cloud.py
"""
Streamlit chat application that:
• stores multi-chat "sessions" using Streamlit's native session state
• accepts PDFs / images / text files / docx and makes their text available as context
• streams answers from Groq's chat-completion API
• optimized for Streamlit Cloud deployment
"""

import base64
import io
import json
import os
import time
import uuid
import hashlib
from datetime import datetime

import pymupdf                             # PyMuPDF
import numpy as np
import streamlit as st
from PIL import Image
from groq import Groq                      # pip install groq

# Try to import optional dependencies with graceful fallbacks
try:
    import easyocr                             # OCR
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    
try:
    import docx                               # python-docx for DOCX files
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# ────────────────────────  CONFIG  ──────────────────────── #
# Streamlit secrets management for API keys
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
# Password for app access
APP_PASSWORD = st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", "password123"))

AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "deepseek-r1-distill-llama-70b",
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen-qwq-32b",
    "meta-llama/llama-4-maverick-17b-128e-instruct"
]
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_TOKENS = 8192
MIN_TOKENS = 256
MAX_TOKENS = 36000

# ────────────────────────  OCR READER  ─────────────────────── #
@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR reader with caching to improve performance"""
    if EASYOCR_AVAILABLE:
        return easyocr.Reader(['en'], gpu=False)
    else: 
        return None


# ──────────────────  SESSION MANAGEMENT  ──────────────── #
def load_sessions() -> dict[str, dict]:
    """Load sessions from session state or initialize empty sessions dictionary"""
    if "all_sessions" not in st.session_state:
        st.session_state.all_sessions = {}
    return st.session_state.all_sessions


def save_session(session: dict) -> None:
    """Save a session to session state"""
    sid = session["id"]
    st.session_state.all_sessions[sid] = session


def create_session(first_msg: str | None = None) -> dict:
    """Create a brand-new empty session and return it."""
    sid   = str(uuid.uuid4())
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = (first_msg[:30] + "…") if first_msg and len(first_msg) > 30 else (first_msg or "New session")
    session = {
        "id"      : sid,
        "name"    : label,
        "created" : now,
        "model"   : DEFAULT_MODEL,
        "messages": [],                      # list[{"role": "...", "content": "..."}]
        "context" : {
            "pdf": {"text": "", "source": None},
            "image": {"text": "", "source": None},
            "txt": {"text": "", "source": None},
            "docx": {"text": "", "source": None}
        },
    }
    save_session(session)
    return session


def delete_session(sid: str) -> None:
    """Remove a session from session state"""
    st.session_state.all_sessions.pop(sid, None)


def export_sessions() -> str:
    """Export all sessions as a downloadable JSON file"""
    return json.dumps(st.session_state.all_sessions, ensure_ascii=False, indent=2)


def import_sessions(json_data: str) -> None:
    """Import sessions from JSON data"""
    try:
        sessions = json.loads(json_data)
        for sid, session in sessions.items():
            st.session_state.all_sessions[sid] = session
        return True
    except Exception as e:
        st.error(f"Failed to import sessions: {e}")
        return False


# ────────────────────────  FILE HELPERS  ─────────────────── #
@st.cache_data
def pdf_to_text(uploaded) -> str:
    """Extract text from PDF with caching for performance"""
    try:
        buf = uploaded.read()
        doc = pymupdf.open(stream=buf, filetype="pdf")
        txt = "".join(page.get_text() for page in doc)
        return txt
    except Exception as e:
        st.error(f"PDF error: {e}")
        return ""


@st.cache_data
def image_to_text_easyocr(uploaded) -> str:
    """OCR an image (PNG/JPG) to plain text using EasyOCR."""
    if not EASYOCR_AVAILABLE:
        st.error("Image OCR requires EasyOCR. Install with: pip install easyocr")
        return ""
        
    try:
        img_bytes = uploaded.read()
        reader = load_ocr_reader()
        if reader:
            result = reader.readtext(img_bytes, detail=0)
            return "\n".join(result)
        else:
            st.error("OCR reader could not be initialized")
            return ""
    except Exception as e:
        st.error(f"Image OCR error: {e}")
        return ""


@st.cache_data
def txt_file_to_text(uploaded) -> str:
    try:
        data = uploaded.read().decode("utf-8", errors="ignore")
        return data
    except Exception as e:
        st.error(f"Text-file error: {e}")
        return ""


@st.cache_data
def docx_to_text(uploaded) -> str:
    """Extract text from a DOCX file."""
    if not DOCX_AVAILABLE:
        st.error("DOCX support is not available. Install python-docx with: pip install python-docx")
        return ""
        
    try:
        document = docx.Document(uploaded)
        return "\n".join([paragraph.text for paragraph in document.paragraphs])
    except Exception as e:
        st.error(f"DOCX error: {e}")
        return ""


# ────────────────────────  GROQ CLIENT  ──────────────────── #
@st.cache_resource
def get_groq_client():
    """Get cached Groq client instance"""
    if not GROQ_API_KEY:
        st.warning("No Groq API key found. Please set it in Streamlit secrets or as environment variable.")
        return None
    return Groq(api_key=GROQ_API_KEY)


def chat_completion_stream(history: list[dict], context_dict: dict, model_name: str, max_tokens: int):
    """Stream a completion from Groq and return the full text."""
    client = get_groq_client()
    if not client:
        yield "⚠️ No Groq API key configured. Please set GROQ_API_KEY in Streamlit secrets."
        return
        
    system_prompt = (
        "You are a helpful, accurate, and friendly AI assistant.\n"
        "• Provide clear, well-structured answers: use concise paragraphs, logical headings, and bullet points where helpful.\n"
        "• If the supplied *context* contains the answer, incorporate or cite it; otherwise answer from your own knowledge.\n"
        "• If you genuinely do not know, state that openly."
    )

    # Get last 4 conversation turns for history (increased from 2)
    conversation_history = history[-4:]
    
    # Build context string based on available sources
    context_parts = []
    
    if context_dict["image"]["text"]:
        context_parts.append(f"context from image:\n{context_dict['image']['text']}")
    
    if context_dict["pdf"]["text"]:
        context_parts.append(f"context from the pdf:\n{context_dict['pdf']['text']}")
    
    if context_dict["docx"]["text"]:
        context_parts.append(f"context from docx:\n{context_dict['docx']['text']}")
    
    if context_dict["txt"]["text"]:
        context_parts.append(f"context from txt:\n{context_dict['txt']['text']}")
    
    # Build the full prompt according to the requested format
    prompt = f"{system_prompt}\n"
    
    if context_parts:
        prompt += "here is the context\n"
        prompt += "\n".join(context_parts)
        prompt += "\n\nuse the context if needed while answering\n"
    
    # Create the messages array for the API call
    msgs = [{"role": "system", "content": prompt}]
    msgs.extend(conversation_history)

    try:
        stream = client.chat.completions.create(
            model       = model_name,
            messages    = msgs,
            temperature = 0.7,
            max_tokens  = max_tokens,
            stream      = True,
        )

        response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            response += (delta.content or "")
            yield response                        # incremental text
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"


# ──────────────────────────  UI  ─────────────────────────── #
st.set_page_config(page_title="AI Assistant", page_icon="💬", layout="centered")

# Initialize session state
if "all_sessions"        not in st.session_state: st.session_state.all_sessions        = {}
if "current_session_id"  not in st.session_state: st.session_state.current_session_id  = None
if "max_tokens"          not in st.session_state: st.session_state.max_tokens          = DEFAULT_MAX_TOKENS

# ───────────────── sidebar ───────────────── #
with st.sidebar:
    st.header("Chat Sessions")

    # Display warnings for missing dependencies
    if not EASYOCR_AVAILABLE:
        st.sidebar.warning("📷 Install easyocr to enable image OCR:\n```pip install easyocr```")
    
    if not DOCX_AVAILABLE:
        st.sidebar.warning("📝 Install python-docx to enable DOCX support:\n```pip install python-docx```")

    # New session
    if st.button("➕  New session"):
        session = create_session()
        sid = session["id"]
        st.session_state.all_sessions[sid] = session
        st.session_state.current_session_id = sid
        st.rerun()
        
    # Export/Import sessions
    with st.expander("Import/Export Sessions"):
        # Export
        if st.session_state.all_sessions:
            export_data = export_sessions()
            st.download_button(
                "Export All Sessions", 
                export_data,
                "groq_chat_sessions.json",
                "application/json",
                key="export_button"
            )
        
        # Import
        uploaded_file = st.file_uploader("Import Sessions", type=["json"], key="import_sessions")
        if uploaded_file is not None:
            import_data = uploaded_file.read().decode("utf-8")
            if import_sessions(import_data):
                st.success("Sessions imported successfully!")
                st.rerun()
        
    # Model selection (only available if there's an active session)
    if st.session_state.current_session_id:
        sess = st.session_state.all_sessions[st.session_state.current_session_id]
        current_model = sess.get("model", DEFAULT_MODEL)
        
        
        selected_model = st.selectbox(
            "Select model",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(current_model) if current_model in AVAILABLE_MODELS else 0,
            key="model_selector"
        )
        
        # Update session if model changed
        if selected_model != current_model:
            sess["model"] = selected_model
            save_session(sess)
            st.success(f"Model changed to {selected_model}")
            
        # Max tokens slider
        st.slider(
            "Max response tokens",
            min_value=MIN_TOKENS,
            max_value=MAX_TOKENS,
            value=st.session_state.max_tokens,
            step=256,
            key="max_tokens_slider",
            help="Control the maximum length of the model's response"
        )
        
        # Update session state when slider changes
        if "max_tokens_slider" in st.session_state:
            st.session_state.max_tokens = st.session_state.max_tokens_slider

    # File uploader (needs an active session)
    file_types = ["pdf", "png", "jpg", "jpeg", "txt"]
    if DOCX_AVAILABLE:
        file_types.append("docx")
        
    upl = st.file_uploader(f"Upload PDF / image / txt{' / docx' if DOCX_AVAILABLE else ''}",
                           type=file_types)
    if upl and st.session_state.current_session_id:
        sess = st.session_state.all_sessions[st.session_state.current_session_id]

        if upl.type.endswith("pdf"):
            text = pdf_to_text(upl)
            src = f"PDF: {upl.name}"
            if text:
                sess["context"]["pdf"] = {"text": text, "source": src}
                save_session(sess)
                st.success(f"Added context from {src}")
                
        elif upl.type.startswith("image/"):
            text = image_to_text_easyocr(upl)
            src = f"Image: {upl.name}"
            if text:
                sess["context"]["image"] = {"text": text, "source": src}
                save_session(sess)
                st.success(f"Added context from {src}")
                
        elif upl.name.endswith(".docx"):
            text = docx_to_text(upl)
            src = f"DOCX: {upl.name}"
            if text:
                sess["context"]["docx"] = {"text": text, "source": src}
                save_session(sess)
                st.success(f"Added context from {src}")
                
        else:  # plain text
            text = txt_file_to_text(upl)
            src = f"Text: {upl.name}"
            if text:
                sess["context"]["txt"] = {"text": text, "source": src}
                save_session(sess)
                st.success(f"Added context from {src}")

    # List existing sessions
    st.markdown("### Existing")
    for sid, sess in sorted(
        st.session_state.all_sessions.items(),
        key=lambda t: t[1].get("created") or t[1].get("created_at", ""),
        reverse=True):

        cols = st.columns([4, 1])
        if cols[0].button(sess["name"], key=f"sel_{sid}"):
            st.session_state.current_session_id = sid
            st.rerun()
        if cols[1].button("🗑️", key=f"del_{sid}"):
            delete_session(sid)
            if st.session_state.current_session_id == sid:
                st.session_state.current_session_id = None
            st.rerun()

    if not GROQ_API_KEY:
        st.warning("Set the **GROQ_API_KEY** in Streamlit secrets to talk to Groq.")
        st.info("In Streamlit Cloud, add this to your secrets.toml file.")

# ───────────────── main pane ───────────────── #
st.title("💬 AI Assistant -- Prof. Saud Afzal")

if not st.session_state.current_session_id:
    st.info("Create a session to start chatting.")
    st.stop()

sess = st.session_state.all_sessions[st.session_state.current_session_id]
st.subheader(f"Session: {sess['name']}")

# Display model info
current_model = sess.get("model", DEFAULT_MODEL)
st.caption(f"🤖 Model: {current_model}")

# Display context sources if available
context_sources = []
for ctx_type, ctx_data in sess["context"].items():
    if ctx_data["source"]:
        context_sources.append(ctx_data["source"])

if context_sources:
    st.caption(f"🔗 Context from: {', '.join(context_sources)}")

# clear button
if st.button("Clear chat & context"):
    sess["messages"].clear()
    sess["context"] = {
        "pdf": {"text": "", "source": None},
        "image": {"text": "", "source": None},
        "txt": {"text": "", "source": None},
        "docx": {"text": "", "source": None}
    }
    save_session(sess)
    st.rerun()

# show history
for msg in sess["messages"]:
    if msg["role"] != "system":              # hide system notes
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# chat input
prompt = st.chat_input("Ask me something…")
if prompt:
    # rename session on first user msg
    if not sess["messages"]:
        sess["name"] = (prompt[:30] + "…") if len(prompt) > 30 else prompt

    sess["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # streaming answer
    with st.chat_message("assistant"):
        container = st.empty()
        answer = ""
        try:
            for partial in chat_completion_stream(sess["messages"], sess["context"], sess.get("model", DEFAULT_MODEL), st.session_state.max_tokens):
                answer = partial
                container.markdown(answer + "▌")
            container.markdown(answer)       # final
        except Exception as e:
            answer = f"⚠️  {e}"
            container.error(answer)
            if "is not available" in str(e):
                container.warning("This may be because the selected model is not available in your Groq API subscription.")
                container.info("Try selecting a different model in the sidebar.")

    sess["messages"].append({"role": "assistant", "content": answer})
    save_session(sess)
