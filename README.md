# AI Chat Assistant with Multi-Model Support

This is an interactive AI chat assistant built with Streamlit, supporting multiple Groq language models. It enables users to upload various document types (PDFs, images, text files, DOCX) and provides context-aware answers using Retrieval-Augmented Generation (RAG).

## Key Features

- Multi-model support with selectable Groq LLMs for diverse use cases
- Upload and process PDFs, images (with OCR), plain text, and DOCX files as context
- Maintains multiple chat sessions with persistent conversation history using Streamlit’s session state
- Streams AI-generated answers incrementally for a smooth user experience
- Lightweight OCR integration using Tesseract for extracting text from images
- Session import/export functionality for easy backup and restoration
- Optimized for deployment on Streamlit Cloud

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ai-chat-assistant.git
   cd ai-chat-assistant

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Set up your Groq API key:
   Create a `.streamlit/secrets.toml` file or set environment variable `GROQ_API_KEY`.

## Running the App

Run the Streamlit app locally:

  ```bash
    streamlit run app.py
 ## Usage

