import logging
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from app.state import State

app_state = State()


st.set_page_config(
    page_title="Query your document",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
gui_state = st.session_state


def _update_document():
    if (document := gui_state["document"]) is None:
        logging.debug("No document to update")
        return

    with st.spinner("Uploading document..."):
        app_state.db.store(document)


def _query_LLM():
    if (query := gui_state["query"]) is None:
        logging.debug("No query available")
        return

    with st.spinner("Getting answer..."):
        texts = app_state.db.retrieve(query=query)
        gui_state["answer"] = app_state.llm.predict(query=query, texts=texts)


def _reset_answer():
    gui_state["answer"] = ""


def _document_load():
    st.write("## Upload your document")
    st.write("Once uploaded, you can ask multiple times over the same document. \
            Upload a new document to replace.")

    gui_state["document"] = st.file_uploader(
        "document",
        type="pdf",
        key="pdf_file",
        label_visibility="hidden",
        on_change=_update_document
    )


def _text_input():
    gui_state["query"] = st.text_input(
        "Query:",
        placeholder="What is the main conclusion of the article?",
    )
    st.button("Get answer", type="primary", on_click=_query_LLM)


def _answer():
    st.text(value=gui_state["answer"])


_document_load()
_text_input()
_answer()
