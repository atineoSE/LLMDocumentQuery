import logging
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from app.state import app_state
from app.models.query import Query, RetrieveStrategy


st.set_page_config(
    page_title="Query your document",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
gui_state = st.session_state


def _update_document():
    if (file := gui_state.get("document", None)) is not None:
        _reset_answer()
        _reset_query()
        logging.debug(f"Loading document \"{file.name}\"")
        with st.spinner("Processing document..."):
            app_state.db.store(file)


def _query_LLM():
    query_text = gui_state.get("query_text", None)
    if query_text:
        query_text = query_text.strip()
    if not query_text:
        logging.debug("No query available")
        return

    if gui_state.get("document", None) is None:
        logging.debug("Cleaning up previous document")
        app_state.db.cleanup_previous_document()

    with st.spinner("Getting answer..."):
        query = Query(
            text=query_text,
            retrieve_strategy=RetrieveStrategy.SIMILAR
        )
        texts = app_state.db.retrieve(query=query)
        gui_state["answer"] = app_state.llm.predict(
            query=query.text, texts=texts)


def _reset_answer():
    logging.debug(f"Resetting answer")
    gui_state["answer"] = ""


def _reset_query():
    logging.debug(f"Resetting query")
    gui_state["query_text"] = ""


def _header():
    st.write("## Query your document")
    st.write(
        f"Powered by open source model: {app_state.llm.llm_type.model_full_name}")
    st.write("Running on a NVIDIA RTX A6000")


def _document_load():
    st.write("### Upload your document")
    st.write("Once uploaded, you can ask multiple times over the same document. \
            Upload a new document to replace.")

    st.file_uploader(
        "document",
        type="pdf",
        key="document",
        label_visibility="hidden",
        on_change=_update_document
    )


def _text_input():
    st.text_input(
        "### Query",
        placeholder="What is the main conclusion of the article?",
        key="query_text",
        on_change=_reset_answer
    )
    st.button("Get answer", type="primary", on_click=_query_LLM)


def _answer():
    answer = gui_state.get("answer", "")
    st.write(answer)


_header()
_document_load()
_text_input()
_answer()
