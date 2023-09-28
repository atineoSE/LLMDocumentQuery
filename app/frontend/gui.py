import logging
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from app.state import State
from app.models.query import Query, RetrieveStrategy

app_state = State()


st.set_page_config(
    page_title="Query your document",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)
gui_state = st.session_state


def _update_document(file):
    logging.debug(f"Loading document \"{file.name}\"")
    with st.spinner("Uploading document..."):
        app_state.db.store(file)


def _query_LLM():
    if (query_text := gui_state.get("query_text", None)) is None:
        logging.debug("No query available")
        return

    with st.spinner("Getting answer..."):
        query = Query(
            text=query_text,
            retrieve_strategy=RetrieveStrategy.SIMILAR
        )
        texts = app_state.db.retrieve(query=query)
        gui_state["answer"] = app_state.llm.predict(query=query, texts=texts)


def _reset_answer():
    gui_state["answer"] = ""


def _reset_query():
    gui_state["query_text"] = ""


def _document_load():
    st.write("## Upload your document")
    st.write("Once uploaded, you can ask multiple times over the same document. \
            Upload a new document to replace.")

    uploaded_file = st.file_uploader(
        "document",
        type="pdf",
        key="document",
        label_visibility="hidden"
    )

    if uploaded_file is not None:
        _reset_answer()
        _reset_query()
        _update_document(uploaded_file)


def _text_input():
    st.text_input(
        "Query:",
        placeholder="What is the main conclusion of the article?",
        key="query_text",
        on_change=_reset_answer
    )
    st.button("Get answer", type="primary", on_click=_query_LLM)


def _answer():
    answer = gui_state.get("answer", "")
    st.text(answer)


_document_load()
_text_input()
_answer()
