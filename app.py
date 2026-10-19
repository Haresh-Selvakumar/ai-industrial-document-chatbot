"""Streamlit interface for the AI-powered industrial document chatbot."""

from __future__ import annotations

import html

import streamlit as st

import config
from rag_pipeline import IndustrialRAG, documents_to_chunks


st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2rem;}
    [data-testid="stSidebar"] {background: #f6f8fa;}
    .source-card {
        border-left: 4px solid #0b6e4f;
        background: #f4fbf8;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0.35rem;
    }
    .status-card {
        background: #eef4ff;
        border: 1px solid #cad9f5;
        padding: 0.8rem;
        border-radius: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def create_pipeline() -> IndustrialRAG:
    return IndustrialRAG()


pipeline = create_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_summary" not in st.session_state:
    st.session_state.document_summary = None


def clear_chat() -> None:
    st.session_state.messages = []


with st.sidebar:
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload industrial documents",
        type=config.SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help="Supported formats: PDF and TXT. Scanned PDFs require OCR and are not supported yet.",
    )

    process_clicked = st.button(
        "Process documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if process_clicked:
        try:
            with st.status("Preparing the knowledge base...", expanded=True) as status:
                st.write("Extracting and cleaning document text")
                file_pairs = [(file.name, file) for file in uploaded_files]
                chunks = documents_to_chunks(file_pairs)
                st.write(f"Created {len(chunks)} searchable text chunks")
                st.write("Loading Hugging Face models and generating embeddings")
                pipeline.build_index(chunks)
                status.update(label="Knowledge base ready", state="complete")

            st.session_state.document_summary = {
                "files": [file.name for file in uploaded_files],
                "chunks": len(chunks),
            }
            clear_chat()
            st.success("Documents processed successfully.")
        except Exception as error:
            st.error(f"Document processing failed: {error}")

    summary = st.session_state.document_summary
    if summary:
        st.markdown(
            f"""
            <div class="status-card">
                <strong>Index ready</strong><br>
                {len(summary['files'])} document(s)<br>
                {summary['chunks']} searchable chunks
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Indexed documents"):
            for filename in summary["files"]:
                st.write(f"• {filename}")
    else:
        st.info("Upload and process at least one document before asking questions.")

    st.divider()
    st.caption(f"Answer model: {config.LLM_MODEL_NAME}")
    st.caption(f"Embedding model: {config.EMBEDDING_MODEL_NAME}")
    st.button("Clear chat", use_container_width=True, on_click=clear_chat)


st.title("⚙️ Industrial Document Assistant")
st.caption(config.APP_SUBTITLE)

with st.expander("How to use this application"):
    st.markdown(
        """
        1. Upload one or more industrial PDF or TXT documents.
        2. Select **Process documents** and wait for indexing to finish.
        3. Ask a specific question about operation, maintenance, safety, or troubleshooting.
        4. Review the cited passages under **Supporting sources**.

        This prototype assists document retrieval. Verify safety-critical decisions against the
        original manual and qualified personnel.
        """
    )

if not st.session_state.messages:
    st.markdown("#### Suggested questions")
    suggestions = [
        "What safety precautions are required before maintenance?",
        "Explain the startup procedure.",
        "What should be checked when the motor overheats?",
    ]
    columns = st.columns(3)
    for column, suggestion in zip(columns, suggestions):
        column.code(suggestion, language=None)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Supporting sources"):
                for source in message["sources"]:
                    safe_name = html.escape(source["source"])
                    safe_text = html.escape(source["text"])
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <strong>{safe_name} — page {source['page']}</strong>
                            <br><small>Similarity: {source['score']:.3f}</small>
                            <br><br>{safe_text}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

question = st.chat_input(
    "Ask a question from the processed documents",
    disabled=not pipeline.ready,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents and generating an answer..."):
                answer, results = pipeline.ask(question)
            st.markdown(answer)

            sources = [
                {
                    "source": result.chunk.source,
                    "page": result.chunk.page,
                    "text": result.chunk.text,
                    "score": result.score,
                }
                for result in results
            ]

            if sources:
                with st.expander("Supporting sources"):
                    for source in sources:
                        st.markdown(
                            f"**{source['source']} — page {source['page']}**  "
                            f"\nSimilarity: `{source['score']:.3f}`\n\n"
                            f"> {source['text']}"
                        )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as error:
            error_message = f"Unable to answer the question: {error}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "sources": []}
            )

