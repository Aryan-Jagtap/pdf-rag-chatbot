
import os
import time
import hashlib
import tempfile

import streamlit as st

from pdf_processor import extract_text_from_pdf, create_chunks
from vector_store import (
    load_embedding_model,
    create_embeddings,
    create_faiss_index,
    search_similar_chunks
)
from rag_pipeline import generate_answer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 PDF RAG Chatbot")
st.caption("Upload a PDF and ask questions about its contents.")


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "processed_hash" not in st.session_state:
    st.session_state.processed_hash = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embedding_model():

    return load_embedding_model()


# --------------------------------------------------
# CACHE PDF EXTRACTION AND CHUNKING
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def process_pdf_cached(pdf_bytes, filename):

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as temp_file:

        temp_file.write(pdf_bytes)
        pdf_path = temp_file.name

    try:

        pages = extract_text_from_pdf(pdf_path)

        if not pages:
            raise ValueError("No pages could be extracted from this PDF.")

        chunks = create_chunks(pages)

        if not chunks:
            raise ValueError(
                "No readable text was found. "
                "Please check whether the PDF is readable."
            )

        # Attach original PDF filename to every chunk
        for chunk in chunks:
            chunk["source"] = filename

        return chunks, len(pages)

    finally:

        if os.path.exists(pdf_path):
            os.remove(pdf_path)


# --------------------------------------------------
# CACHE EMBEDDINGS
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def generate_embeddings_cached(texts, _model):

    # Reuse the tested embedding function
    # batch_size=1 and CPU processing are configured there.

    embedding_chunks = [
        {"text": text}
        for text in texts
    ]

    return create_embeddings(embedding_chunks, _model)


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

st.subheader("Upload your PDF")

uploaded_file = st.file_uploader(
    "Choose one PDF file",
    type=["pdf"],
    accept_multiple_files=False
)


if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    # Identify the uploaded PDF using its contents
    current_hash = hashlib.sha256(pdf_bytes).hexdigest()

    st.info(f"Selected PDF: {uploaded_file.name}")

    # --------------------------------------------------
    # PROCESS PDF
    # --------------------------------------------------

    if st.button("Process PDF", type="primary"):

        try:

            total_start = time.time()

            progress = st.progress(0)
            status = st.empty()

            # STEP 1: Extract text and create chunks

            status.info("Step 1/3: Extracting PDF text...")

            extraction_start = time.time()

            chunks, total_pages = process_pdf_cached(
                pdf_bytes,
                uploaded_file.name
            )

            extraction_time = time.time() - extraction_start

            progress.progress(33)

            st.write(
                f"**Extraction completed:** "
                f"{total_pages} pages, {len(chunks)} chunks "
                f"in {extraction_time:.2f} seconds."
            )

            # STEP 2: Generate embeddings

            status.info("Step 2/3: Loading model and generating embeddings...")

            embedding_start = time.time()

            model = get_embedding_model()

            texts = tuple(
                chunk["text"]
                for chunk in chunks
            )

            embeddings = generate_embeddings_cached(
                texts,
                model
            )

            embedding_time = time.time() - embedding_start

            progress.progress(66)

            st.write(
                f"**Embedding generation completed:** "
                f"{embeddings.shape[0]} vectors "
                f"in {embedding_time:.2f} seconds."
            )

            # STEP 3: Build FAISS index

            status.info("Step 3/3: Building FAISS index...")

            indexing_start = time.time()

            index = create_faiss_index(embeddings)

            indexing_time = time.time() - indexing_start

            progress.progress(100)

            # SAVE RESULTS IN SESSION STATE

            st.session_state.embedding_model = model
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.processed_hash = current_hash
            st.session_state.messages = []

            total_time = time.time() - total_start

            status.success("PDF processed successfully!")

            st.success(
                f"Processing completed in {total_time:.2f} seconds."
            )

            st.write(
                f"**FAISS indexing time:** {indexing_time:.2f} seconds."
            )

        except Exception as e:

            st.error(f"PDF processing failed: {str(e)}")

            st.exception(e)


# --------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------

if (
    uploaded_file is not None
    and st.session_state.processed_hash
    == hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    and st.session_state.index is not None
):

    st.divider()

    st.subheader("Ask Questions")

    st.caption(
        f"Currently answering questions from: {uploaded_file.name}"
    )

    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":

                sources = message.get("sources", [])

                if sources:

                    with st.expander("View source references"):

                        for source in sources:

                            st.write(
                                f"**{source['source']}** "
                                f"— Page {source['page']}"
                            )

                retrieved = message.get("retrieved", [])

                if retrieved:

                    with st.expander("Debug: Retrieved text"):

                        for item in retrieved:

                            st.write(
                                f"**Page {item['page']}** "
                                f"| Similarity: {item['score']:.4f}"
                            )

                            st.write(item["text"])

                            st.divider()

    # Accept new question

    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Searching PDF and generating answer..."):

                try:

                    # Retrieve relevant chunks

                    retrieved_chunks = search_similar_chunks(
                        question,
                        st.session_state.embedding_model,
                        st.session_state.index,
                        st.session_state.chunks,
                        top_k=8
                    )

                    # Generate answer using Groq

                    answer = generate_answer(
                        question,
                        retrieved_chunks
                    )

                    st.markdown(answer)

                    # Prepare source references

                    sources = []

                    for item in retrieved_chunks:

                        source_info = {
                            "source": item.get("source", uploaded_file.name),
                            "page": item["page"]
                        }

                        if source_info not in sources:
                            sources.append(source_info)

                    # Save conversation

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "retrieved": retrieved_chunks
                    })

                    if sources:

                        with st.expander("View source references"):

                            for source in sources:

                                st.write(
                                    f"**{source['source']}** "
                                    f"— Page {source['page']}"
                                )

                    with st.expander("Debug: Retrieved text"):

                        for item in retrieved_chunks:

                            st.write(
                                f"**Page {item['page']}** "
                                f"| Similarity: {item['score']:.4f}"
                            )

                            st.write(item["text"])

                            st.divider()

                except Exception as e:

                    st.error(f"Error generating answer: {str(e)}")


elif uploaded_file is not None:

    st.warning(
        "Please click 'Process PDF' before asking questions."
    )
