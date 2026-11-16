
import os
import hashlib
import streamlit as st

from pdf_processor import extract_text_from_pdf, create_chunks

from vector_store import (
    load_embedding_model,
    create_embeddings,
    create_faiss_index,
    search_similar_chunks
)

from rag_pipeline import generate_answer


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")

st.write(
    "Upload multiple PDF documents and ask questions "
    "about their content using AI."
)


# ==========================================
# 2. SESSION STATE
# ==========================================

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


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def get_embedding_model():

    return load_embedding_model()


# ==========================================
# 4. MULTIPLE PDF UPLOAD
# ==========================================

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

current_hash = None

if uploaded_files:

    # Generate a unique hash for all uploaded PDFs
    hash_object = hashlib.sha256()

    for uploaded_file in uploaded_files:

        hash_object.update(
            uploaded_file.name.encode("utf-8")
        )

        hash_object.update(
            uploaded_file.getvalue()
        )

    current_hash = hash_object.hexdigest()

    st.success(
        f"{len(uploaded_files)} PDF(s) selected successfully."
    )

    # Display uploaded documents
    with st.expander("View uploaded documents"):

        for uploaded_file in uploaded_files:

            st.write(f"📄 {uploaded_file.name}")


    # ==========================================
    # 5. PROCESS PDF DOCUMENTS
    # ==========================================

    if st.button("Process PDFs", type="primary"):

        try:

            with st.spinner(
                "Processing PDFs. OCR may take some time..."
            ):

                os.makedirs("data", exist_ok=True)

                all_chunks = []

                total_pages = 0

                processed_files = 0

                # Process each uploaded PDF
                for file_number, uploaded_file in enumerate(
                    uploaded_files,
                    start=1
                ):

                    st.write(
                        f"Processing: {uploaded_file.name}"
                    )

                    pdf_path = (
                        f"data/upload_{file_number}.pdf"
                    )

                    # Save uploaded PDF temporarily
                    with open(pdf_path, "wb") as file:

                        file.write(
                            uploaded_file.getvalue()
                        )

                    # Extract text from PDF
                    pages = extract_text_from_pdf(
                        pdf_path
                    )

                    if not pages:

                        st.warning(
                            f"No readable text found in "
                            f"{uploaded_file.name}. Skipping it."
                        )

                        continue

                    # Create chunks
                    pdf_chunks = create_chunks(pages)

                    # Add PDF filename to every chunk
                    for chunk in pdf_chunks:

                        chunk["source"] = uploaded_file.name

                    all_chunks.extend(pdf_chunks)

                    total_pages += len(pages)

                    processed_files += 1


                # Check whether any PDF was processed
                if not all_chunks:

                    raise ValueError(
                        "No readable text could be extracted "
                        "from the uploaded PDFs."
                    )


                # ==========================================
                # 6. GENERATE EMBEDDINGS
                # ==========================================

                st.write("Generating embeddings...")

                model = get_embedding_model()

                embeddings = create_embeddings(
                    all_chunks,
                    model
                )


                # ==========================================
                # 7. CREATE FAISS INDEX
                # ==========================================

                st.write("Creating FAISS vector database...")

                index = create_faiss_index(
                    embeddings
                )


                # ==========================================
                # 8. SAVE DATA IN SESSION STATE
                # ==========================================

                st.session_state.embedding_model = model

                st.session_state.index = index

                st.session_state.chunks = all_chunks

                st.session_state.processed_hash = current_hash

                # Clear previous chat
                st.session_state.messages = []


            st.success(
                f"Successfully processed {processed_files} PDF(s)! "
                f"{total_pages} pages with extracted text and "
                f"{len(all_chunks)} chunks created."
            )

        except Exception as e:

            st.error(
                f"PDF processing failed: {e}"
            )


# ==========================================
# 9. CHAT INTERFACE
# ==========================================

if (
    uploaded_files
    and st.session_state.processed_hash is not None
    and st.session_state.processed_hash == current_hash
    and st.session_state.index is not None
):

    st.divider()

    st.subheader("💬 Ask Questions Across Your PDFs")

    st.caption(
        "Ask questions about the information contained "
        "in your uploaded documents."
    )


    # ==========================================
    # 10. DISPLAY CHAT HISTORY
    # ==========================================

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(
                message["content"]
            )

            if message.get("sources"):

                with st.expander("📄 View PDF sources"):

                    for source in message["sources"]:

                        st.write(
                            f"**PDF:** "
                            f"{source.get('source', 'Unknown PDF')}"
                        )

                        st.write(
                            f"**Page:** {source['page']}"
                        )

                        st.write(
                            f"**Similarity:** "
                            f"{source['score']:.4f}"
                        )

                        st.divider()


    # ==========================================
    # 11. QUESTION INPUT
    # ==========================================

    question = st.chat_input(
        "Ask a question about your PDFs..."
    )


    if question:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):

            st.markdown(question)


        # ==========================================
        # 12. RETRIEVE RELEVANT CHUNKS
        # ==========================================

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Searching PDFs and generating your answer..."
                ):

                    retrieved_chunks = search_similar_chunks(
                        question,
                        st.session_state.embedding_model,
                        st.session_state.index,
                        st.session_state.chunks,
                        top_k=8
                    )


                    # ==========================================
                    # 13. DEBUG RETRIEVED CONTENT
                    # ==========================================

                    with st.expander(
                        "🔍 Debug: Retrieved text",
                        expanded=False
                    ):

                        if retrieved_chunks:

                            for i, chunk in enumerate(
                                retrieved_chunks,
                                start=1
                            ):

                                st.markdown(
                                    f"### Result {i}"
                                )

                                st.write(
                                    "PDF:",
                                    chunk.get(
                                        "source",
                                        "Unknown PDF"
                                    )
                                )

                                st.write(
                                    "Page:",
                                    chunk["page"]
                                )

                                st.write(
                                    "Similarity:",
                                    round(
                                        chunk["score"],
                                        4
                                    )
                                )

                                st.write(
                                    "Retrieved text:"
                                )

                                st.write(
                                    chunk["text"]
                                )

                                st.divider()

                        else:

                            st.write(
                                "No relevant content was retrieved."
                            )


                    # ==========================================
                    # 14. GENERATE ANSWER USING GROQ
                    # ==========================================

                    if not retrieved_chunks:

                        answer = (
                            "I could not find relevant information "
                            "in the uploaded PDFs."
                        )

                    else:

                        answer = generate_answer(
                            question,
                            retrieved_chunks
                        )


                    # Display generated answer
                    st.markdown(answer)


                    # ==========================================
                    # 15. DISPLAY SOURCE REFERENCES
                    # ==========================================

                    sources = retrieved_chunks

                    if sources:

                        with st.expander(
                            "📄 View PDF sources"
                        ):

                            for source in sources:

                                st.write(
                                    f"**PDF:** "
                                    f"{source.get('source', 'Unknown PDF')}"
                                )

                                st.write(
                                    f"**Page:** {source['page']}"
                                )

                                st.write(
                                    f"**Similarity:** "
                                    f"{source['score']:.4f}"
                                )

                                st.divider()


                # ==========================================
                # 16. SAVE ASSISTANT RESPONSE
                # ==========================================

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer,

                    "sources": sources

                })


            except Exception as e:

                st.error(
                    f"Error generating answer: {e}"
                )


