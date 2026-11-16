
import os
import json
import faiss

from dotenv import load_dotenv
from groq import Groq

from vector_store import (
    load_embedding_model,
    search_similar_chunks
)


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

# If the key is not in .env, check Streamlit Cloud Secrets
if not API_KEY:

    try:
        import streamlit as st
        API_KEY = st.secrets["GROQ_API_KEY"]

    except Exception:
        API_KEY = None

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file "
        "or Streamlit Cloud Secrets."
    )


# ==========================================
# 2. INITIALIZE GROQ
# ==========================================

client = Groq(
    api_key=API_KEY,
    timeout=45.0,
    max_retries=2
)

GROQ_MODEL = "openai/gpt-oss-20b"


# ==========================================
# 3. LOAD SAVED VECTOR DATABASE
# ==========================================

def load_vector_store():

    print("Loading FAISS vector database...")

    index = faiss.read_index("data/faiss_index.bin")

    with open("data/chunks.json", "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print("Vector database loaded successfully!")

    return index, chunks


# ==========================================
# 4. GENERATE ANSWER USING GROQ
# ==========================================

def generate_answer(question, retrieved_chunks):

    # Prepare all retrieved PDF content
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        context_parts.append(
            f"""
            DOCUMENT CHUNK {i}

            PDF Name: {chunk.get('source', 'Unknown PDF')}
            Page Number: {chunk['page']}

            CONTENT:
            {chunk['text']}

            --------------------------------
            """
        )

    context = "\n".join(context_parts)

    # Instructions for Groq
    system_prompt = """
    You are an accurate PDF question-answering assistant.

    Your task is to answer questions using the information
    provided in the retrieved PDF content.

    FOLLOW THESE RULES STRICTLY:

    1. Read ALL the document chunks before preparing your answer.

    2. Do not focus only on the first or most relevant chunk.
       Information in later chunks is equally important.

    3. When the user asks for a list, identify EVERY relevant
       item mentioned across ALL the provided chunks.

    4. If the user asks for projects mentioned in a resume,
       identify every distinct project name you can find.
       Do not stop after finding the first project.

    5. For each project, include:
       - Project name
       - Description or purpose
       - Technologies used, if mentioned
       - The user's contribution, if explicitly mentioned

    6. Combine information about the same project if it appears
       in multiple chunks. Do not list the same project twice.

    7. Never ignore a project just because its description
       appears in a different document chunk.

    8. Do not invent project names, technologies, responsibilities,
       achievements, or other information.

    9. If a detail is not available in the PDF, write
       "Not specified in the provided PDF."

    10. If the PDF contains insufficient information to answer,
        clearly explain what information is missing.

    11. Mention the PDF filename and page number when possible.

    12. Use simple, clear language and organize answers with
        headings and bullet points.

    COMPLETENESS IS IMPORTANT:
    Before finalizing a list, review every provided document chunk
    again and check that no relevant item has been omitted.
    """

    user_prompt = f"""
    RETRIEVED PDF CONTENT:

    {context}

    USER QUESTION:

    {question}

    INSTRUCTIONS:

    Answer the question using all the retrieved content.

    If the question asks for all projects, list every distinct
    project that appears anywhere in the provided content.

    Do not provide only examples or a partial list.

    Check all document chunks before finalizing your answer.
    """

    print("\nSending question to Groq...")

    response = client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        max_completion_tokens=2048
    )

    answer = response.choices[0].message.content

    if not answer:

        raise RuntimeError(
            "Groq returned an empty response."
        )

    print("\nAnswer generated successfully using Groq!")

    return answer



   

# ==========================================
# 5. COMPLETE RAG PIPELINE
# ==========================================

def ask_question(question, model, index, chunks):

    print("\nSearching for relevant PDF content...")

    # Retrieve relevant chunks using FAISS
    retrieved_chunks = search_similar_chunks(
        question,
        model,
        index,
        chunks,
        top_k=4
    )

    if not retrieved_chunks:

        return "No relevant PDF content was found."

    # Display retrieved sources
    print("\nRetrieved PDF sources:")

    for result in retrieved_chunks:

        print(
            f"Page {result['page']} "
            f"| Similarity: {result['score']:.4f}"
        )

    print("\nGenerating answer using Groq...")

    # Generate final answer
    answer = generate_answer(
        question,
        retrieved_chunks
    )

    return answer


# ==========================================
# 6. RUN THE RAG CHATBOT
# ==========================================

if __name__ == "__main__":

    try:

        # Load embedding model
        embedding_model = load_embedding_model()

        # Load FAISS index and chunks
        index, chunks = load_vector_store()

        print("\nPDF RAG Chatbot is ready!")

        print("Type 'exit' to stop the chatbot.")

        while True:

            question = input("\nAsk a question: ").strip()

            if question.lower() in ["exit", "quit"]:

                print("Chatbot closed.")

                break

            if not question:

                print("Please enter a question.")

                continue

            try:

                answer = ask_question(
                    question,
                    embedding_model,
                    index,
                    chunks
                )

                print("\n========== ANSWER ==========")

                print(answer)

                print("\n============================")

            except Exception as e:

                print("\nError generating answer:", e)

    except Exception as e:

        print("\nApplication startup error:", e)
