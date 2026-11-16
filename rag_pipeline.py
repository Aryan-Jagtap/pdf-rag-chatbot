
import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Groq model configuration
MODEL_NAME = "openai/gpt-oss-20b"


def get_groq_client():
    """
    Creates and returns the Groq API client.
    Reads the API key from Streamlit secrets or the local .env file.
    """

    api_key = os.getenv("GROQ_API_KEY")

    # Use Streamlit Cloud secrets if available
    try:
        if not api_key:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    if not api_key:
        raise ValueError(
            "Groq API key not found. Add GROQ_API_KEY to your .env "
            "file or Streamlit Cloud secrets."
        )

    return Groq(
        api_key=api_key,
        timeout=45,
        max_retries=2
    )


def generate_answer(question, retrieved_chunks):
    """
    Generates an answer using retrieved PDF content.
    If the PDF does not contain the answer, general knowledge is used
    and clearly identified.
    """

    client = get_groq_client()

    # Prepare context from retrieved PDF chunks
    context = ""

    for i, chunk in enumerate(retrieved_chunks, start=1):
        page = chunk.get("page", "Unknown")
        source = chunk.get("source", "Uploaded PDF")
        text = chunk.get("text", "")

        context += (
            f"\n[Source {i} | PDF: {source} | Page: {page}]\n"
            f"{text}\n"
        )

    # Instructions for the AI model
    system_prompt = """
You are a helpful and accurate PDF-based AI assistant.

Your job is to answer the user's questions using the uploaded PDF
whenever the required information is available.

Follow these rules carefully:

1. FIRST check whether the retrieved PDF content contains information
   that answers the user's question.

2. IF THE ANSWER IS AVAILABLE IN THE PDF:
   - Answer using the PDF content.
   - Preserve important terminology and concepts from the PDF.
   - Explain the answer clearly in simple language.
   - Give relevant examples where useful.
   - Mention the correct PDF page number.
   - Do not unnecessarily introduce unrelated information.

3. IF THE ANSWER IS NOT AVAILABLE IN THE PDF:
   - Do not refuse to answer.
   - Use your general knowledge to answer the question.
   - Start your response with:
     "Note: This answer is based on general knowledge because the
     uploaded PDF does not contain the explanation."
   - Provide a complete, accurate and useful answer.
   - Do not pretend that general knowledge came from the PDF.
   - Do not invent PDF page references.

4. IF THE PDF CONTAINS ONLY THE QUESTION BUT NOT ITS ANSWER:
   - Recognize that the question itself is not an explanation.
   - Answer using general knowledge.
   - Clearly identify the answer as general knowledge.

5. IF THE PDF CONTAINS PARTIAL INFORMATION:
   - Use the available PDF information first.
   - Fill missing details using general knowledge if necessary.
   - Clearly distinguish the additional information from PDF content.

6. NEVER invent information, quotations, page numbers or sources.

7. If the question is unclear, ask the user to clarify it.

8. Structure answers with headings, paragraphs and bullet points
   wherever appropriate.

9. For academic questions, provide answers that are easy to understand,
   sufficiently detailed and suitable for examination preparation.

10. Always answer the actual question instead of simply describing
    what information is or is not present in the PDF.
"""

    user_prompt = f"""
UPLOADED PDF CONTENT:

{context}

USER QUESTION:

{question}

Answer the question by following the instructions carefully.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
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
            temperature=0.2,
            max_tokens=1500
        )

        answer = response.choices[0].message.content

        return answer.strip() if answer else "Sorry, I could not generate an answer."

    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}"
