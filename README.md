# PDF RAG Chatbot

An AI-powered PDF question-answering application built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload a PDF document and ask questions about its contents using natural language.

The application extracts text from digital and scanned PDFs, converts the extracted content into embeddings, retrieves relevant information using FAISS, and generates answers using the Groq LLM API.

If the uploaded PDF does not contain the answer, the chatbot can use general knowledge and clearly identify that the answer is not directly available in the document.

## Live Demo

[Try the PDF RAG Chatbot](https://pdf-rag-chatbot-kt2399tduelqjsnwyyxwdk.streamlit.app/)

## Features

- **Single PDF Upload:** Upload one PDF document at a time.
- **Text Extraction:** Extracts text from digitally generated PDFs using PyPDF.
- **OCR Support:** Uses Tesseract OCR to extract text from scanned PDF pages.
- **Text Chunking:** Divides extracted content into smaller overlapping chunks for efficient retrieval.
- **Semantic Search:** Uses Sentence Transformers to convert text into numerical embeddings.
- **FAISS Vector Search:** Retrieves relevant document chunks based on similarity to the user's question.
- **AI-Powered Answers:** Uses Groq API with the `openai/gpt-oss-20b` model to generate answers.
- **General Knowledge Fallback:** Answers questions even when the PDF does not contain the required explanation, while clearly labelling such answers.
- **Source References:** Displays relevant PDF page numbers to help users verify information.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| PyPDF | PDF text extraction |
| PyMuPDF | PDF page rendering for OCR |
| Tesseract OCR | Text extraction from scanned PDFs |
| Pillow | Image processing |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Pre-trained embedding model |
| FAISS | Vector similarity search |
| Groq API | AI answer generation |
| NumPy | Numerical operations |

## System Architecture

The application follows a Retrieval-Augmented Generation (RAG) workflow.

1. **PDF Upload:** The user uploads a PDF document.
2. **Text Extraction:** Text is extracted using PyPDF. If a page has no extractable text, OCR is performed using Tesseract.
3. **Text Chunking:** Extracted text is divided into smaller chunks with overlapping content to preserve context.
4. **Embedding Generation:** Sentence Transformers converts each chunk into a 384-dimensional vector.
5. **Vector Storage:** FAISS stores the vectors and enables efficient similarity search.
6. **Question Processing:** The user's question is converted into an embedding.
7. **Context Retrieval:** FAISS retrieves the most relevant chunks from the uploaded PDF.
8. **Answer Generation:** The retrieved context is sent to the Groq LLM, which generates the response.
9. **Response Display:** The answer and relevant PDF page references are displayed in the Streamlit interface.

## Project Structure

```text
pdf-rag-chatbot/
│
├── data/
│   └── bda.pdf
│
├── app.py
├── pdf_processor.py
├── vector_store.py
├── rag_pipeline.py
│
├── requirements.txt
├── packages.txt
├── .env
├── .gitignore
└── README.md
```

### File Description

- `app.py` – Handles the Streamlit interface, PDF upload, processing workflow, and chatbot interaction.
- `pdf_processor.py` – Extracts text from PDFs, performs OCR when required, and creates text chunks.
- `vector_store.py` – Generates embeddings and manages FAISS indexing and similarity search.
- `rag_pipeline.py` – Connects to the Groq API and generates answers using retrieved context.
- `requirements.txt` – Contains the Python dependencies.
- `packages.txt` – Contains system-level packages required for deployment.
- `.env` – Stores the local Groq API key.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd pdf-rag-chatbot
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your GitHub repository URL.

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate the environment:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

**macOS**
```bash
brew install tesseract
```

**Ubuntu / Debian**
```bash
sudo apt install tesseract-ocr
```

For Windows, install Tesseract OCR using its Windows installer.

### 5. Configure the Groq API Key

Create a `.env` file in the root project directory:

```env
GROQ_API_KEY=your_groq_api_key
```

Get an API key from [Groq Console](https://console.groq.com/).

**Important:** Never upload your `.env` file or expose your API key on GitHub.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## How to Use

1. Open the application.
2. Upload a PDF document.
3. Click the processing button.
4. Wait for text extraction, embedding generation, and FAISS indexing to complete.
5. Enter a question related to the uploaded document.
6. View the generated answer and relevant source page references.

## Retrieval-Augmented Generation (RAG)

RAG combines information retrieval with language generation.

Instead of depending only on the language model's pre-trained knowledge, the application first retrieves relevant information from the uploaded PDF and then uses that information to generate an answer.

This helps make responses more relevant to the uploaded document.

The project also includes a general knowledge fallback for questions whose answers are missing from the PDF. Such answers are clearly identified so users can distinguish document-based information from general knowledge.

## Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment steps:

1. Push the project code to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect the GitHub repository.
4. Select `app.py` as the main application file.
5. Add `GROQ_API_KEY` under the application's Secrets settings.
6. Deploy the application.

## Current Limitations

- Supports one uploaded PDF at a time.
- OCR processing can take longer for scanned or image-heavy PDFs.
- OCR accuracy depends on document quality, image resolution, and text clarity.
- Answer quality depends on the relevance of retrieved chunks and the language model.
- Groq API usage is subject to API availability and rate limits.
- General knowledge answers may contain information that is not present in the uploaded PDF.

## Future Enhancements

- Support for multiple PDF uploads.
- Improved OCR accuracy and processing speed.
- Conversation history and follow-up questions.
- Downloadable answers and summaries.
- Better retrieval strategies and reranking.
- Support for additional document formats.

## Author

**Aryan Jagtap**

Computer Engineering Student

---

*Built as an academic project to explore Retrieval-Augmented Generation, Natural Language Processing, semantic search, vector databases, and Large Language Models.*