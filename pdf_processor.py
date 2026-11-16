
import fitz
import pytesseract

from PIL import Image
from pypdf import PdfReader


# ==========================================
# 1. EXTRACT TEXT FROM PDF (WITH OCR)
# ==========================================

def extract_text_from_pdf(pdf_path):

    extracted_pages = []

    reader = PdfReader(pdf_path)
    pdf_document = fitz.open(pdf_path)

    try:
        for page_number, page in enumerate(reader.pages, start=1):

            # First, try normal text extraction
            text = page.extract_text() or ""

            # If no text is found, use OCR
            if not text.strip():

                print(f"Page {page_number}: Running OCR...")

                pdf_page = pdf_document[page_number - 1]

                # Convert PDF page into a high-resolution image
                pix = pdf_page.get_pixmap(
                    dpi=350,
                    colorspace=fitz.csRGB,
                    alpha=False
                )

                image = Image.frombytes(
                    "RGB",
                    (pix.width, pix.height),
                    pix.samples
                )

                # Extract text from image using Tesseract
                text = pytesseract.image_to_string(
                    image,
                    lang="eng"
                )

            # Store extracted text and page number
            if text.strip():

                extracted_pages.append({
                    "page": page_number,
                    "text": text.strip()
                })

            else:
                print(f"Warning: No text found on page {page_number}")

    finally:
        pdf_document.close()

    return extracted_pages


# ==========================================
# 2. CREATE TEXT CHUNKS
# ==========================================

def create_chunks(extracted_pages, chunk_size=450, overlap=80):

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    chunks = []
    chunk_id = 1

    for page in extracted_pages:

        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            # Try to end at a word boundary
            if end < len(text):

                boundary = text.rfind(
                    " ",
                    start + int(chunk_size * 0.8),
                    end
                )

                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "text": chunk_text
                })

                chunk_id += 1

            if end >= len(text):
                break

            # Move forward while retaining overlapping text
            start = end - overlap

    return chunks


# ==========================================
# 3. TEST THE COMPLETE PDF PROCESSOR
# ==========================================

if __name__ == "__main__":

    pdf_path = input("Enter PDF file path: ")

    try:

        # Extract text
        pages = extract_text_from_pdf(pdf_path)

        print("\nPDF processing completed!")

        print(
            "Total pages with extracted text:",
            len(pages)
        )

        # Create chunks
        chunks = create_chunks(pages)

        print("\nChunking completed!")

        print("Total chunks created:", len(chunks))

        # Display first three chunks
        for chunk in chunks[:3]:

            print(f"\n--- Chunk {chunk['chunk_id']} ---")

            print("Page:", chunk["page"])

            print("Text:", chunk["text"][:300])

    except Exception as e:

        print("\nError processing PDF:", e)
