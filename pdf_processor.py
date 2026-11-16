
import time
import fitz
import pytesseract

from PIL import Image
from io import BytesIO
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF.

    1. Try normal text extraction using PyPDF.
    2. If a page has no readable text, use Tesseract OCR.
    3. Return extracted text page by page.
    """

    extracted_pages = []

    pdf_start = time.time()

    print(f"\nStarting PDF processing: {pdf_path}")

    # Open PDF using PyPDF
    reader = PdfReader(pdf_path)

    # Open the same PDF using PyMuPDF for OCR
    pdf_document = fitz.open(pdf_path)

    total_pages = len(reader.pages)

    print(f"Total pages: {total_pages}")

    try:
        for page_number in range(total_pages):

            page_start = time.time()

            print(f"\nProcessing page {page_number + 1}/{total_pages}")

            # STEP 1: Try normal text extraction
            text = reader.pages[page_number].extract_text() or ""

            # STEP 2: Use OCR only if normal extraction finds no text
            if not text.strip():

                print("No readable text found. Starting OCR...")

                ocr_start = time.time()

                pdf_page = pdf_document.load_page(page_number)

                # Render page at 200 DPI
                pix = pdf_page.get_pixmap(
                    dpi=200,
                    colorspace=fitz.csRGB,
                    alpha=False
                )

                # Convert rendered page into an image
                image = Image.open(
                    BytesIO(pix.tobytes("png"))
                )

                # Perform OCR
                text = pytesseract.image_to_string(
                    image,
                    lang="eng",
                    config="--psm 3"
                )

                ocr_time = time.time() - ocr_start

                print(f"OCR completed in {ocr_time:.2f} seconds")

            else:
                print("Normal text extraction successful. OCR skipped.")

            # STEP 3: Store extracted page text
            extracted_pages.append({
                "page": page_number + 1,
                "text": text.strip()
            })

            page_time = time.time() - page_start

            print(f"Page completed in {page_time:.2f} seconds")

    finally:
        pdf_document.close()

    total_time = time.time() - pdf_start

    print("\nPDF extraction completed.")
    print(f"Total pages processed: {total_pages}")
    print(f"Total extraction time: {total_time:.2f} seconds")

    return extracted_pages


def create_chunks(extracted_pages, chunk_size=450, overlap=80):
    """
    Divide extracted PDF text into smaller overlapping chunks.

    Each chunk retains its original page number.
    """

    chunks = []

    for page_data in extracted_pages:

        text = page_data["text"]
        page_number = page_data["page"]

        if not text.strip():
            continue

        start = 0

        while start < len(text):

            end = start + chunk_size

            # Try to end at a word boundary
            if end < len(text):

                space_position = text.rfind(" ", start, end)

                if space_position > start:
                    end = space_position

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "page": page_number,
                    "text": chunk_text
                })

            # Move forward with overlap
            next_start = end - overlap

            if next_start <= start:
                next_start = end

            start = next_start

    print(f"\nTotal chunks created: {len(chunks)}")

    return chunks
