
import os
import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from pdf_processor import extract_text_from_pdf, create_chunks


# ==========================================
# 1. LOAD THE EMBEDDING MODEL
# ==========================================

def load_embedding_model():

    print("Loading Sentence Transformer model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding model loaded successfully!")

    return model


# ==========================================
# 2. GENERATE EMBEDDINGS
# ==========================================


def create_embeddings(chunks, model):
    """
    Generate normalized embeddings in small batches.
    """

    import torch

    torch.set_num_threads(1)

    print("Generating embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device="cpu"
    )

    print("Embeddings generated successfully!")

    return embeddings









# ==========================================
# 3. CREATE FAISS INDEX
# ==========================================

def create_faiss_index(embeddings):

    # FAISS requires float32 vectors
    embeddings = np.asarray(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]

    # Create an inner-product index
    index = faiss.IndexFlatIP(dimension)

    # Add embeddings to the index
    index.add(embeddings)

    print("\nFAISS index created successfully!")

    print("Total vectors stored:", index.ntotal)

    return index


# ==========================================
# 4. SAVE INDEX AND CHUNKS
# ==========================================

def save_vector_store(index, chunks):

    os.makedirs("data", exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, "data/faiss_index.bin")

    # Save chunk text and metadata
    with open("data/chunks.json", "w", encoding="utf-8") as file:

        json.dump(chunks, file, ensure_ascii=False, indent=4)

    print("\nVector store saved successfully!")


# ==========================================
# 5. SEARCH FOR RELEVANT CHUNKS
# ==========================================

def search_similar_chunks(query, model, index, chunks, top_k=3):

    # Convert question into an embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
    "score": float(score),
    "source": chunks[idx].get("source", "Unknown PDF"),
    "page": chunks[idx]["page"],
    "text": chunks[idx]["text"]
})

    return results


# ==========================================
# 6. TEST THE COMPLETE VECTOR STORE
# ==========================================

if __name__ == "__main__":

    pdf_path = input("Enter PDF file path: ")

    try:

        # Extract PDF text
        pages = extract_text_from_pdf(pdf_path)

        # Create chunks
        chunks = create_chunks(pages)

        print("\nTotal chunks:", len(chunks))

        # Load embedding model
        model = load_embedding_model()

        # Generate embeddings
        embeddings = create_embeddings(chunks, model)

        print("\nEmbedding shape:", embeddings.shape)

        # Create FAISS index
        index = create_faiss_index(embeddings)

        # Save index and chunks
        save_vector_store(index, chunks)

        # Test semantic search
        query = input("\nEnter your question about the PDF: ")

        results = search_similar_chunks(
            query,
            model,
            index,
            chunks,
            top_k=5
        )

        print("\nMost relevant PDF chunks:")

        for i, result in enumerate(results, start=1):

            print(f"\n--- Result {i} ---")

            print("Page:", result["page"])

            print("Similarity score:", round(result["score"], 4))

            print("Text:", result["text"][:500])

    except Exception as e:

        print("\nError:", e)
