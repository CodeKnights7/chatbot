import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
import numpy as np

# -----------------------------
# PDF paths (ADD AS MANY AS YOU WANT)
# -----------------------------
PDF_PATHS = [
    "data/Mutual Funds Database Expanded.pdf",
    "data/Mutual Funds Complete Guide.pdf"
]

INDEX_PATH = "data/faiss.index"
CHUNKS_PATH = "data/chunks.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks

# -----------------------------
# Ingest PDFs
# -----------------------------
def ingest_pdfs():
    print("📄 Reading PDFs...")
    all_chunks = []

    for pdf_path in PDF_PATHS:
        if not os.path.exists(pdf_path):
            print(f"⚠️ File not found: {pdf_path}")
            continue

        print(f"➡️ Processing: {pdf_path}")
        full_text = ""

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                except Exception as e:
                    print(f"⚠️ Skipping page {page_num} in {pdf_path}: {e}")

        chunks = chunk_text(full_text)

        # OPTIONAL: store source info (recommended)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": os.path.basename(pdf_path)
            })

    if not all_chunks:
        raise RuntimeError("❌ No text extracted from any PDF")

    print(f"🧩 Total chunks from all PDFs: {len(all_chunks)}")

    # -----------------------------
    # Embeddings
    # -----------------------------
    texts = [c["text"] for c in all_chunks]

    print("🔢 Generating embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # -----------------------------
    # FAISS
    # -----------------------------
    print("📦 Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("✅ PDFs ingested successfully")
    print(f"📁 FAISS index → {INDEX_PATH}")
    print(f"📁 Chunks → {CHUNKS_PATH}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    ingest_pdfs()
