"""
RAG System Backend — Educational Document Summarization & Q&A
Final Year Project · Alex · Computer Engineering

Run with:
    pip install fastapi uvicorn chromadb sentence-transformers pymupdf python-docx groq ragas rank_bm25
    uvicorn main:app --reload

Set your Groq API key (free at console.groq.com):
    Windows (PowerShell):  $env:GROQ_API_KEY="your-key-here"
    Mac/Linux:             export GROQ_API_KEY="your-key-here"
"""

from __future__ import annotations
import os, uuid, json, time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# ── FastAPI ────────────────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── NLP / ML ──────────────────────────────────────────────────────────────────
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from groq import Groq
import fitz          # PyMuPDF
import docx          # python-docx


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

EMBED_MODEL   = "all-MiniLM-L6-v2"   # free, runs locally, 384-dim vectors
GROQ_MODEL    = "llama-3.3-70b-versatile"   # free tier, fast inference
CHUNK_SIZE    = 400                   # tokens (approx characters / 4)
CHUNK_OVERLAP = 50
TOP_K         = 5                     # chunks to retrieve per query
CHROMA_PATH   = "./chroma_db"
DATA_PATH     = Path("./data/raw")
DATA_PATH.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STARTUP — load models & DB
# ══════════════════════════════════════════════════════════════════════════════

embedder: SentenceTransformer = None
collection = None          # ChromaDB collection
bm25_index = None          # BM25 index for hybrid retrieval
bm25_corpus: list[str] = []
bm25_ids:    list[str] = []
groq_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, collection, groq_client
    print("⏳ Loading embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("⏳ Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="edu_docs",
        metadata={"hnsw:space": "cosine"},
    )

    print("⏳ Initialising Groq client...")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

    print("✅ RAG system ready.")
    yield
    print("👋 Shutting down.")


app = FastAPI(title="EduRAG API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — DOCUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_pdf(path: Path) -> list[dict]:
    """Extract text page-by-page from a PDF."""
    doc = fitz.open(str(path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def parse_docx(path: Path) -> list[dict]:
    """Extract paragraphs from a DOCX file (treat each paragraph as a pseudo-page)."""
    doc = docx.Document(str(path))
    # Group paragraphs into ~page-sized chunks
    pages, current, count = [], [], 0
    for para in doc.paragraphs:
        if para.text.strip():
            current.append(para.text.strip())
            count += 1
        if count >= 20:
            pages.append({"page": len(pages) + 1, "text": "\n".join(current)})
            current, count = [], 0
    if current:
        pages.append({"page": len(pages) + 1, "text": "\n".join(current)})
    return pages


def parse_document(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    elif suffix in (".docx", ".doc"):
        return parse_docx(path)
    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Split txt into fake pages of ~1000 chars
        pages = []
        for i in range(0, len(text), 1000):
            pages.append({"page": i // 1000 + 1, "text": text[i:i+1000]})
        return pages
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def chunk_text(text: str, source: str, page: int) -> list[dict]:
    """
    Split text into overlapping character-based chunks.
    (In production, use a proper tokenizer for accurate token counts.)
    """
    char_size    = CHUNK_SIZE * 4       # ~4 chars per token
    char_overlap = CHUNK_OVERLAP * 4
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + char_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "id":     f"{source}__p{page}__c{idx}",
                "text":   chunk,
                "source": source,
                "page":   page,
            })
            idx += 1
        start += char_size - char_overlap
    return chunks


def chunk_document(pages: list[dict], source: str) -> list[dict]:
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_text(page["text"], source, page["page"]))
    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — EMBEDDING & STORAGE
# ══════════════════════════════════════════════════════════════════════════════

def embed_and_store(chunks: list[dict]) -> int:
    """Embed chunks and store in ChromaDB. Returns number stored."""
    global embedder
    if embedder is None:
        embedder = SentenceTransformer(EMBED_MODEL)
    if not chunks:
        return 0

    texts      = [c["text"] for c in chunks]
    ids        = [c["id"]   for c in chunks]
    metadatas  = [{"source": c["source"], "page": c["page"]} for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    # ChromaDB upsert (safe to re-run)
    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    # Update BM25 index
    global bm25_index, bm25_corpus, bm25_ids
    bm25_corpus.extend(texts)
    bm25_ids.extend(ids)
    tokenized = [t.lower().split() for t in bm25_corpus]
    bm25_index = BM25Okapi(tokenized)

    return len(chunks)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — RETRIEVAL (Dense + Hybrid)
# ══════════════════════════════════════════════════════════════════════════════

def dense_retrieval(query: str, k: int = TOP_K) -> list[dict]:
    """Retrieve top-k chunks by cosine similarity."""
    global embedder
    if embedder is None:
        embedder = SentenceTransformer(EMBED_MODEL)
    q_embed = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=q_embed,
        n_results=min(k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":   doc,
            "source": meta["source"],
            "page":   meta["page"],
            "score":  round(1 - dist, 4),   # cosine distance → similarity
        })
    return chunks


def hybrid_retrieval(query: str, k: int = TOP_K) -> list[dict]:
    """
    Combine dense retrieval (cosine similarity) with BM25 keyword matching.
    Uses Reciprocal Rank Fusion (RRF) to merge the two ranked lists.
    """
    if bm25_index is None or not bm25_corpus:
        return dense_retrieval(query, k)

    # Dense results
    dense = dense_retrieval(query, k * 2)
    dense_rank = {c["text"][:80]: i for i, c in enumerate(dense)}

    # BM25 results
    tokens = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokens)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:k * 2]
    bm25_rank = {bm25_corpus[i][:80]: rank for rank, i in enumerate(top_bm25_idx)}

    # RRF fusion
    all_keys = set(dense_rank) | set(bm25_rank)
    K_RRF = 60
    def rrf(key):
        dr = dense_rank.get(key, k * 4)
        br = bm25_rank.get(key, k * 4)
        return 1 / (K_RRF + dr) + 1 / (K_RRF + br)

    ranked = sorted(all_keys, key=rrf, reverse=True)[:k]
    # Map back to full chunk dicts
    text_to_chunk = {c["text"][:80]: c for c in dense}
    return [text_to_chunk[r] for r in ranked if r in text_to_chunk]


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — GENERATION (LLM)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an academic assistant for a university educational system.
Answer questions ONLY based on the retrieved document context provided.
Always cite the source document and page number.
If the answer is not in the context, say: "I couldn't find information about this in the uploaded documents."
Be concise but thorough. Use academic language."""

def generate_answer(query: str, chunks: list[dict]) -> dict:
    """Call Groq (Llama 3.3) to generate an answer grounded in retrieved chunks."""
    context_str = "\n\n".join(
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in chunks
    )
    user_message = f"Context:\n{context_str}\n\nQuestion: {query}"

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return {
        "answer":  completion.choices[0].message.content,
        "sources": [{"source": c["source"], "page": c["page"], "score": c["score"]} for c in chunks],
        "model":   completion.model,
        "tokens":  completion.usage.total_tokens,
    }


def generate_summary(doc_name: str, chunks: list[dict], mode: str = "abstractive") -> str:
    """Generate a structured summary of a document."""
    if mode == "extractive":
        # Extractive: just return the highest-scoring chunks concatenated
        return "\n\n---\n\n".join(c["text"] for c in chunks[:5])

    # Abstractive: LLM-generated
    context_str = "\n\n".join(c["text"] for c in chunks[:10])
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"You are an academic summariser. Create a well-structured summary of "
                f"the document '{doc_name}' based on the following extracted content.\n\n"
                f"Content:\n{context_str}\n\n"
                "Format your summary as:\n"
                "1. One-paragraph overview\n"
                "2. Key topics covered (bullet points)\n"
                "3. Who would benefit from reading this"
            )
        }],
    )
    return completion.choices[0].message.content


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# In-memory document registry (use a real DB in production)
DOCUMENT_REGISTRY: dict[str, dict] = {}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document.
    Pipeline: save → parse → chunk → embed → store in ChromaDB
    """
    allowed = {".pdf", ".docx", ".doc", ".txt"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"File type {suffix} not supported. Use: {allowed}")

    # Save file
    doc_id   = str(uuid.uuid4())[:8]
    save_path = DATA_PATH / f"{doc_id}_{file.filename}"
    content  = await file.read()
    save_path.write_bytes(content)

    # Parse → chunk → embed
    t0 = time.time()
    pages  = parse_document(save_path)
    chunks = chunk_document(pages, source=file.filename)
    n      = embed_and_store(chunks)
    elapsed = round(time.time() - t0, 2)

    # Register
    DOCUMENT_REGISTRY[doc_id] = {
        "id":        doc_id,
        "name":      file.filename,
        "pages":     len(pages),
        "chunks":    n,
        "size_kb":   round(len(content) / 1024, 1),
        "ingested_at": time.strftime("%Y-%m-%d %H:%M"),
        "elapsed_s": elapsed,
    }

    return {"status": "ok", "document": DOCUMENT_REGISTRY[doc_id]}


class QueryRequest(BaseModel):
    question: str
    retrieval: str = "hybrid"   # "dense" or "hybrid"
    top_k: int = TOP_K


@app.post("/query")
async def query_documents(req: QueryRequest):
    """Answer a question using the RAG pipeline."""
    if collection.count() == 0:
        raise HTTPException(400, "No documents ingested yet. Upload some first.")

    # Retrieve
    if req.retrieval == "hybrid":
        chunks = hybrid_retrieval(req.question, req.top_k)
    else:
        chunks = dense_retrieval(req.question, req.top_k)

    if not chunks:
        return {"answer": "No relevant content found.", "sources": []}

    # Generate
    result = generate_answer(req.question, chunks)
    return result


class SummariseRequest(BaseModel):
    document_name: str
    mode: str = "abstractive"   # "abstractive" or "extractive"
    top_k: int = 10


@app.post("/summarise")
async def summarise_document(req: SummariseRequest):
    """Generate a summary for a specific document."""
    # Retrieve representative chunks for this document
    results = collection.query(
        query_embeddings=embedder.encode([f"summary overview {req.document_name}"]).tolist(),
        n_results=min(req.top_k, collection.count() or 1),
        where={"source": req.document_name},
        include=["documents", "metadatas"],
    )
    chunks = [
        {"text": doc, "source": meta["source"], "page": meta["page"], "score": 1.0}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
    if not chunks:
        raise HTTPException(404, f"Document '{req.document_name}' not found in knowledge base.")

    summary = generate_summary(req.document_name, chunks, req.mode)
    return {"document": req.document_name, "mode": req.mode, "summary": summary}


@app.get("/documents")
async def list_documents():
    """List all ingested documents."""
    return {"documents": list(DOCUMENT_REGISTRY.values()), "total_chunks": collection.count()}


@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """Remove a document from the knowledge base."""
    if doc_id not in DOCUMENT_REGISTRY:
        raise HTTPException(404, "Document not found.")
    doc = DOCUMENT_REGISTRY.pop(doc_id)
    # Delete from ChromaDB (filter by source name)
    collection.delete(where={"source": doc["name"]})
    return {"status": "deleted", "document": doc["name"]}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "embed_model":   EMBED_MODEL,
        "total_chunks":  collection.count(),
        "total_docs":    len(DOCUMENT_REGISTRY),
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION MODULE (run separately)
# ══════════════════════════════════════════════════════════════════════════════

def run_ragas_evaluation(test_set_path: str = "evaluation/test_set.json"):
    """
    Evaluate the RAG system using RAGAS.

    test_set.json format:
    [
        {
            "question": "What is a binary tree?",
            "ground_truth": "A binary tree is a hierarchical data structure...",
            "contexts": ["...relevant passage..."]
        },
        ...
    ]

    Usage:
        python -c "from main import run_ragas_evaluation; run_ragas_evaluation()"
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError:
        print("Install: pip install ragas datasets")
        return

    with open(test_set_path) as f:
        raw = json.load(f)

    # Generate answers for all test questions
    for item in raw:
        chunks = hybrid_retrieval(item["question"])
        result = generate_answer(item["question"], chunks)
        item["answer"]   = result["answer"]
        item["contexts"] = [c["text"] for c in chunks]

    dataset = Dataset.from_list(raw)
    scores  = evaluate(dataset, metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ])

    print("\n── RAGAS Evaluation Results ──")
    for k, v in scores.items():
        print(f"  {k:25s}: {v:.4f}")

    # Save results
    Path("evaluation").mkdir(exist_ok=True)
    with open("evaluation/ragas_results.json", "w") as f:
        json.dump(dict(scores), f, indent=2)
    print("\nResults saved to evaluation/ragas_results.json")
    return scores


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)