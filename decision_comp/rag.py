"""
RAG (Retrieval-Augmented Generation) for Decision Companion.
Builds an ephemeral Chroma collection from uploaded PDFs and returns retrieved context
for the research LLM. Scores (l, m, u) must be based strictly on this context.
"""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import CriterionSchema, OptionSchema


def build_rag_context(
    documents: List[Tuple[str, bytes]],
    options: List[OptionSchema],
    criteria: List[CriterionSchema],
    problem_description: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 12,
) -> str:
    """
    Load PDFs, chunk, embed into an ephemeral Chroma collection, and retrieve relevant
    context for the given problem, options, and criteria. Returns a single string of
    concatenated chunk texts for injection into the research prompt.
    """
    if not documents:
        return ""

    all_chunks: List[str] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    for filename, content in documents:
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".pdf", delete=False, prefix="decision_rag_"
            ) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                for doc in docs:
                    text = doc.page_content.strip()
                    if text:
                        chunks = splitter.split_text(text)
                        for c in chunks:
                            all_chunks.append((c, filename))
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            continue

    if not all_chunks:
        return ""

    # Ephemeral Chroma: unique collection name so no cross-request leakage
    client = chromadb.EphemeralClient()
    collection_name = f"rag_{uuid.uuid4().hex[:16]}"
    collection = client.create_collection(name=collection_name)

    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    texts = [t for t, _ in all_chunks]

    collection.add(documents=texts, ids=ids)

    # Query: problem + option names/descriptions + criterion names/descriptions
    query_parts = [problem_description]
    for o in options:
        query_parts.append(f"Option: {o.name}. {o.description or ''}")
    for c in criteria:
        query_parts.append(f"Criterion: {c.name}. {c.description or ''}")
    query = " ".join(query_parts).strip()

    results = collection.query(query_texts=[query], n_results=min(top_k, len(texts)))
    if not results or not results.get("documents") or not results["documents"][0]:
        return ""

    retrieved_docs = results["documents"][0]
    retrieved_ids = results.get("ids", [[]])[0] if results.get("ids") else []
    source_by_id = {cid: all_chunks[i][1] for i, cid in enumerate(ids)}
    out_parts = []
    for i, doc in enumerate(retrieved_docs):
        cid = retrieved_ids[i] if i < len(retrieved_ids) else f"chunk_{i}"
        src = source_by_id.get(cid, "PDF")
        out_parts.append(f"[Source: {src}]\n{doc}")
    return "\n\n---\n\n".join(out_parts)
