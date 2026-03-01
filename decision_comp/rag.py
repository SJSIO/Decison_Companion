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

    When len(documents) <= len(options), context is returned as per-option sections
    ([Context for Option 0 (name)], etc.) so the LLM can score each option from
    that option's document only. Option index i is assumed to correspond to document
    index i (first option = first uploaded file). When there are more documents than
    options, a single merged context is returned (legacy behavior).
    """
    if not documents:
        return ""

    # (chunk_text, filename, doc_index)
    all_chunks: List[Tuple[str, str, int]] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    for doc_index, (filename, content) in enumerate(documents):
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
                            all_chunks.append((c, filename, doc_index))
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
    texts = [t for t, _, _ in all_chunks]
    metadatas = [{"doc_index": d} for _, _, d in all_chunks]

    collection.add(documents=texts, ids=ids, metadatas=metadatas)

    num_options = len(options)
    num_documents = len(documents)
    use_per_option = num_documents <= num_options and num_documents > 0

    if use_per_option:
        # Per-option retrieval: one section per option; option i = document i
        out_parts = []
        for option_index in range(num_options):
            opt = options[option_index]
            section_header = f"[Context for Option {option_index} ({opt.name})]"
            if option_index >= num_documents:
                out_parts.append(f"{section_header}\nNo uploaded document for this option.")
                continue
            # Query with filter: only chunks from this option's document
            query_parts = [
                problem_description,
                f"Option: {opt.name}. {opt.description or ''}",
            ]
            for c in criteria:
                query_parts.append(f"Criterion: {c.name}. {c.description or ''}")
            query = " ".join(query_parts).strip()
            # How many chunks exist for this doc?
            doc_chunk_count = sum(1 for _, _, d in all_chunks if d == option_index)
            n_results = min(top_k, doc_chunk_count) if doc_chunk_count else 0
            if n_results == 0:
                out_parts.append(f"{section_header}\n(No text chunks extracted from this document.)")
                continue
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"doc_index": option_index},
            )
            if not results or not results.get("documents") or not results["documents"][0]:
                out_parts.append(f"{section_header}\n(No relevant chunks retrieved.)")
                continue
            chunk_texts = results["documents"][0]
            out_parts.append(f"{section_header}\n" + "\n\n".join(chunk_texts))
        return "\n\n---\n\n".join(out_parts)

    # Legacy: single global query, merged context with [Source: filename]
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
