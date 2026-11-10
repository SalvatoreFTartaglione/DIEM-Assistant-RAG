"""
bm25_indexer.py

Standalone BM25 index builder for the DIEM Agentic RAG pipeline.

Responsibilities:
- Extract all child chunks from the ChromaDB child collection via get().
- Build two BM25Okapi indices per chunk using a dual-tokenization strategy:
    1. Lemmatized index : spaCy it_core_news_lg pipeline (lemma, no stopwords,
                        no punctuation, alphabetic tokens only).
    2. Raw index       : lowercase original tokens, preserving acronyms and
                        technical terms (e.g. ISPD, IT4DiMe, CUN).
- Serialize both indices and the shared parent_id mapping to disk via pickle.

Output files (all inside CHROMA_DB_DIRECTORY):
    bm25_lemma.pkl   — BM25Okapi on lemmatized corpus
    bm25_raw.pkl     — BM25Okapi on raw-lowercase corpus
    bm25_meta.pkl    — list of dicts {parent_id, child_id, source} aligned
                    by position with both corpora

The dual-index approach maximises recall:
- Lemmatized index catches morphological variants ("rappresentate" -> "rappresentare").
- Raw index catches exact acronyms and technical terms that spaCy would mangle.
Both indices are queried at retrieval time and their results are unioned before
passing to the cross-encoder reranker.

Dependencies:
    pip install rank-bm25 spacy
    python -m spacy download it_core_news_lg
"""

import pickle
import re
import time
from pathlib import Path
from typing import Any
import config
import spacy
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Configuration (paths come from config.py)
# ---------------------------------------------------------------------------
CHROMA_DB_DIRECTORY   = config.CHROMA_DB_DIRECTORY
CHILD_DB_DIRECTORY    = config.CHILD_DB_DIRECTORY
CHILD_COLLECTION_NAME = config.CHILD_COLLECTION_NAME
BM25_LEMMA_PATH = config.BM25_LEMMA_PATH
BM25_RAW_PATH   = config.BM25_RAW_PATH
BM25_META_PATH  = config.BM25_META_PATH

# Batch size for ChromaDB get() calls — avoids memory spikes on large collections.
CHROMA_FETCH_BATCH = 5_000

# Minimum token count after tokenization to retain a chunk in the index.
# Chunks shorter than this (e.g. empty PDFs, stubs) are excluded.
MIN_TOKENS = 3


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _load_spacy() -> spacy.language.Language:
    """
    Load the Italian spaCy model with only the components needed for
    lemmatization and stop-word detection (tagger, morphologizer, lemmatizer).
    Disables the parser and NER to reduce memory usage and speed up processing.
    """
    print("[spaCy] Loading it_core_news_lg ...")
    nlp = spacy.load(
        "it_core_news_lg",
        disable=["parser", "ner", "senter"],
    )
    print(f"[spaCy] Model loaded. Pipeline: {nlp.pipe_names}")
    return nlp


def _iter_truncated_texts(
    chunks: list[dict[str, Any]],
    max_chars: int = 1_000_000,
):
    """
    Generator that yields each chunk's document text truncated to max_chars.

    Passing a generator to nlp.pipe() instead of a materialised list avoids
    duplicating the entire text corpus in memory before spaCy starts processing.
    The max_chars guard prevents spaCy from exceeding its internal limits on
    unusually large documents.
    """
    for c in chunks:
        yield c["document"][:max_chars]


def tokenize_raw(text: str) -> list[str]:
    """
    Raw tokenization: split on whitespace, non-alphanumeric characters, and
    underscores, then lowercase everything and keep tokens of at least 2
    characters.

    Using [\\s\\W_]+ (explicit underscore) instead of [\\s\\W]+ because \\W
    in Python excludes underscore, which would leave compound tokens such as
    'bando_incarichi_2024' unsplit, causing missed matches on individual terms.

    This pass preserves acronyms (ISPD, CUN, IT4DiMe) and technical terms
    that spaCy's lemmatizer would normalise or drop.
    """
    tokens = re.split(r"[\s\W_]+", text)
    return [t.lower() for t in tokens if len(t) >= 2]


# ---------------------------------------------------------------------------
# ChromaDB extraction
# ---------------------------------------------------------------------------

def fetch_all_children(child_db_dir: str, collection_name: str) -> list[dict[str, Any]]:
    """
    Extract all child chunks from ChromaDB using batched get() calls.

    Returns a list of dicts with keys:
        id          — ChromaDB document id (child_id)
        document    — raw page_content string
        metadata    — dict of ChromaDB metadata fields
    """
    print(f"\n[Fetch] Connecting to ChromaDB at '{child_db_dir}' " f"(collection: '{collection_name}') ...")

    # The raw chromadb client is used directly: get() needs no embedding model.
    from chromadb import PersistentClient

    client     = PersistentClient(path=child_db_dir)
    collection = client.get_collection(name=collection_name)

    total = collection.count()
    print(f"[Fetch] Total child chunks in collection: {total}")

    all_ids       = []
    all_documents = []
    all_metadatas = []

    offset = 0
    while offset < total:
        batch = collection.get(
            limit=CHROMA_FETCH_BATCH,
            offset=offset,
            include=["documents", "metadatas"],
        )
        all_ids.extend(batch["ids"])
        all_documents.extend(batch["documents"])
        all_metadatas.extend(batch["metadatas"])
        offset += CHROMA_FETCH_BATCH
        print(f"[Fetch] Retrieved {min(offset, total)}/{total} chunks "
            f"(ids: {len(all_ids)}, docs: {len(all_documents)}, "
            f"metadatas: {len(all_metadatas)}) ...")

    print(f"[Fetch] Extraction complete: {len(all_ids)} chunks loaded "
        f"({len(all_ids)} ids, {len(all_documents)} documents, "
        f"{len(all_metadatas)} metadata entries).")
    return [
        {"id": cid, "document": doc, "metadata": meta}
        for cid, doc, meta in zip(all_ids, all_documents, all_metadatas)
    ]


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def build_indices(
    chunks: list[dict[str, Any]],
    nlp: spacy.language.Language,
) -> tuple[BM25Okapi, BM25Okapi, list[dict[str, str]]]:
    """
    Build two BM25Okapi indices (lemmatized and raw) over the child chunks.

    Both corpora share the same positional alignment so that a hit at index i
    in either corpus maps to meta[i] for parent_id lookup.

    Chunks whose tokenized length falls below MIN_TOKENS are skipped and
    represented by an empty token list to preserve positional alignment.

    Returns:
        bm25_lemma  — BM25Okapi on lemmatized corpus
        bm25_raw    — BM25Okapi on raw-lowercase corpus
        meta        — list of {parent_id, child_id, source} aligned by position
    """
    corpus_lemma: list[list[str]] = []
    corpus_raw:   list[list[str]] = []
    meta:         list[dict[str, str]] = []

    skipped   = 0
    total     = len(chunks)
    log_every = max(1, total // 20)  # log progress every 5 %

    print(f"\n[Build] Tokenizing {total} child chunks ...")
    t0 = time.time()

    # Batch spaCy processing for efficiency — pipe() accepts a generator,
    # avoiding the memory cost of materialising all texts as a separate list.
    # The truncation guardrail (1_000_000 chars) is applied inside the generator.
    lemma_tokens_list: list[list[str]] = []
    for i, doc in enumerate(
        nlp.pipe(_iter_truncated_texts(chunks), batch_size=256, n_process=1)
    ):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and not token.is_stop and len(token.lemma_) >= 2
        ]
        lemma_tokens_list.append(tokens)
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            print(f"[Build] spaCy: {i + 1}/{total} chunks " f"({elapsed:.1f}s elapsed) ...")

    print(f"[Build] spaCy lemmatization complete in {time.time() - t0:.1f}s.")

    print("[Build] Starting raw tokenization and metadata collection ...")
    t2 = time.time()
    for i, chunk in enumerate(chunks):
        lemma_tokens = lemma_tokens_list[i]
        raw_tokens   = tokenize_raw(chunk["document"])

        # Both lists must be non-empty to be meaningful; if either is too
        # short we still append (empty list) to maintain positional alignment.
        if len(lemma_tokens) < MIN_TOKENS or len(raw_tokens) < MIN_TOKENS:
            skipped += 1
            corpus_lemma.append([])
            corpus_raw.append([])
        else:
            corpus_lemma.append(lemma_tokens)
            corpus_raw.append(raw_tokens)

        meta.append({
            "parent_id": chunk["metadata"].get("parent_id", ""),
            "child_id":  chunk["id"],
            "source":    chunk["metadata"].get("source", ""),
        })

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t2
            print(f"[Build] Raw tokenization: {i + 1}/{total} chunks " f"({elapsed:.1f}s elapsed) ...")

    print(f"[Build] Raw tokenization complete in {time.time() - t2:.1f}s.")

    if skipped:
        print(f"[Build] Skipped {skipped} chunks below MIN_TOKENS={MIN_TOKENS} " f"(empty lists preserved for alignment).")

    print("[Build] Building BM25Okapi indices ...")
    t1 = time.time()
    bm25_lemma = BM25Okapi(corpus_lemma)
    bm25_raw   = BM25Okapi(corpus_raw)
    print(f"[Build] Indices built in {time.time() - t1:.1f}s.")

    return bm25_lemma, bm25_raw, meta


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_indices(
    bm25_lemma: BM25Okapi,
    bm25_raw:   BM25Okapi,
    meta:       list[dict[str, str]],
) -> None:
    """Serialize the two BM25 indices and the metadata mapping to disk."""
    Path(CHROMA_DB_DIRECTORY).mkdir(parents=True, exist_ok=True)

    artifacts = [
        (bm25_lemma, BM25_LEMMA_PATH, "BM25 lemmatized index"),
        (bm25_raw,   BM25_RAW_PATH,   "BM25 raw index"),
        (meta,       BM25_META_PATH,  "BM25 metadata mapping"),
    ]
    for obj, path, label in artifacts:
        print(f"[Save] Saving {label} to '{path}' ...")
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = Path(path).stat().st_size / (1024 ** 2)
        print(f"[Save] {label} saved ({size_mb:.1f} MB).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  DIEM RAG — BM25 Dual-Index Builder")
    print("=" * 60)
    t_start = time.time()

    chunks = fetch_all_children(CHILD_DB_DIRECTORY, CHILD_COLLECTION_NAME)
    if not chunks:
        raise SystemExit("[BM25] The child collection is empty: run indexer.py first.")

    nlp = _load_spacy()

    bm25_lemma, bm25_raw, meta = build_indices(chunks, nlp)

    save_indices(bm25_lemma, bm25_raw, meta)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("  BM25 Index Construction Complete")
    print(f"  Total chunks indexed : {len(meta)}")
    print(f"  Lemma index path     : {BM25_LEMMA_PATH}")
    print(f"  Raw index path       : {BM25_RAW_PATH}")
    print(f"  Metadata path        : {BM25_META_PATH}")
    print(f"  Total elapsed time   : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()