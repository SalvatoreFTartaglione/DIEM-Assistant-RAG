"""Models, vector store, parent store and BM25 indices, loaded once at import time."""
import json
import pickle

import spacy
from langchain_chroma import Chroma

from diem_rag import config

DEVICE = config.DEVICE
print(f"Using device: {DEVICE}")
llm = config.get_llm()
llm_judge = config.get_llm_judge()

# ──────────────────────────────────────────────
# 1. MODEL AND DB INITIALIZATION
# ──────────────────────────────────────────────
embeddings = config.get_embeddings()
reranker = config.get_reranker()

# Connect to the child-chunk collection built by indexer.py.
child_db = Chroma(
    persist_directory=config.CHILD_DB_DIRECTORY,
    embedding_function=embeddings,
    collection_name=config.CHILD_COLLECTION_NAME,
)

# Load the parent store into a plain dict (parent_id -> {page_content, metadata}) for O(1) lookups.
try:
    with open(config.PARENT_STORE_PATH, "r", encoding="utf-8") as f:
        parent_store = json.load(f)
    print(f"Parent store loaded: {len(parent_store)} entries")
except (FileNotFoundError, json.JSONDecodeError) as e:
    raise RuntimeError(
        f"[FATAL] Unable to load {config.PARENT_STORE_PATH}: {e}. Run indexer.py first."
    ) from e

try:
    with open(config.BM25_LEMMA_PATH, "rb") as fh:
        bm25_lemma = pickle.load(fh)
    with open(config.BM25_RAW_PATH, "rb") as fh:
        bm25_raw = pickle.load(fh)
    with open(config.BM25_META_PATH, "rb") as fh:
        bm25_meta: list[dict[str, str]] = pickle.load(fh)
    # spaCy is loaded with the same disabled components used during indexing.
    nlp_bm25 = spacy.load("it_core_news_lg", disable=["parser", "ner", "senter"])
    BM25_AVAILABLE = True
    print(f"[BM25] Indices loaded: {len(bm25_meta)} entries.")
except (FileNotFoundError, OSError) as e:  # missing index files or spaCy model
    bm25_lemma = bm25_raw = bm25_meta = nlp_bm25 = None
    BM25_AVAILABLE = False
    print(f"[BM25] WARNING: BM25 disabled ({e}). Run bm25_indexer.py to enable it.")
