"""
Central configuration for the DIEM Assistant RAG pipeline.

This module only holds plain settings plus a few lazy factory functions
(get_llm, get_llm_judge, get_reranker, get_embeddings). Heavy objects such as
LLM clients, the cross-encoder and the embedding model are created on first
use, so scripts that only need paths or constants (e.g. bm25_indexer.py) do
not pay the cost of loading models they never use.

Secrets are read from the environment (or from a local .env file, if
python-dotenv is installed). See .env.example.
"""

import os
import re
from functools import lru_cache
from typing import Dict

try:  # Optional: load variables from a local .env file.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ============================================================
# PATHS
# ============================================================
JSON_PATH                  = "./diem_knowledge_base.json"
CHROMA_DB_DIRECTORY        = "./diem_chroma_db"
CHROMA_DB_BACKUP_DIRECTORY = "./diem_chroma_db_backup"
CHILD_DB_DIRECTORY         = os.path.join(CHROMA_DB_DIRECTORY, "children")
PARENT_DB_DIRECTORY        = os.path.join(CHROMA_DB_DIRECTORY, "parents")  # legacy, removed on full rebuild
CHILD_COLLECTION_NAME      = "diem_children"
PARENT_STORE_PATH          = os.path.join(CHROMA_DB_DIRECTORY, "parent_store.json")
INDEXED_SOURCES_PATH       = os.path.join(CHROMA_DB_DIRECTORY, "indexed_sources.json")

BM25_LEMMA_PATH = os.path.join(CHROMA_DB_DIRECTORY, "bm25_lemma.pkl")
BM25_RAW_PATH   = os.path.join(CHROMA_DB_DIRECTORY, "bm25_raw.pkl")
BM25_META_PATH  = os.path.join(CHROMA_DB_DIRECTORY, "bm25_meta.pkl")


# ============================================================
# GENERAL
# ============================================================
# Verbose tracing of every pipeline stage (retrieval passes, scores, budget allocation...).
DEBUG = os.getenv("DIEM_DEBUG", "true").lower() in ("1", "true", "yes")

# Embedding model: "embeddinggemma" is served locally by Ollama; any other value
# is treated as a HuggingFace sentence-transformers model id (e.g. "BAAI/bge-m3").
EMBEDDING_MODEL = "embeddinggemma"


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _detect_device()


# ============================================================
# INDEXING (indexer.py)
# ============================================================
INCREMENTAL_INDEXING = True   # True: index only new/changed documents and drop removed ones.

# Parent-child chunk sizes (characters), differentiated by source type.
CHUNK_CONFIG = {
    "html": {
        "parent_size": 1200,
        "parent_overlap": 100,
        "child_size": 400,
        "child_overlap": 50,
    },
    "pdf": {
        "parent_size": 1500,   # larger parents for long legal sections / tables
        "parent_overlap": 175,
        "child_size": 500,     # larger children for dense academic language
        "child_overlap": 80,
    },
}
PARENT_MIN_CHARS = 50          # parent chunks shorter than this are discarded

# Export a sample of parent/child pairs to JSON for manual inspection of chunking quality.
DEBUG_EXPORT      = False
DEBUG_EXPORT_PATH = "./debug_parent_child_chunks.json"
DEBUG_SAMPLE_SIZE = 50         # number of parents (with their children) to export


# ============================================================
# RETRIEVAL & GENERATION (chatbot.py)
# ============================================================
K_CHILDREN        = 125   # child chunks retrieved per semantic pass (before parent expansion)
K_CHILDREN_BM25   = 60    # child chunks retrieved per BM25 index
TOP_K             = 12    # documents kept after cross-encoder re-ranking
MAX_CONTEXT_CHARS = 16000 # character budget for the generation context
MAX_HISTORY_TURNS = 3     # user+assistant pairs kept in memory
MAX_HISTORY_CHARS = 5000  # hard cap on history size (characters)
N_QUERIES         = 1     # query variants generated for retrieval
MIN_FILTERED_DOCS_BEFORE_FALLBACK = 50  # below this pool size a global semantic pass is added
RRF_K             = 60    # Reciprocal Rank Fusion constant
MIN_CHARS_PER_DOC = 700   # minimum budget for low-relevance documents
RERANK_MAX_CHARS  = 1000  # characters of each parent passed to the cross-encoder

# Feature flags: every optional node of the LangGraph agent can be toggled independently.
ENABLE_DOMAIN_CHECK              = True
ENABLE_QUERY_REWRITING           = True
ENABLE_TITLE_FILTER              = True
ENABLE_HALLUCINATION_CHECK       = False
ENABLE_CONTEXT_BUDGET_ALLOCATION = True
MAX_HALLUCINATION_RETRIES        = 1

# Adaptive relevance threshold on reranker scores (disabled by default).
APPLY_RETRIEVAL_THRESHOLD = False
RELATIVE_MULTIPLIER       = 0.15
MIN_ABSOLUTE_THRESHOLD    = 0.004

# Metadata keys the query analyzer is allowed to turn into ChromaDB $eq filters.
METADATA_KEYS = {"bando_label", "anno", "anno_accademico", "professor_name", "sezione_ricerca", "easycourse_category"}


# ============================================================
# MODELS
# ============================================================
# True: use OpenRouter (OpenAI-compatible API). False: use a local Ollama model.
USE_OPENROUTER      = True
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = "openai/gpt-oss-120b"
OLLAMA_MODEL        = "llama3.1"
RERANKER_MODEL      = "BAAI/bge-reranker-v2-m3"


def build_chat_model(temperature: float, max_tokens: int = 2048):
    """Create a new chat model client according to the provider settings above."""
    if USE_OPENROUTER:
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it or add it to a .env file "
                "(see .env.example), or set USE_OPENROUTER = False to use Ollama."
            )
        return ChatOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            model=OPENROUTER_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(model=OLLAMA_MODEL, temperature=temperature, num_predict=max_tokens)


@lru_cache(maxsize=1)
def get_llm():
    """Main LLM, used only for answer generation."""
    return build_chat_model(temperature=0.1)


@lru_cache(maxsize=1)
def get_llm_judge():
    """Auxiliary LLM (domain classification, query rewriting, filter extraction)."""
    return build_chat_model(temperature=0.0)


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL, device=DEVICE)


@lru_cache(maxsize=1)
def get_embeddings():
    """Embedding model shared by the indexer, the chatbot and the RAGAS evaluation."""
    if EMBEDDING_MODEL == "embeddinggemma":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=EMBEDDING_MODEL)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


# ============================================================
# READABLE TITLES (used at indexing time and as retrieval filters)
# ============================================================
# Manual mapping course code -> human-readable title for degree programs.
COURSE_CODES: Dict[str, str] = {
    "06127": "Laurea Triennale in Ingegneria Informatica 2024/25",
    "IE127": "Laurea Triennale in Ingegneria Informatica 2025/26",
    "06128": "Laurea Triennale in Ingegneria dell'Informazione per la Medicina Digitale 2024/25",
    "IE128": "Laurea Triennale in Ingegneria dell'Informazione per la Medicina Digitale 2025/26",
    "06227": "Laurea Magistrale in Ingegneria Informatica 2024/25",
    "06232": "Laurea Magistrale in Information Engineering for Digital Medicine 2024/25",
    "IE232": "Laurea Magistrale in Information Engineering for Digital Medicine 2025/26",
    "06233": "Laurea Magistrale in Electrical Engineering for Digital Energy 2024/25",
    "IE233": "Laurea Magistrale in Electrical Engineering for Digital Energy 2025/26",
}

URL_NOISE_SEGMENTS = re.compile(
    r"^(id|module|row|rescue|page|dettaglio|unisa-rescue-page|\d+)$", re.IGNORECASE
)

# Manual map source_url -> human-readable title.
SOURCE_TITLE_OVERRIDES: Dict[str, str] = {
    # --- MAIN PORTALS & SERVICES ---
    "https://www.diem.unisa.it":    "DIEM - Sito Ufficiale",
    "https://corsi.unisa.it":       "UniSa - Corsi di Studio",
    "https://docenti.unisa.it":     "UniSa - Docenti",
    "https://www.unisa.it":         "UniSa - Sito Ufficiale",
    "https://web.unisa.it":         "UniSa - Portale Web",
    "https://esse3web.unisa.it":    "UniSa - Segreteria Studenti (Esse3)",
    "https://elearning.unisa.it":   "UniSa - Piattaforma E-Learning",
    "https://biblioteche.unisa.it": "UniSa - Sistema Bibliotecario di Ateneo",

    # --- DIEM DEPARTMENT SECTIONS ---
    "https://www.diem.unisa.it/ricerca":                     "DIEM - Ricerca",
    "https://www.diem.unisa.it/ricerca/progetti-finanziati": "DIEM - Progetti di Ricerca Finanziati",
    "https://www.diem.unisa.it/ricerca/laboratori":          "DIEM - Laboratori di Ricerca",
    "https://www.diem.unisa.it/didattica":                   "DIEM - Didattica",
    "https://www.diem.unisa.it/didattica/orientamento":      "DIEM - Orientamento Studenti",
    "https://www.diem.unisa.it/didattica/esami":             "DIEM - Appelli d'Esame",
    "https://www.diem.unisa.it/didattica/lauree":            "DIEM - Sedute di Laurea",
    "https://www.diem.unisa.it/dipartimento":                "DIEM - Dipartimento",
    "https://www.diem.unisa.it/dipartimento/organi":         "DIEM - Organi Dipartimentali",
    "https://www.diem.unisa.it/international":               "DIEM - Internazionalizzazione",
    "https://www.diem.unisa.it/terza-missione":              "DIEM - Terza Missione",
    "https://www.diem.unisa.it/home/bandi":                  "DIEM - Bandi",
    "https://www.diem.unisa.it/home/avvisi":                 "DIEM - Avvisi",
    "https://www.diem.unisa.it/home/news":                   "DIEM - News",
    "https://www.diem.unisa.it/dipartimento/strutture":      "DIEM - Strutture",
    "https://www.diem.unisa.it/dipartimento/personale":      "DIEM - Personale",
    "https://www.diem.unisa.it/home/contatti":               "DIEM - Contatti",

    # --- DEGREE PROGRAMS ---
    "https://corsi.unisa.it/ingegneria-informatica":                               "UniSa - Ingegneria Informatica",
    "https://corsi.unisa.it/ingegneria-informatica-magistrale":                    "UniSa - Ingegneria Informatica Magistrale",
    "https://corsi.unisa.it/electrical-engineering-for-digital-energy":            "UniSa - Electrical Engineering for Digital Energy",
    "https://corsi.unisa.it/ingegneria-dell-informazione-per-la-medicina-digitale":"UniSa - Ingegneria dell'Informazione per la Medicina Digitale",
    "https://corsi.unisa.it/information-Engineering-for-digital-medicine":         "UniSa - Information Engineering for Digital Medicine",
    "https://corsi.unisa.it/photovoltaics":                                        "UniSa - Photovoltaics",
    "https://corsi.unisa.it/ingegneria-dell-informazione":                         "UniSa - Ingegneria dell'Informazione",
    "https://corsi.unisa.it/informatica-e-ingegneria-dell-informazione":           "UniSa - Informatica e Ingegneria dell'Informazione",
    "https://corsi.unisa.it/digital-health-and-bioinformatic-engineering":         "UniSa - Digital Health and Bioinformatic Engineering",
    "https://corsi.unisa.it/ingegneria-elettronica":                               "UniSa - Ingegneria Elettronica",
    "https://corsi.unisa.it/ingegneria-elettronica-magistrale":                    "UniSa - Ingegneria Elettronica Magistrale",
    "https://corsi.unisa.it/matematica":                                           "UniSa - Matematica",
    "https://corsi.unisa.it/matematica-magistrale":                                "UniSa - Matematica Magistrale",

    # --- DOCUMENTS & PDFs ---
    "https://corsi.unisa.it/uploads/rescue/__regolamenti-cds":     "UniSa - Regolamenti Corsi di Studio",
    "http://www.unisa.it/uploads/rescue/__schede-sua":             "UniSa - Schede SUA",
    "https://corsi.unisa.it/uploads/rescue/__almalaurea":          "UniSa - Dati AlmaLaurea",
    "https://corsi.unisa.it/uploads/rescue/__piano-studi-cds":     "UniSa - Piani di Studio",
    "https://corsi.unisa.it/uploads/rescue/__statistiche-corsi":   "UniSa - Statistiche Corsi",
    "https://corsi.unisa.it/uploads/rescue/__orario-lezioni":      "UniSa - Orario Lezioni",
    "https://corsi.unisa.it/uploads/rescue/__calendario-didattico":"UniSa - Calendario Didattico",
    "https://corsi.unisa.it/uploads/rescue/__appelli-esame":       "UniSa - Appelli d'Esame",
    "https://corsi.unisa.it/uploads/rescue/__bando-ammissione":    "UniSa - Bandi di Ammissione",
    "https://easycourse.unisa.it":                                 "UniSa - Orario Lezioni (EasyCourse)",
}
