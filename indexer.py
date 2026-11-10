"""
indexer.py

Indexing stage of the DIEM Assistant RAG pipeline (Parent-Child chunking).

    diem_knowledge_base.json
        -> load_documents      (ftfy encoding repair, readable titles, metadata)
        -> split_into_parents  (structure-aware splitting, deterministic SHA-256 ids)
        -> split_into_children (small chunks enriched with a title + metadata header)
        -> children embedded into ChromaDB, parents stored in a JSON key-value store

Children are used for similarity search; at query time each retrieved child is
expanded to its parent (O(1) lookup) so that the LLM receives a wider context.

Supports incremental indexing (only new/changed documents are re-embedded and
documents removed from the knowledge base are purged) and runs a post-indexing
sanity check. Run bm25_indexer.py afterwards to rebuild the lexical indices.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import ftfy
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config



# ========================
# INDEXER CONFIGURATION
# ========================
JSON_PATH              = config.JSON_PATH
CHILD_DB_DIRECTORY     = config.CHILD_DB_DIRECTORY
CHILD_COLLECTION_NAME  = config.CHILD_COLLECTION_NAME
PARENT_STORE_PATH      = config.PARENT_STORE_PATH

CHUNK_CONFIG           = config.CHUNK_CONFIG

EMBEDDING_MODEL   = config.EMBEDDING_MODEL
DEVICE            = config.DEVICE
DEBUG_EXPORT      = config.DEBUG_EXPORT
DEBUG_EXPORT_PATH = config.DEBUG_EXPORT_PATH
DEBUG_SAMPLE_SIZE = config.DEBUG_SAMPLE_SIZE
debug             = config.DEBUG
PARENT_MIN_CHARS  = config.PARENT_MIN_CHARS

# Regex that strips any existing "Document: ...<blank line>" prefix from a
# chunk's text. Used in both split_into_parents and split_into_children to
# guarantee idempotent prefix injection.
_DOC_PREFIX_RE = re.compile(r"^Document:.*?\n\n", re.DOTALL)


def _clean_previous_outputs() -> None:
    """
    Remove persisted indexing outputs so each run starts from a clean slate.
    A timestamped backup of the existing ChromaDB directory is created before
    any deletion so that a previous working index can always be restored.
    """
    chroma_path = Path(config.CHROMA_DB_DIRECTORY)
    backup_path = Path(config.CHROMA_DB_BACKUP_DIRECTORY)

    if chroma_path.exists():
        if backup_path.exists():
            shutil.rmtree(backup_path)
            if debug:
                print(f"    [DEBUG] Removed previous backup: {backup_path}")
        shutil.copytree(chroma_path, backup_path)
        print(f"    [BACKUP] Existing DB backed up to: {backup_path}")
    else:
        print("    [BACKUP] No existing DB found, skipping backup.")

    child_path = Path(CHILD_DB_DIRECTORY)
    if child_path.exists():
        shutil.rmtree(child_path)
        if debug:
            print(f"    [DEBUG] Removed previous child DB directory: {child_path}")

    parent_store_path = Path(PARENT_STORE_PATH)
    if parent_store_path.exists():
        parent_store_path.unlink()
        if debug:
            print(f"    [DEBUG] Removed previous parent store file: {parent_store_path}")

    old_parent_db = getattr(config, "PARENT_DB_DIRECTORY", None)
    if old_parent_db:
        old_path = Path(old_parent_db)
        if old_path.exists():
            shutil.rmtree(old_path)
            if debug:
                print(f"    [DEBUG] Removed legacy parent DB directory: {old_path}")

    indexed_sources_path = Path(config.INDEXED_SOURCES_PATH)
    if indexed_sources_path.exists():
        indexed_sources_path.unlink()
        if debug:
            print(f"    [DEBUG] Removed indexed sources file: {indexed_sources_path}")




class ParentStore:
    """
    Persistent dictionary {parent_id -> Document} saved to disk as JSON.
    Parents do not require embeddings: they are always retrieved by exact
    ID lookup, never by semantic similarity.

    Usage in the indexer:
        store = ParentStore(PARENT_STORE_PATH)
        store.add(parents)
        store.save()

    Usage in the chatbot (at startup, once):
        store = ParentStore.load(PARENT_STORE_PATH)
        doc   = store.get(parent_id)   # O(1)
    """

    def __init__(self, path: str):
        self.path: str = path
        self._store: Dict[str, Document] = {}

    def add(self, parents: List[Document]) -> None:
        """Add (or overwrite) parents into the in-memory dictionary."""
        for p in parents:
            pid = p.metadata.get("parent_id")
            if pid:
                self._store[pid] = p

    def get(self, parent_id: str) -> Document | None:
        """Return the Document corresponding to parent_id, or None if missing."""
        return self._store.get(parent_id)

    def remove_sources(self, sources: set[str]) -> int:
        """Drop every parent whose metadata source is in `sources`. Returns the number removed."""
        stale = [pid for pid, doc in self._store.items() if doc.metadata.get("source") in sources]
        for pid in stale:
            del self._store[pid]
        return len(stale)

    def save(self) -> None:
        """Serialize the dictionary to disk as a record-oriented JSON."""
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            pid: {
                "page_content": doc.page_content,
                "metadata":     doc.metadata,
            }
            for pid, doc in self._store.items()
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ParentStore":
        """Load the ParentStore from disk. Call this ONCE at chatbot startup."""
        store = cls(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        store._store = {
            pid: Document(
                page_content=entry["page_content"],
                metadata=entry["metadata"]
            )
            for pid, entry in data.items()
        }
        print(f"    ParentStore loaded: {len(store._store)} entries from '{path}'")
        return store

    @classmethod
    def load_or_create(cls, path: str) -> "ParentStore":
        """Load the store if it exists on disk, otherwise return an empty one."""
        if Path(path).exists():
            return cls.load(path)
        print(f"    ParentStore not found at '{path}', starting from an empty store.")
        return cls(path)

    def __len__(self) -> int:
        return len(self._store)



# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def _fix_encoding(text: str) -> str:
    """Fix corrupted encoding using ftfy."""
    return ftfy.fix_text(text)


def _content_hash(doc: Document) -> str:
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


def _save_indexed_sources(documents: List[Document], path: str) -> None:
    """
    Persist the mapping {source_url -> content_hash} describing the documents
    currently present in the index. The file is fully rewritten, so sources
    that are no longer in the knowledge base disappear from the state too.
    The hash is computed on the cleaned document text before chunking, so it
    can be compared directly against freshly loaded JSON entries.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        doc.metadata["source"]: _content_hash(doc)
        for doc in documents
        if doc.metadata.get("source")
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)



def _load_indexed_state(path: str) -> Dict[str, str]:
    """
    Load the {source_url -> content_hash} mapping saved by _save_indexed_sources.
    Returns an empty dict if the file does not exist yet.
    """
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_readable_title(source_url: str) -> str:
    """
    Convert a raw URL into a readable title.

    Priority:
        1. Exact manual override (SOURCE_TITLE_OVERRIDES)
        2. Partial prefix match — most specific key wins (longest prefix first)
        to prevent a shorter key (e.g. .../didattica) from shadowing a
        more specific one (e.g. .../didattica/esami).
        3. Automatic reconstruction from path segments
    """
    if not source_url or source_url == "unknown":
        return "Unknown Document"

    # Dynamic rule for Regulations
    if "__regolamenti-cds" in source_url:
        match = re.search(r"/([A-Za-z0-9]+)\.pdf$", source_url, re.IGNORECASE)
        if match:
            course_code = match.group(1).upper()
            course_name = config.COURSE_CODES.get(course_code, f"Codice {course_code}")
            return f"Regolamento Didattico - {course_name}"

    # Exact override
    if source_url in config.SOURCE_TITLE_OVERRIDES:
        return config.SOURCE_TITLE_OVERRIDES[source_url]

    # Partial override longest key first so the most specific
    # entry always wins over a shorter ancestor path.
    base_title = None
    for key in sorted(config.SOURCE_TITLE_OVERRIDES, key=len, reverse=True):
        if source_url.startswith(key):
            base_title = config.SOURCE_TITLE_OVERRIDES[key]
            break

    # Automatic reconstruction from the path
    try:
        path = re.sub(r"https?://[^/]+", "", source_url)
        path = path.split("?")[0].split("#")[0].rstrip("/")
        segments = [s for s in path.split("/") if s]
        clean_segments = [
            s for s in segments
            if not config.URL_NOISE_SEGMENTS.match(s)
            and not s.startswith("__")
            and not re.match(r"^\d{4,}$", s)
        ]
        readable_parts = []
        for seg in clean_segments:
            seg = re.sub(r"\.(pdf|html|htm|php|aspx)$", "", seg, flags=re.IGNORECASE)
            seg = re.sub(r"[-_]+", " ", seg)
            seg = seg.title()
            readable_parts.append(seg)

        if readable_parts:
            path_label = " > ".join(readable_parts)
            return f"{base_title} > {path_label}" if base_title else path_label
        else:
            return base_title or source_url
    except Exception:
        return base_title or source_url


def _make_metadata_context(metadata: dict, title: str = "") -> str:
    """
    Build a compact metadata string injected into every child chunk header.

    The string serves two purposes:
    1. Embedding signal: the model sees structured labels (professor,
        year, section) alongside the raw text, improving recall for queries
        that name these entities explicitly.
    2. Title-filter anchor: the "title" field is the same string used by
        chatbot.py's $contains filter, so its presence in the chunk text
        guarantees the filter can match even when the body text alone would
        not contain the exact title wording.

    Fields injected (in order of priority):
    - Documento  : always present; the human-readable title from metadata
                    (computed by _make_readable_title in load_documents and
                    stored in metadata["title"]).
    - Tipo       : document type (html / pdf); helps queries that
                    implicitly expect one format (e.g. "regolamento" -> pdf).
    - Professore : professor full name, when present.
    - Anno Accademico: academic year for course/teaching pages.
    - Tipo Bando : bando category label.
    - Anno       : calendar year for bandi and course pages.
    - Sezione    : research section (pubblicazioni, progetti, etc.).
    """
    parts = []

    if title and title != "Unknown Document":
        parts.append(f"Documento: {title}")

    # "Tipo" — always injected; helps format-based queries
    doc_type = metadata.get("type", "")
    if doc_type:
        parts.append(f"Tipo: {doc_type}")

    # Conditional fields
    if metadata.get("professor_name"):
        parts.append(f"Professore: {metadata['professor_name']}")
    if metadata.get("anno_accademico"):
        parts.append(f"Anno Accademico: {metadata['anno_accademico']}")
    if metadata.get("bando_label"):
        parts.append(f"Tipo Bando: {metadata['bando_label']}")
    if metadata.get("anno"):
        parts.append(f"Anno: {metadata['anno']}")
    if metadata.get("sezione_ricerca"):
        parts.append(f"Sezione: {metadata['sezione_ricerca']}")
    if metadata.get("easycourse_category"):
        parts.append(f"Categoria: {metadata['easycourse_category']}")

    return "\nMetadati: [" + " | ".join(parts) + "]" if parts else ""



def _sanitise_metadata(metadata: dict) -> dict:
    sanitised = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if k == "title":          # title lives only in page_content, not in ChromaDB metadata
            continue
        if k == "anno":
            sanitised[k] = str(v)
        elif isinstance(v, (str, int, float, bool)):
            sanitised[k] = v
        else:
            sanitised[k] = str(v)
    return sanitised


def load_documents(json_path: str) -> List[Document]:
    """
    Load the knowledge base JSON and convert it into LangChain Documents.
    Applies encoding repair, readable title generation, and metadata enrichment.
    """
    print(f"\n[1/5] Loading data from: {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {json_path}")

    if debug:
        print(f"    [DEBUG] Raw JSON entries: {len(data)}")

    documents     = []
    skipped_empty = 0
    skipped_short = 0

    for entry_index, entry in enumerate(data):
        raw_text = entry.get("text", "").strip()

        if not raw_text:
            skipped_empty += 1
            continue

        clean_text = _fix_encoding(raw_text)

        if len(clean_text) < PARENT_MIN_CHARS:
            skipped_short += 1
            if debug:
                print(f"    [DEBUG] Skipped short record ({len(clean_text)} chars): " f"{entry.get('source', '?')[:80]}")
            continue

        source_url = entry.get("source", "unknown")
        extra_meta = entry.get("metadata", {})
        bando_label = extra_meta.get("bando_label")
        # normalise "anno" to str so ChromaDB always stores
        # it as a string, making $eq filters in chatbot.py reliable.
        # Without this, anno arrives as int from crawl_courses (int(qs_anno))
        # and as str from other paths, causing silent $eq mismatches.
        anno_raw    = extra_meta.get("anno")
        anno        = str(anno_raw) if anno_raw is not None else None
        anno_acc    = extra_meta.get("anno_accademico")
        sezione_ric = extra_meta.get("sezione_ricerca")

        if bando_label and anno:
            readable_title = f"DIEM - {bando_label} {anno}"
        elif anno_acc and "docenti.unisa.it" in source_url:
            readable_title = f"UniSa - Docenti > Corso {anno_acc}"
        elif sezione_ric and "docenti.unisa.it" in source_url:
            readable_title = f"UniSa - Docenti > {sezione_ric.replace('/', ' ').title()}"
        else:
            readable_title = _make_readable_title(source_url)

        doc = Document(
            page_content=clean_text,
            metadata={
                "source":       source_url,
                "title":        readable_title,
                "type":         entry.get("type", "unknown"),
                "record_index": entry_index,
                **extra_meta,
            }
        )
        documents.append(doc)

        # --- LOGGING ---
    type_counts    = Counter(d.metadata.get("type", "unknown") for d in documents)
    has_prof       = sum(1 for d in documents if d.metadata.get("professor_name"))
    has_anno_acc   = sum(1 for d in documents if d.metadata.get("anno_accademico"))
    has_bando      = sum(1 for d in documents if d.metadata.get("bando_label"))
    has_anno       = sum(1 for d in documents if d.metadata.get("anno"))

    print(f"    Loaded {len(documents)} valid documents "f"(skipped: {skipped_empty} empty, {skipped_short} too short).")
    print(f"    Document type breakdown: {dict(type_counts)}")
    print("    Metadata coverage:")
    print(f"      professor_name  : {has_prof}/{len(documents)}")
    print(f"      anno_accademico : {has_anno_acc}/{len(documents)}")
    print(f"      bando_label     : {has_bando}/{len(documents)}")
    print(f"      anno            : {has_anno}/{len(documents)}")

    if debug:
        # Sample documents with professor_name to verify extraction quality
        prof_samples = [d for d in documents if d.metadata.get("professor_name")][:10]
        print("\n    [DEBUG] Sample professor_name values:")
        for s in prof_samples:
            print(f"      source: {s.metadata.get('source', '')[:70]}")
            print(f"      professor_name: {repr(s.metadata.get('professor_name'))}")
            print(f"      anno_accademico: {s.metadata.get('anno_accademico')}")

    return documents


def split_into_parents(documents: List[Document]) -> List[Document]:
    """
    Split documents into Parent Chunks.

    Parent separators privilege Markdown structural boundaries (headings,
    blank lines) over sentence boundaries, producing semantically cohesive
    sections that give the reranker enough context to score relevance.

    """
    print("\n[2/5] Creating Parent Chunks via Structural Splitter...")

    # One splitter per doc_type. Separators prefer structural boundaries
    # (blank lines, newlines) over sentence or character positions.
    parent_splitters: Dict[str, RecursiveCharacterTextSplitter] = {
        doc_type: RecursiveCharacterTextSplitter(
            chunk_size=cfg["parent_size"],
            chunk_overlap=cfg["parent_overlap"],
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,
        )
        for doc_type, cfg in CHUNK_CONFIG.items()
    }

    parents: List[Document] = []
    skipped_short    = 0
    skipped_examples = []   # stores (source, length, preview) of discarded chunks

    for document in documents:
        doc_type = document.metadata.get("type", "html")
        if doc_type not in CHUNK_CONFIG:
            doc_type = "html"

        chunks = parent_splitters[doc_type].split_documents([document])
        for chunk in chunks:
            if len(chunk.page_content.strip()) >= PARENT_MIN_CHARS:
                parents.append(chunk)
            else:
                skipped_short += 1
                if debug and len(skipped_examples) < 10:
                    skipped_examples.append((
                        chunk.metadata.get("source", "?")[:70],
                        len(chunk.page_content.strip()),
                        repr(chunk.page_content.strip()[:80]),
                    ))

    print(f"    [PARENTS] Skipped {skipped_short} chunks shorter than {PARENT_MIN_CHARS} chars.")
    if debug and skipped_examples:
        print(f"    [DEBUG] Sample discarded parent chunks (first {len(skipped_examples)}):")
        for src, length, preview in skipped_examples:
            print(f"      source={src}  len={length}  text={preview}")

    chunk_counter: Dict[str, int] = defaultdict(int)

    for parent in parents:
        source_url     = parent.metadata.get("source", "unknown")
        readable_title = parent.metadata.get("title") or _make_readable_title(source_url)

        # Strip any pre-existing prefix BEFORE hashing so the parent_id
        # depends only on source + position + raw content, not on the
        # (potentially varying) title string.
        raw_text = _DOC_PREFIX_RE.sub("", parent.page_content, count=1)

        chunk_pos = chunk_counter[source_url]
        chunk_counter[source_url] += 1
        hash_input = f"{source_url}::{chunk_pos}::{raw_text}"
        parent.metadata["parent_id"] = hashlib.sha256(
            hash_input.encode("utf-8")
        ).hexdigest()

        # Inject the prefix AFTER the ID has been computed.
        _prefix = f"Document: {readable_title}\n\n"
        if not parent.page_content.startswith(_prefix):
            parent.page_content = f"{_prefix}{raw_text}"


    # Breakdown by type and metadata coverage
    parent_type_counts = Counter(p.metadata.get("type", "unknown") for p in parents)
    parent_has_prof    = sum(1 for p in parents if p.metadata.get("professor_name"))
    parent_has_anno    = sum(1 for p in parents if p.metadata.get("anno_accademico"))

    print(f"    Created {len(parents)} parent chunks (after enrichment).")
    print(f"    Parent type breakdown: {dict(parent_type_counts)}")
    print("    Metadata coverage in parents:")
    print(f"      professor_name  : {parent_has_prof}/{len(parents)}")
    print(f"      anno_accademico : {parent_has_anno}/{len(parents)}")
    _print_chunk_stats(parents, "Parent")

    if debug and parents:
        print(f"    [DEBUG] Sample parent_id : {parents[0].metadata['parent_id']}")
        print(f"    [DEBUG] Sample title str : {parents[0].page_content.splitlines()[0]}")
        # Show 3 parents from professor pages to verify content quality
        prof_parents = [p for p in parents if p.metadata.get("professor_name")][:3]
        if prof_parents:
            print("    [DEBUG] Sample professor parent chunks:")
            for p in prof_parents:
                print(f"      professor_name : {p.metadata.get('professor_name')}")
                print(f"      source         : {p.metadata.get('source', '')[:70]}")
                print(f"      content preview: {repr(p.page_content[:120])}")

    return parents

def split_into_children(parents: List[Document]) -> List[Document]:
    """
    Split each Parent Chunk into smaller Child Chunks.

    - Unknown doc_type values fall back to "html".
    - child_id is hashed on the raw child text before prefix injection.
    - The parent "Document:" prefix is stripped with a regex so that it can
      never end up duplicated inside the enriched child text.
    """
    # Child separators target sentence / clause boundaries, giving the
    # embedding model compact, semantically dense units to encode.
    child_splitters: Dict[str, RecursiveCharacterTextSplitter] = {
        doc_type: RecursiveCharacterTextSplitter(
            chunk_size=cfg["child_size"],
            chunk_overlap=cfg["child_overlap"],
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
            keep_separator=True,
        )
        for doc_type, cfg in CHUNK_CONFIG.items()
    }

    print("\n[3/5] Creating Child Chunks...")

    children: List[Document] = []
    skipped_short_parent = 0

    for parent in parents:
        doc_type = parent.metadata.get("type", "html")
        if doc_type not in CHUNK_CONFIG:
            doc_type = "html"

        cfg            = CHUNK_CONFIG[doc_type]
        child_splitter = child_splitters[doc_type]
        source_url     = parent.metadata.get("source", "unknown")
        readable_title = parent.metadata.get("title") or _make_readable_title(source_url)

        parent_text = _DOC_PREFIX_RE.sub("", parent.page_content, count=1)
        metadata_context = _make_metadata_context(parent.metadata, title=readable_title)

        if len(parent_text) <= cfg["child_size"]:
            # Parent fits in a single child — no splitting needed.
            enriched_text = (
                f"Document: {readable_title}{metadata_context}\nContent: {parent_text}"
            )
            child_id = hashlib.sha256(
                f"{parent.metadata['parent_id']}::c0::{parent_text}".encode("utf-8")
            ).hexdigest()
            child = Document(
                page_content=enriched_text,
                metadata={**parent.metadata, "child_id": child_id, "short_doc": True},
            )
            children.append(child)
            skipped_short_parent += 1
            continue

        _stripped_parent = Document(page_content=parent_text, metadata=parent.metadata)
        child_docs = child_splitter.split_documents([_stripped_parent])

        for child_index, child in enumerate(child_docs):
            raw_child_text = child.page_content
            child_id = hashlib.sha256(
                f"{child.metadata['parent_id']}::c{child_index}::{raw_child_text}".encode("utf-8")
            ).hexdigest()
            enriched_text = (
                f"Document: {readable_title}{metadata_context}\nContent: {raw_child_text}"
            )
            child.page_content         = enriched_text
            child.metadata["child_id"] = child_id
            child.metadata["short_doc"] = False
            children.append(child)

    child_type_counts = Counter(c.metadata.get("type", "unknown") for c in children)
    child_has_prof    = sum(1 for c in children if c.metadata.get("professor_name"))
    child_has_anno    = sum(1 for c in children if c.metadata.get("anno_accademico"))
    child_short_docs  = sum(1 for c in children if c.metadata.get("short_doc"))

    print(f"    Created {len(children)} child chunks.")
    print(f"    Child type breakdown: {dict(child_type_counts)}")
    print("    Metadata coverage in children:")
    print(f"      professor_name  : {child_has_prof}/{len(children)}")
    print(f"      anno_accademico : {child_has_anno}/{len(children)}")
    print(f"      short_doc=True  : {child_short_docs}/{len(children)}")
    _print_chunk_stats(children, "Child")

    if debug:
        print(f"    [DEBUG] Short parents passed as single child: {skipped_short_parent}")
        # Show 3 child chunks from professor pages to verify content and metadata
        prof_children = [c for c in children if c.metadata.get("professor_name")][:3]
        if prof_children:
            print("    [DEBUG] Sample professor child chunks:")
            for c in prof_children:
                print(f"      professor_name : {c.metadata.get('professor_name')}")
                print(f"      anno_accademico: {c.metadata.get('anno_accademico')}")
                print(f"      source         : {c.metadata.get('source', '')[:70]}")
                print(f"      content preview: {repr(c.page_content[:150])}")
        if children:
            print(f"    [DEBUG] Sample child_id : {children[0].metadata.get('child_id')}")
            print(f"    [DEBUG]        parent_id: {children[0].metadata.get('parent_id')}")

    return children

def _print_chunk_stats(chunks: List[Document], label: str) -> None:
    """Print min / max / mean character lengths for a list of chunks."""
    lengths = [len(c.page_content) for c in chunks]
    if not lengths:
        print(f"    [{label}] no chunks produced.")
        return
    print(f"    [{label}] min={min(lengths)} | " f"max={max(lengths)} | " f"mean={int(sum(lengths) / len(lengths))} chars")


def _index_documents_with_progress(
    db:         Chroma,
    documents:  List[Document],
    ids:        List[str],
    label:      str,
    batch_size: int = 500,
) -> None:
    """
    Index documents in batches with progress logging.

    Metadata values are sanitised to ChromaDB-compatible scalar types before
    each batch is submitted (None, lists and other types would make the
    whole batch fail).
    """
    total = len(documents)
    if total == 0:
        return

    for start in range(0, total, batch_size):
        end        = min(start + batch_size, total)
        batch_docs = documents[start:end]
        batch_ids  = ids[start:end]

        for doc in batch_docs:
            doc.metadata = _sanitise_metadata(doc.metadata)

        db.add_documents(documents=batch_docs, ids=batch_ids)

        if debug:
            print(f"    [DEBUG] {label} upserted {end}/{total}")


def export_debug(
    parents:  List[Document],
    children: List[Document],
    path:     str,
    n:        int = 5,
) -> None:
    """Export a sample of parent/child pairs to JSON for visual inspection, including all metadata."""
    print(f"\n[4a] Debug export -> {path}  (sample of {n} parents with their children)")

    # Map parent documents by their parent_id
    parent_map: Dict[str, Document] = {
        p.metadata["parent_id"]: p for p in parents
    }

    # Map child documents to their corresponding parent_id
    child_map: Dict[str, List[Document]] = {}
    for c in children:
        pid = c.metadata.get("parent_id")
        if pid:
            child_map.setdefault(pid, []).append(c)

    # Take only the first 'n' parent IDs for the sample
    sample_pids = list(parent_map.keys())[:n]
    debug_data  = []

    for pid in sample_pids:
        p     = parent_map[pid]
        clist = child_map.get(pid, [])

        debug_data.append({
            "parent_id":      pid,
            # Export the entire metadata dictionary for the parent
            "parent_metadata": p.metadata,
            "parent_length":  len(p.page_content),
            "parent_preview": p.page_content[:2000] + ("..." if len(p.page_content) > 2000 else ""),
            "num_children":   len(clist),
            "children": [
                {
                    "child_id": c.metadata.get("child_id"),
                    # Export the entire metadata dictionary for the child
                    "child_metadata": c.metadata,
                    "length":   len(c.page_content),
                    "preview":  c.page_content[:150] + ("..." if len(c.page_content) > 150 else ""),
                }
                for c in clist
            ],
        })

    # Save the JSON file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    print("    Debug export saved.")


def _open_child_db(embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=CHILD_COLLECTION_NAME,
        persist_directory=CHILD_DB_DIRECTORY,
        embedding_function=embeddings,
    )


def build_vector_stores(
    parents:    List[Document],
    children:   List[Document],
    embeddings: Embeddings,
) -> tuple[Chroma, ParentStore]:
    """Build the child ChromaDB and the on-disk ParentStore."""
    print(f"\n    => Indexing {len(children)} child chunks in '{CHILD_DB_DIRECTORY}'...")
    child_ids = [c.metadata["child_id"] for c in children]
    child_db  = _open_child_db(embeddings)
    _index_documents_with_progress(child_db, children, child_ids, "Child chunks")
    print("    Child DB ready.")

    print(f"\n    => Saving {len(parents)} parent chunks to ParentStore...")
    parent_store = ParentStore(PARENT_STORE_PATH)
    parent_store.add(parents)
    parent_store.save()
    print("    ParentStore ready.")

    return child_db, parent_store


def _run_sanity_check(
    child_db:     Chroma,
    parent_store: ParentStore,
    children:     List[Document],
    parents:      List[Document],
) -> bool:
    """
    Verify that the newly indexed chunks are retrievable and internally
    consistent.  Prints a report and returns False if any check fails.

    Checks performed:
    1. Child count   — ChromaDB collection size matches the number of
                       child chunks passed to the indexer.
    2. Parent count  — ParentStore entry count matches the number of
                       parent chunks.
    3. Spot retrieval — a random sample of child chunks can be fetched
                       from ChromaDB by their child_id.
    4. Parent linkage — every sampled child's parent_id resolves to an
                       entry in the ParentStore.
    5. Prefix check  — every sampled parent chunk starts with the expected
                       'Document: ...' prefix injected by split_into_parents.
    """
    print("\n[SANITY CHECK] Running post-indexing integrity checks...")
    passed = True
    sample_size = min(20, len(children))
    sample = random.sample(children, sample_size) if children else []

    # Check 1 — child count in ChromaDB
    db_count = child_db._collection.count()
    expected_children = len(children)
    # In incremental mode the DB holds more chunks than the current batch.
    if not config.INCREMENTAL_INDEXING and db_count != expected_children:
        print(f"  [FAIL] Child count mismatch: DB has {db_count}, expected {expected_children}.")
        passed = False
    else:
        print(f"  [OK] Child count in ChromaDB : {db_count}")

    # Check 2 — parent count in ParentStore
    expected_parents = len(parents)
    store_count = len(parent_store)
    if not config.INCREMENTAL_INDEXING and store_count != expected_parents:
        print(f"  [FAIL] Parent count mismatch: store has {store_count}, expected {expected_parents}.")
        passed = False
    else:
        print(f"  [OK] Parent count in ParentStore : {store_count}")

    # Check 3 — spot retrieval from ChromaDB by child_id
    retrieval_failures = 0
    for child in sample:
        child_id = child.metadata.get("child_id")
        try:
            result = child_db._collection.get(ids=[child_id])
            if not result or not result.get("ids"):
                retrieval_failures += 1
        except Exception as e:
            print(f"  [FAIL] ChromaDB retrieval error for child_id {child_id[:12]}...: {e}")
            retrieval_failures += 1

    if retrieval_failures:
        print(f"  [FAIL] Spot retrieval: {retrieval_failures}/{sample_size} child chunks not found in ChromaDB.")
        passed = False
    else:
        print(f"  [OK] Spot retrieval: {sample_size}/{sample_size} sampled child chunks found in ChromaDB.")

    # Check 4 — parent linkage
    linkage_failures = 0
    for child in sample:
        parent_id = child.metadata.get("parent_id")
        if parent_store.get(parent_id) is None:
            linkage_failures += 1
            if debug:
                print(f"  [DEBUG] Missing parent_id: {parent_id[:12]}...")

    if linkage_failures:
        print(f"  [FAIL] Parent linkage: {linkage_failures}/{sample_size} children have no matching parent.")
        passed = False
    else:
        print(f"  [OK] Parent linkage: all {sample_size} sampled children resolve to a parent.")

    # Check 5 — Document prefix present in parent chunks
    prefix_failures = 0
    parent_sample = random.sample(parents, min(20, len(parents))) if parents else []
    for parent in parent_sample:
        if not parent.page_content.startswith("Document:"):
            prefix_failures += 1
            if debug:
                print(f"  [DEBUG] Missing prefix in parent: {repr(parent.page_content[:80])}")

    if prefix_failures:
        print(f"  [FAIL] Prefix check: {prefix_failures}/{len(parent_sample)} parent chunks missing 'Document:' prefix.")
        passed = False
    else:
        print(f"  [OK] Prefix check: all {len(parent_sample)} sampled parent chunks have correct prefix.")

    if passed:
        print("[SANITY CHECK] All checks passed.")
    else:
        print("[SANITY CHECK] One or more checks FAILED. Consider restoring from backup:")
        print(f"               {config.CHROMA_DB_BACKUP_DIRECTORY}")

    return passed


# ============================================================
# ENTRY POINT
# ============================================================

def _purge_sources(child_db: Chroma, parent_store: ParentStore, sources: set[str]) -> None:
    """Remove every child (ChromaDB) and parent (ParentStore) belonging to `sources`."""
    if not sources:
        return
    for source in sources:
        child_db._collection.delete(where={"source": source})
    removed_parents = parent_store.remove_sources(sources)
    print(f"[INCREMENTAL] Purged {len(sources)} sources ({removed_parents} parent chunks).")


def build_db_from_json(json_path: str = JSON_PATH) -> tuple[Chroma, ParentStore]:
    print("=" * 60)
    print("  RAG INDEXING -- Parent-Child Chunking")
    print("=" * 60)

    incremental = config.INCREMENTAL_INDEXING

    if not incremental:
        _clean_previous_outputs()

    all_documents = load_documents(json_path)
    documents = all_documents

    if incremental:
        print("\n[INCREMENTAL] Loading existing indexed state...")
        indexed_state = _load_indexed_state(config.INDEXED_SOURCES_PATH)
        print(f"[INCREMENTAL] {len(indexed_state)} source URLs already indexed.")

        new_docs, changed_docs = [], []
        for doc in all_documents:
            source = doc.metadata.get("source", "")
            if source not in indexed_state:
                new_docs.append(doc)
            elif indexed_state[source] != _content_hash(doc):
                changed_docs.append(doc)
        current_sources = {d.metadata.get("source", "") for d in all_documents}
        removed_sources = set(indexed_state) - current_sources

        print(f"[INCREMENTAL] New documents     : {len(new_docs)}")
        print(f"[INCREMENTAL] Changed documents : {len(changed_docs)}")
        print(f"[INCREMENTAL] Removed documents : {len(removed_sources)}")

        child_db     = _open_child_db(config.get_embeddings())
        parent_store = ParentStore.load_or_create(PARENT_STORE_PATH)

        # Changed and removed documents: drop their old chunks first, otherwise
        # stale children would keep being retrieved next to the updated ones.
        stale_sources = removed_sources | {d.metadata["source"] for d in changed_docs}
        _purge_sources(child_db, parent_store, stale_sources)

        documents = new_docs + changed_docs
        if not documents:
            if stale_sources:
                parent_store.save()
                _save_indexed_sources(all_documents, config.INDEXED_SOURCES_PATH)
                print("[INCREMENTAL] Stale documents removed. Re-run bm25_indexer.py.")
            else:
                print("[INCREMENTAL] Nothing to index. DB is up to date.")
            return child_db, parent_store

    # --- chunking (common to both branches) ---
    parents  = split_into_parents(documents)
    children = split_into_children(parents)

    if DEBUG_EXPORT:
        export_debug(parents, children, DEBUG_EXPORT_PATH, n=DEBUG_SAMPLE_SIZE)

    print(f"\n[5/5] Embedding and storing chunks ({EMBEDDING_MODEL} on {DEVICE})...")
    if incremental:
        print(f"\n    => Appending {len(children)} child chunks to existing DB...")
        child_ids = [c.metadata["child_id"] for c in children]
        _index_documents_with_progress(child_db, children, child_ids, "Child chunks")

        parent_store.add(parents)
        parent_store.save()
        print("    ParentStore updated.")
    else:
        child_db, parent_store = build_vector_stores(parents, children, config.get_embeddings())
    _save_indexed_sources(all_documents, config.INDEXED_SOURCES_PATH)

    sanity_ok = _run_sanity_check(child_db, parent_store, children, parents)
    if not sanity_ok:
        print("\n[WARNING] Sanity check failed. The previous DB backup is available at:")
        print(f"          {config.CHROMA_DB_BACKUP_DIRECTORY}")

    print("\n" + "=" * 60)
    print("  OPERATION COMPLETED")
    print(f"  Child DB      => '{CHILD_DB_DIRECTORY}'  [{CHILD_COLLECTION_NAME}]")
    print(f"  Parent Store  => '{PARENT_STORE_PATH}'")
    print(f"  Indexed child chunks : {len(children)}")
    print(f"  Saved parent chunks  : {len(parents)}")
    print(f"  Sanity check         : {'PASSED' if sanity_ok else 'FAILED — restore from backup'}")
    print("  Next step            : python bm25_indexer.py  (rebuild the lexical indices)")
    print("=" * 60)

    return child_db, parent_store


if __name__ == "__main__":
    build_db_from_json()