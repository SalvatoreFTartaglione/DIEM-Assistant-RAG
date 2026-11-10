"""
chatbot.py

Agentic RAG chatbot for the DIEM department (University of Salerno), built as a
LangGraph state machine:

    guardrail_input -> rewrite_query -> retrieve (hybrid + rerank) -> generate
          |                                                              |
          +--> out_of_domain_response              (optional) check_hallucination

Retrieval combines multi-pass semantic search on child chunks (ChromaDB, with
LLM-extracted metadata filters), dual BM25 indices and Reciprocal Rank Fusion,
followed by cross-encoder re-ranking and a score-aware context budget.

Run `python chatbot.py` for a terminal chat, or `chainlit run DIEM_assistant.py`
for the web UI. Every optional node can be toggled from config.py.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import pickle
import re
import time
from datetime import datetime
from typing import List, Literal, TypedDict

import numpy as np
import spacy
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

import config

DEVICE = config.DEVICE
print(f"Using device: {DEVICE}")
K_CHILDREN = config.K_CHILDREN
TOP_K = config.TOP_K
MAX_CONTEXT_CHARS = config.MAX_CONTEXT_CHARS
MAX_HISTORY_TURNS = config.MAX_HISTORY_TURNS
MAX_HISTORY_CHARS = config.MAX_HISTORY_CHARS
N_QUERIES = config.N_QUERIES
debug = config.DEBUG
ENABLE_TITLE_FILTER = config.ENABLE_TITLE_FILTER
ENABLE_DOMAIN_CHECK = config.ENABLE_DOMAIN_CHECK
ENABLE_QUERY_REWRITING = config.ENABLE_QUERY_REWRITING
ENABLE_HALLUCINATION_CHECK = config.ENABLE_HALLUCINATION_CHECK
K_BM25 = config.K_CHILDREN_BM25
RELATIVE_MULTIPLIER = config.RELATIVE_MULTIPLIER
MIN_ABSOLUTE_THRESHOLD = config.MIN_ABSOLUTE_THRESHOLD
RRF_K = config.RRF_K
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


def retrieve_with_parent_context(query: str, child_db: Chroma, parent_store: dict, k_children: int = 20, metadata_filter: dict = None, where_document: dict = None) -> list[Document]:
    """
    Two-stage retrieval without internal fallback.

    Stage 1 -- Execute a single similarity search on the child vector
                store with the exact filters provided by the caller.
    Stage 2 -- Expand retrieved children to parent documents via
                O(1) dictionary lookup in parent_store.
    """
    # ── Stage 1: Child retrieval ──
    kwargs = {}
    if metadata_filter is not None:
        kwargs["filter"] = metadata_filter
    if where_document is not None:
        kwargs["where_document"] = where_document

    child_results = child_db.similarity_search(
        query, k=k_children, **kwargs
    )

    if debug:
        filter_desc = []
        if metadata_filter is not None:
            filter_desc.append("metadata")
        if where_document is not None:
            filter_desc.append("title")
        mode = "+".join(filter_desc) if filter_desc else "semantic-only"
        print(
            f"[retrieval] Search ({mode}): "
            f"{len(child_results)} children found. "
            f"Query: '{query}'"
        )

    # ── Stage 2: Expand children to parent documents ──
    seen_parent_ids: set[str] = set()
    parent_docs: list[Document] = []

    for child in child_results:
        parent_id = child.metadata.get("parent_id")
        if not parent_id or parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)

        entry = parent_store.get(parent_id)

        if entry is not None:
            parent_docs.append(Document(
                page_content=entry["page_content"],
                metadata=entry["metadata"],
            ))
        else:
            if debug:
                print(
                    f"[retrieval] WARNING: parent_id {parent_id[:16]}... "
                    f"not found in parent_store -- using child as fallback"
                )
            parent_docs.append(child)

    if debug:
        print(f"[retrieval] {len(parent_docs)} parent chunks returned")

    return parent_docs


def _tokenize_lemma_query(text: str) -> list[str]:
    """Lemmatized tokenization of a query string using the module-level spaCy model."""
    doc = nlp_bm25(text)
    return [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop and len(token.lemma_) >= 2
    ]


def _tokenize_raw_query(text: str) -> list[str]:
    tokens = re.split(r"[\s\W_]+", text)
    return [t.lower() for t in tokens if len(t) >= 2]


def retrieve_bm25(
    query: str,
    k: int = K_BM25,
) -> list[Document]:
    """
    Retrieve parent documents using dual BM25 indices.

    Queries both the lemmatized and the raw BM25 index, unions the results
    by parent_id, and expands each matching child chunk to its parent document
    via the in-memory parent_store.

    Returns an empty list if BM25 indices are not available.
    """
    if not BM25_AVAILABLE:
        return []

    tokens_lemma = _tokenize_lemma_query(query)
    tokens_raw   = _tokenize_raw_query(query)

    seen_parent_ids: set[str] = set()
    parent_docs: list[Document] = []

    for tokens, index, label in [
        (tokens_lemma, bm25_lemma, "lemma"),
        (tokens_raw,   bm25_raw,   "raw"),
    ]:
        if not tokens:
            continue

        scores  = index.get_scores(tokens)
        if len(scores) <= k:
            top_indices = np.argsort(scores)[::-1].tolist()
        else:
            top_k_partition = np.argpartition(scores, -k)[-k:]
            top_indices = top_k_partition[np.argsort(scores[top_k_partition])[::-1]].tolist()
        hits = 0
        for idx in top_indices:
            if scores[idx] == 0.0:
                # BM25 returns 0 for chunks with no token overlap — skip.
                break
            entry     = bm25_meta[idx]
            parent_id = entry.get("parent_id", "")
            if not parent_id or parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)

            ps_entry = parent_store.get(parent_id)
            if ps_entry is not None:
                parent_docs.append(Document(
                    page_content=ps_entry["page_content"],
                    metadata=ps_entry["metadata"],
                ))
                hits += 1

        if debug:
            print(
                f"[BM25] {label}: {hits} unique parents retrieved "
                f"(tokens: {tokens[:8]}{'...' if len(tokens) > 8 else ''})"
            )

    return parent_docs


def _register_pass(
    docs: list[Document],
    all_docs: dict[str, Document],
    rrf_rankings: dict[str, list[int]],
) -> None:
    """
    Register a ranked list of documents into the shared pool and RRF ranking tracker.
    Each document's position in the list is treated as its rank for this pass.
    Documents already present in all_docs are not overwritten, but their rank
    is always recorded for RRF score computation.
    """
    for rank, doc in enumerate(docs):
        pid = doc.metadata.get("parent_id", doc.page_content[:50])
        if pid not in all_docs:
            all_docs[pid] = doc
        if pid not in rrf_rankings:
            rrf_rankings[pid] = []
        rrf_rankings[pid].append(rank + 1)  # 1-based rank


def allocate_context_budget(
    docs: list[Document],
    scores: list[float],
    total_budget: int,
    min_chars_per_doc: int = 200,
    softmax_temperature: float = 0.1,
    max_docs_hard_cap: int = 10,
) -> list[Document]:
    """
    Truncate documents to fit within total_budget characters, concentrating
    budget on high-scoring documents and excluding low-scoring ones when a
    significant score gap is detected.

    Strategy:
    1. Detect the largest consecutive score gap among the top max_docs_hard_cap
       documents. Documents beyond the gap cutoff are treated as low-relevance
       and receive only min_chars_per_doc. This approach is scale-invariant:
       it works whether scores range from 0.86-1.00 or 0.05-0.50.
    2. If no meaningful gap exists (all scores nearly equal), all documents
       up to max_docs_hard_cap are treated as high-relevance.
    3. Among high-relevance documents, softmax-weighted allocation with low
       temperature amplifies compressed cross-encoder score differences.
    4. Short documents that do not consume their allocation return unused
       budget to longer documents in the same group.
    """
    if not docs:
        return docs

    n = len(docs)
    scores_arr = np.array(scores, dtype=float)

    # ── Find natural relevance boundary using largest consecutive gap ──
    gaps = [
        (scores_arr[i] - scores_arr[i + 1], i + 1)
        for i in range(min(n - 1, max_docs_hard_cap - 1))
    ]

    if gaps:
        max_gap_value, gap_cutoff = max(gaps, key=lambda x: x[0])
        min_gap = 0.05
        if max_gap_value >= min_gap:
            high_indices = list(range(gap_cutoff))
        else:
            high_indices = list(range(min(n, max_docs_hard_cap)))
    else:
        high_indices = list(range(n))

    low_indices = [i for i in range(n) if i not in high_indices]

    # ── Reserve minimum budget for low-relevance documents ──
    low_reserved = min_chars_per_doc * len(low_indices)
    high_budget  = max(0, total_budget - low_reserved)

    # ── Softmax allocation over high-relevance documents ──
    high_scores = scores_arr[high_indices]
    high_scores_shifted = high_scores / softmax_temperature
    high_scores_shifted -= high_scores_shifted.max()
    exp_scores = np.exp(high_scores_shifted)
    weights = exp_scores / exp_scores.sum()

    raw_allocations_high = [
        max(int(w * high_budget), min_chars_per_doc)
        for w in weights
    ]

    # ── Redistribute unused budget from short high-relevance docs ──
    unused = 0
    final_allocations = [0] * n
    needs_extra = []

    for rank, i in enumerate(high_indices):
        alloc = raw_allocations_high[rank]
        doc_len = len(docs[i].page_content)
        if doc_len <= alloc:
            unused += alloc - doc_len
            final_allocations[i] = doc_len
        else:
            final_allocations[i] = alloc
            needs_extra.append((rank, i))

    if unused > 0 and needs_extra:
        extra_weights = np.array([weights[rank] for rank, _ in needs_extra])
        extra_weights /= extra_weights.sum()
        for (rank, i), w in zip(needs_extra, extra_weights):
            final_allocations[i] += int(w * unused)

    # ── Assign minimum budget to low-relevance documents ──
    for i in low_indices:
        final_allocations[i] = min_chars_per_doc

    # ── Truncate ──
    truncated = []
    for i, doc in enumerate(docs):
        alloc = final_allocations[i]
        if len(doc.page_content) <= alloc:
            truncated.append(doc)
        else:
            truncated.append(Document(
                page_content=doc.page_content[:alloc],
                metadata=doc.metadata,
            ))

    if debug:
        total_used = sum(len(d.page_content) for d in truncated)
        print(f"\n[CONTEXT] Budget allocation (total: {total_used}/{total_budget} chars):")
        print(f"[CONTEXT] High-relevance docs: {len(high_indices)} | Low-relevance docs: {len(low_indices)}")
        for i, (doc, score) in enumerate(zip(docs, scores)):
            alloc = final_allocations[i]
            original_len = len(docs[i].page_content)
            source = doc.metadata.get("source", "N/A")
            group = "HIGH" if i in high_indices else "LOW"
            status = "OK" if original_len <= alloc else "TRUNCATED"
            w = weights[high_indices.index(i)] if i in high_indices else 0.0
            print(
                f"  {i+1}. [{group}|{status}] Score: {score:.2f} | "
                f"Weight: {w:.3f} | Alloc: {alloc} chars | "
                f"Original: {original_len} chars | Source: {source}"
            )

    return truncated


# ──────────────────────────────────────────────
# 2. GRAPH STATE
# ──────────────────────────────────────────────
class AgentState(TypedDict):
    question: str                   # original question
    rewritten_question: str         # question rewritten for retrieval
    chat_history: List              # conversation history
    documents: List                 # retrieved documents
    reranker_scores: List           # reranker scores aligned with documents
    answer: str                     # generated answer
    domain_check: str               # "in_domain" | "out_of_domain"
    hallucination_grade: str        # "grounded" | "hallucinated"
    hallucination_retry_count: int  # tracks hallucination regeneration loop


def _history_window(history: list) -> list:
    """Last MAX_HISTORY_TURNS user/assistant pairs of the conversation."""
    return list(history or [])[-(MAX_HISTORY_TURNS * 2):]


def _current_academic_year() -> str:
    """Academic year in the 'YYYY/YYYY+1' form (a new year starts in September)."""
    today = datetime.now()
    start = today.year if today.month >= 9 else today.year - 1
    return f"{start}/{start + 1}"


# ──────────────────────────────────────────────
# 3. GUARDRAIL IN – Domain Classifier
# ──────────────────────────────────────────────
domain_check_system = """You are a domain classifier for a university chatbot dedicated to DIEM (Department of Information Engineering at the University of Salerno)
CRITICAL RULES:
    1. CONTEXT AWARENESS (CRITICAL): Always read the Chat History to resolve pronouns and references (e.g., "il primo della lista", "lui", "questo corso"). If the user refers to a university topic or entity previously mentioned in the history, you must classify it as "in_domain".
    2. PERSON ENTITIES: If the user asks for information about a specific person (by name or via reference from the chat history), you MUST classify it as "in_domain". Assume the individual could be a professor, researcher, or staff member at DIEM.
    3. IN-DOMAIN SCOPE: Classify as "in_domain" if the topic relates to DIEM, its courses, professors, facilities, research, degree programs, exams, schedules, regulations, or UNISA (University of Salerno).
    4. OUT-OF-DOMAIN SCOPE: Classify as "out_of_domain" ONLY if the topic is blatantly unrelated (e.g., recipes, sports, movies, irrelevant general knowledge) AND there is no logical connection to the previous Chat History.

Reply ONLY with one of these two exact labels. Do not add punctuation or any other text:
in_domain
out_of_domain

EXAMPLES OF CLASSIFICATION:
- "Chi è Alessia Saggese?" -> Classification: in_domain
- "Come si cucina la pasta?" -> Classification: out_of_domain
- If history says "I docenti sono Rossi e Bianchi" and user asks "Mi dai informazioni sul primo?" -> Classification: in_domain"""

domain_check_prompt = ChatPromptTemplate.from_messages([
    ("system", domain_check_system),
    ("placeholder", "{chat_history}"),
    ("human", "{question}\n\nClassification:")
])

def guardrail_input(state: AgentState) -> AgentState:
    """
        Input guardrail: detect out-of-domain questions.
        If the question is out of domain, route to a special response node.
    """
    history = _history_window(state.get("chat_history", []))

    chain = domain_check_prompt | llm_judge | StrOutputParser()

    result = chain.invoke({
        "question": state["question"],
        "chat_history": history
    }).strip().lower()

    # Normalize the response (anything mentioning "out" is treated as out of domain).
    if "out" in result:
        domain = "out_of_domain"
    else:
        domain = "in_domain"

    print(f"[GUARDRAIL IN] Classification: {domain}")
    return {**state, "domain_check": domain}


def route_after_domain_check(state: AgentState) -> Literal["rewrite_query", "out_of_domain_response"]:
    """
    Route based on domain check result.
    If in-domain, go to query rewriter. If out-of-domain, go to special response.
    """
    if state["domain_check"] == "out_of_domain":
        return "out_of_domain_response"
    return "rewrite_query"


# ──────────────────────────────────────────────
# 4. QUERY REWRITER
# ──────────────────────────────────────────────
rewrite_system_template = """You are a SYNTACTIC linguistic analyzer. Your task is to prepare the user's question for a precise semantic search.

CRITICAL RULES:
    1. RESOLVE PRONOUNS AND POSSESSIVES (CRITICAL): Check if the question contains Italian pronouns (e.g., "lui", "lei", "loro") or possessive adjectives (e.g., "suo", "sua", "loro"). If it does, you MUST replace them with the explicit name of the person, course, or entity they refer to, extracting this information from the Chat History.
    2. TYPO CORRECTION VS. PROPER NOUNS (CRITICAL):
        - You ARE ALLOWED to silently correct obvious typos in common Italian words (e.g., "Csoa" -> "Cosa", "infromatica" -> "informatica").
        - You MUST NEVER auto-correct, guess, or modify Proper Nouns, surnames, or names of people, even if they look like typos or you don't recognize them. Keep them EXACTLY as typed by the user (e.g., if the user types "professore addeo", keep "addeo". DO NOT change it to "addesso" or any other real name).
    3. CONDITIONAL TIMEFRAME RULE (CRITICAL):
    - PRIORITY EXCLUSIONS: If the question is about general information, entire degree programs (corsi di laurea come Ingegneria Informatica), calculations, Doctorate, graduation grades, admission rules, facilities, Erasmus, regulations, study materials (libri, testi consigliati, materiale didattico), OR departmental bodies, committees, councils, and representatives (commissioni, organi collegiali, rappresentanze studentesche, consigli) EVEN IF it mentions a specific exam, course, or the word "Docenti": DO NOT add any year.
    - GENERAL INCLUSION: If the question is SPECIFICALLY about classes or exams taught by a named professor (insegnamenti, esami di un docente specifico) AND does NOT fall into the exclusions above AND does NOT explicitly mention an academic year, THEN you MUST append " nell'anno accademico {academic_year}" at the end of the query.
    4. TOPIC SHIFT: If the user completely changes the subject, do not drag entities from the previous Chat History into the new question.
    5. LANGUAGE: The final independent question MUST be generated exclusively in Italian.
    6. For question about the final grade of a student where the degree program is not explicitly mentioned (e.g., "Ho una media di 26, a che voto posso aspirare?"),
        you must rewrite the query to explicitly mention the degree program (e.g., "Ho una media di 26, a che voto posso aspirare in ingegneria informatica magistrale?").

Return ONLY the final independent question, nothing else."""

# The academic year is injected once at startup (no other braces in the template).
rewrite_system = rewrite_system_template.replace("{academic_year}", _current_academic_year())

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", rewrite_system),
    ("placeholder", "{chat_history}"),
    ("human", "Follow-up Question: {question}\n\nIndependent Question:")
])


def rewrite_query(state: AgentState) -> AgentState:
    """Rewrite the question using the chat history."""

    history_window = _history_window(state.get("chat_history", []))

    chain = rewrite_prompt | llm_judge | StrOutputParser()
    rewritten = chain.invoke({
        "chat_history": history_window,
        "question": state["question"]
    }).strip()

    print(f"[QUERY REWRITER] Original: '{state['question']}'")
    print(f"[QUERY REWRITER] Rewritten: '{rewritten}'")
    return {**state, "rewritten_question": rewritten}


# ──────────────────────────────────────────────
# 5. RETRIEVAL with re-ranking
# ──────────────────────────────────────────────
# Canonical document titles the query analyzer may use as a "title" filter.
VALID_TITLES_STR = ", ".join(
    f'"{t}"'
    for t in sorted(set(config.COURSE_CODES.values()) | set(config.SOURCE_TITLE_OVERRIDES.values()))
)


def retrieve_and_rerank(state: AgentState, n_queries: int = 3) -> AgentState:
    """Hybrid retrieval (filtered semantic passes + BM25, fused with RRF) followed by cross-encoder re-ranking."""
    query = state.get("rewritten_question") or state["question"]
    valid_titles_str = VALID_TITLES_STR

    # Prompt for generating query variants and metadata filters
    variant_prompt_with_title = f"""You are an expert Information Retrieval assistant optimizing queries for a Vector Database.
    Your goal is to analyze the user request and generate a JSON object containing:

    1. Exactly {n_queries} optimized search query variants in Italian.
    2. A dictionary of metadata filters to strictly apply to the database.

    FILTER SELECTION GUIDE — choose keys based on question topic:
    - Professor courses/exams  → "professor_name" + "anno_accademico"
    - Bandi/dottorati/concorsi/assegni/borse → "bando_label" (NEVER use "anno_accademico" for these)
    - Exam calendar / lesson schedule → "easycourse_category"
    - A specific professor's research publications or projects (pages on docenti.unisa.it) → "sezione_ricerca": "progetti" or "pubblicazioni" (NO "title" filter)
    - Institutional projects, ministerial programs, or department-level initiatives (not tied to a specific professor) → no filters
    - Lab info, lab equipment or physical structures → "title": "DIEM - Strutture > Dipartimento > Strutture" (NEVER "DIEM - Laboratori di Ricerca")
    - Degree programs, regulations, department sections → "title" from the menu below
    - General DIEM info with no clear category → no filters

    ALLOWED FILTER KEYS:
    - "professor_name" (e.g., "Matteo Gaeta", "Mario Vento")
    - "anno_accademico" (e.g., "2025/2026") — ONLY for professor courses, NEVER for bandi
    - "bando_label": one of: "Incarichi di Insegnamento", "Collaborazioni con il Dipartimento", "Personale Tecnico Amministrativo", "Assegni di Ricerca", "Borse di Ricerca", "Borse e Premi", "Altri Bandi", "Dottorati di Ricerca"
    - "anno" (e.g., "2024", "2025") — calendar year for bandi
    - "sezione_ricerca": "pubblicazioni" or "progetti"
    - "easycourse_category": "calendario_esami" or "orario_lezioni"
    - "title": MUST be chosen EXACTLY from this list — [{valid_titles_str}]

    QUERY VARIANT RULES:
    0. Answer exclusively in Italian.
    1. Generate exactly {n_queries} lexically diverse variants (synonyms, different focus, no noise words).
    2. Strip noise: "fammi", "dimmi", "voglio", "tabella", "elenco", "riassunto", "struttura".
    3. Preserve proper nouns and professor names EXACTLY as written. NEVER autocorrect.
    4. Preserve all timeframes (e.g., "2025/2026") intact.
    5. Variants must be semantically equivalent to the original question.

    EXAMPLES:
    User: "Quali sono i corsi di Vento del 2024/2025?"
    JSON: {{"queries": ["corsi Mario Vento 2024/2025", "didattica prof Vento 2024", "insegnamenti Vento 2024/2025"], "filters": {{"professor_name": "Mario Vento", "anno_accademico": "2024/2025"}}}}

    User: "Informazioni sul bando erasmus del 2024"
    JSON: {{"queries": ["bando erasmus DIEM 2024", "opportunità mobilità erasmus 2024"], "filters": {{"anno": "2024"}}}}

    User: "Ho la media del 23.5 e frequento ingegneria informatica magistrale, a che voto posso aspirare?"
    JSON: {{"queries": ["calcolo voto laurea ingegneria informatica magistrale", "formula punteggio finale magistrale informatica"], "filters": {{"title": "Laurea Magistrale in Ingegneria Informatica 2024/25"}}}}

    User: "Quando sono gli appelli di Artificial Vision?"
    JSON: {{"queries": ["appelli esame Artificial Vision", "calendario esami Artificial Vision informatica magistrale", "date esame Artificial Vision DIEM"], "filters": {{"easycourse_category": "calendario_esami"}}}}

    User: "Qual è l'orario delle lezioni di ingegneria informatica?"
    JSON: {{"queries": ["orario lezioni ingegneria informatica", "calendario lezioni DIEM informatica", "orario corsi informatica"], "filters": {{"easycourse_category": "orario_lezioni"}}}}

    OUTPUT FORMAT:
    Return ONLY valid JSON with exactly these two keys: "queries" and "filters".
    The "filters" key MUST always be present, even if empty: {{"queries": [...], "filters": {{}}}}.
    No markdown fences, no explanation, no preamble.

    User Request: {query}
    Output:"""

    # Retrieval prompt without title filter
    variant_prompt_without_title = f"""You are an expert Information Retrieval assistant optimizing queries for a Vector Database.
    Your goal is to analyze the user request and generate a JSON object containing:

    1. Exactly {n_queries} optimized search query variants in Italian.
    2. A dictionary of metadata filters to strictly apply to the database.

    CRITICAL RULE FOR FILTERS:
    You are ONLY allowed to use the following exact keys for the "filters" dictionary. If a piece of information does not fit these keys, DO NOT create a new key.
    - "professor_name" (e.g., "Matteo Gaeta", "Mario Vento", etc.)
    - "anno_accademico" (e.g., "2025/2026", "2024/2025", etc.)
    - "bando_label" (e.g., "Incarichi di Insegnamento", "Collaborazioni con il Dipartimento", "Personale Tecnico Amministrativo", "Assegni di Ricerca", "Borse di Ricerca", "Borse e Premi", "Altri Bandi", "Dottorati di Ricerca", "Incarichi di Insegnamento", "Collaborazioni con il Dipartimento", "Personale Tecnico Amministrativo", "Assegni di Ricerca", "Borse di Ricerca", "Borse e Premi", "Altri Bandi", "Dottorati di Ricerca", "Personale Tecnico Amministrativo", "Assegni di Ricerca", "Collaborazioni con il Dipartimento", "Borse di Ricerca", "Borse e Premi", "Altri Bandi", "Dottorati di Ricerca")
    - "anno" (e.g., "2024", "2025", etc)
    - "sezione_ricerca" ("pubblicazioni", "progetti"), if the question is about RESEARCH AREAS or PROJECTS of professors -> use "sezione_ricerca"
    - "easycourse_category" ("calendario_esami", "orario_lezioni")

    CRITICAL RULES:
    0. answer exclusively in Italian
    1. STRIP THE NOISE: Completely remove instructional/formatting words (e.g., "tabella", "riassunto", "elenco", "fammi", "dimmi", "voglio", "struttura").
    2. PRESERVE ENTITIES: Keep academic entities, course names, professors, departments, and timeframes (e.g., "2025/2026") absolutely intact.
    3. SEMANTIC DIVERSITY: Generate equivalent but lexically distinct queries. Use synonyms, different wordings, or focus on different facets of the entity to maximize vector match probability.
    4. STRICT FORMATTING: Output ONLY the {n_queries} queries, one per line. NO bullet points, NO numbering, NO conversational text.
    5. PRESERVE ENTITIES (CRITICAL): Keep academic entities, course names, departments, and timeframes absolutely intact. You MUST preserve professor names and surnames EXACTLY as received. DO NOT auto-correct names (e.g. keep "addeo" as "addeo").
    6. The rewritten queries must be semantically equivalent to the original question.

    EXAMPLES:
    User: "Quali sono i corsi di Vento del 2024/2025?"
    (assuming {n_queries}=3):
    JSON: {{"queries": ["corsi Mario Vento 2024/2025", "didattica prof Vento 2024", "insegnamenti Vento 2024"], "filters": {{"professor_name": "Mario Vento", "anno_accademico": "2024/2025"}}}}

    User: "Informazioni sul bando erasmus del 2024"
    (assuming {n_queries}=2):
    JSON: {{"queries": ["informazioni bando erasmus 2024", "dettagli mobilità erasmus"], "filters": {{"anno": "2024"}}}}

    User: "Ho la media del 23.5 e frequento ingegneria informatica magistrale, a che voto posso aspirare?"
    JSON: {{"queries": ["calcolo voto laurea ingegneria informatica magistrale", "formula punteggio finale magistrale informatica anno"], "filters": {{}}}}

    User: "Quando sono gli appelli di Artificial Vision?" or "Quali sono le date dei prossimi appelli di Artificial Vision?"
    JSON: {{"queries": ["appelli esame Artificial Vision", "calendario esami Artificial Vision", "date esami Artificial Vision"], "filters": {{"easycourse_category": "calendario_esami"}}}}

    User: "Qual è l'orario delle lezioni di ingegneria informatica?"
    JSON: {{"queries": ["orario lezioni ingegneria informatica", "calendario lezioni DIEM", "orario corsi informatica"], "filters": {{"easycourse_category": "orario_lezioni"}}}}

    OUTPUT FORMAT:
    Return ONLY valid JSON with exactly these two keys: "queries" and "filters".
    The "filters" key MUST always be present, even if empty: {{"queries": [...], "filters": {{}}}}.
    No markdown blocks, no conversational text.

    User Request: {query}
    Output:"""

    if ENABLE_TITLE_FILTER:
        variant_prompt = variant_prompt_with_title
    else:
        variant_prompt = variant_prompt_without_title

    variants_raw = (llm_judge | StrOutputParser()).invoke(variant_prompt)

    # ── Parsing and cleaning ──
    variant_queries  = []
    metadata_filters = {}
    title_contains   = None

    try:
        json_match  = re.search(r'\{[\s\S]*\}', variants_raw)
        parsed_data = (
            json.loads(json_match.group(0))
            if json_match
            else json.loads(variants_raw)
        )

        variant_queries = parsed_data.get("queries") or [query]
        if isinstance(variant_queries, str):
            variant_queries = [variant_queries]
        variant_queries = [str(q).strip() for q in variant_queries if str(q).strip()]
        raw_filters     = parsed_data.get("filters") or {}
        if not isinstance(raw_filters, dict):
            raw_filters = {}

        for filter_key, filter_val in raw_filters.items():
            if not filter_val:
                continue
            clean_val = re.sub(r'\s+', ' ', str(filter_val)).strip()
            if filter_key == "professor_name":
                clean_val = clean_val.title()
            if filter_key == "title":
                if ENABLE_TITLE_FILTER:
                    title_contains = clean_val
                # If ENABLE_TITLE_FILTER is False, the key is ignored.
                continue
            if filter_key in config.METADATA_KEYS:
                metadata_filters[filter_key] = clean_val

    except (json.JSONDecodeError, AttributeError, TypeError):
        if debug:
            print(
                f"[RETRIEVAL ERROR] Failed to parse the query-analysis JSON. "
                f"Raw output:\n{variants_raw}"
            )
            print("Falling back to the original query without filters.")
        variant_queries  = [query]
        metadata_filters = {}
        title_contains   = None

    if not variant_queries:
        variant_queries = [query]

    variant_queries = variant_queries[:n_queries]

    # ── Build ChromaDB filter objects ──
    exact_filters = [
        {k: {"$eq": v}} for k, v in metadata_filters.items()
    ]

    if len(exact_filters) == 0:
        chroma_filter = None
    elif len(exact_filters) == 1:
        chroma_filter = exact_filters[0]
    else:
        chroma_filter = {"$and": exact_filters}
    where_document = ({"$contains": title_contains} if (ENABLE_TITLE_FILTER and title_contains) else None)

    if debug:
        print("\n[RETRIEVAL] Generated Queries for Vector Search:")
        for idx, vq in enumerate(variant_queries):
            print(f"  {idx + 1}: '{vq}'")
        print(f"[RETRIEVAL] Metadata Filter (Strict): {chroma_filter}")
        if ENABLE_TITLE_FILTER:
            print(f"[RETRIEVAL] Document Content Filter (Title): {where_document}")

    # ================================================================
    # Phase 2: Hybrid retrieval with global fallback
    # ================================================================
    all_docs: dict[str, Document] = {}
    rrf_rankings: dict[str, list[int]] = {}  # parent_id -> list of ranks across all passes

    for q in variant_queries:
        # ── Pass A: full filters (metadata + title (if ENABLE_TITLE_FILTER)) ──
        if chroma_filter is not None or where_document is not None:
            if debug:
                print(f"\n[RETRIEVAL] Pass A (full filters): Query: '{q}'")
            docs_a = retrieve_with_parent_context(
                q, child_db, parent_store, K_CHILDREN,
                metadata_filter=chroma_filter,
                where_document=where_document,
            )
            _register_pass(docs_a, all_docs, rrf_rankings)


            # ── Pass B: title bypass (metadata only) ──
            # Executed only when a title filter exists, to counteract
            # potential title hallucinations from the LLM.
            if ENABLE_TITLE_FILTER and where_document is not None and chroma_filter is not None:
                if debug:
                    print(f"\n[RETRIEVAL] Pass B (title bypass): Query: '{q}'")
                docs_b = retrieve_with_parent_context( q, child_db, parent_store, K_CHILDREN, metadata_filter=chroma_filter, where_document=None)
                _register_pass(docs_b, all_docs, rrf_rankings)
        else:
            # No filters generated — direct semantic search
            if debug:
                print(f"\n[RETRIEVAL] Direct semantic search (no filters): Query: '{q}'")
            docs_direct = retrieve_with_parent_context(q, child_db, parent_store, K_CHILDREN, metadata_filter=None, where_document=None)
            _register_pass(docs_direct, all_docs, rrf_rankings)

    # Pass C: Global semantic fallback
    # Activated when the filtered passes returned nothing, or fewer unique
    # parents than config.MIN_FILTERED_DOCS_BEFORE_FALLBACK.
    title_only_mode = (
        ENABLE_TITLE_FILTER
        and chroma_filter is None
        and where_document is not None
    )

    filtered_pool_too_small = 0 < len(all_docs) < config.MIN_FILTERED_DOCS_BEFORE_FALLBACK

    if not all_docs or filtered_pool_too_small:
        if debug:
            if not all_docs:
                reason = "all filtered passes returned 0 results"
            elif title_only_mode:
                reason = "only filter was title (no metadata anchor)"
            else:
                reason = f"filtered pool too small ({len(all_docs)} < {config.MIN_FILTERED_DOCS_BEFORE_FALLBACK})"
            print(
                f"\n[RETRIEVAL] Pass C (global fallback): {reason}. "
                f"Executing pure semantic search."
            )
        for q in variant_queries:
            docs_c = retrieve_with_parent_context(
                q, child_db, parent_store, K_CHILDREN,
                metadata_filter=None,
                where_document=None,
            )
            _register_pass(docs_c, all_docs, rrf_rankings)

    # ================================================================
    # Pass D: BM25 hybrid retrieval (runs unconditionally in parallel
    # with the semantic passes and unions results by parent_id).
    # ================================================================
    if BM25_AVAILABLE:
        bm25_docs_added = 0
        for q in variant_queries:
            bm25_results = retrieve_bm25(q, k=K_BM25)
            prev_size = len(all_docs)
            _register_pass(bm25_results, all_docs, rrf_rankings)
            bm25_docs_added += len(all_docs) - prev_size
        if debug:
            print(
                f"\n[RETRIEVAL] Pass D (BM25): {bm25_docs_added} new parent "
                f"chunks added to pool (total pool: {len(all_docs)})"
            )

    # ── RRF fusion: compute final score and sort pool ──
    def _rrf_score(ranks: list[int], k: int = RRF_K) -> float:
        return sum(1.0 / (k + r) for r in ranks)

    initial_docs = sorted(
        all_docs.values(),
        key=lambda doc: _rrf_score(
            rrf_rankings.get(
                doc.metadata.get("parent_id", doc.page_content[:50]), [9999]
            )
        ),
        reverse=True,
    )

    if debug:
        print(
            f"\n[RETRIEVAL] Found {len(initial_docs)} unique parent "
            f"chunks before reranking (RRF pre-sorted)."
        )

    if not initial_docs:
        print("[RETRIEVAL] No documents found (empty index?).")
        return {**state, "documents": [], "reranker_scores": []}

    # ================================================================
    # Phase 3: Cross-encoder re-ranking and selection
    # ================================================================
    pairs  = [
        [query, doc.page_content[:config.RERANK_MAX_CHARS]] for doc in initial_docs
    ]
    scores = [float(x) for x in reranker.predict(pairs)]
    ranked = sorted(
        zip(initial_docs, scores), key=lambda x: x[1], reverse=True
    )

    max_score         = ranked[0][1]
    threshold         = max_score * RELATIVE_MULTIPLIER
    current_threshold = max(threshold, MIN_ABSOLUTE_THRESHOLD)

    if debug and config.APPLY_RETRIEVAL_THRESHOLD:
            print(f"\n[RETRIEVAL] Max score found: {max_score:.2f}")
            print(
                f"[RETRIEVAL] Applying RELATIVE threshold: "
                f"{current_threshold:.2f} "
                f"({RELATIVE_MULTIPLIER*100:.0f}% of max, "
                f"min {MIN_ABSOLUTE_THRESHOLD})"
            )

    if config.APPLY_RETRIEVAL_THRESHOLD:
        top_docs = [
            doc for doc, score in ranked[:TOP_K]
            if score >= current_threshold
        ]
    else:
        top_docs = [doc for doc, score in ranked[:TOP_K]]

    if debug:
        print(
            f"\n[RETRIEVAL] Kept {len(top_docs)} documents "
            f"out of top {TOP_K}:"
        )
        for i, (doc, score) in enumerate(ranked[:TOP_K]):
            parent_id = doc.metadata.get("parent_id", "N/A")
            source    = doc.metadata.get("source", "N/A")
            if config.APPLY_RETRIEVAL_THRESHOLD:
                status = (
                    "KEPT" if score >= current_threshold
                    else "DISCARDED"
                )
                print(
                    f"  {i+1}. [{status}] Score: {score:.2f} "
                    f"| ID: {parent_id} | Source: {source}"
                )
            else:
                print(
                    f"  {i+1}. Score: {score:.2f} "
                    f"| ID: {parent_id} | Source: {source}"
                )

    top_scores = [
    score for doc, score in ranked[:TOP_K]
    if (not config.APPLY_RETRIEVAL_THRESHOLD or score >= current_threshold)
    ]

    return {**state, "documents": top_docs, "reranker_scores": top_scores}


# ──────────────────────────────────────────────
# 7. ANSWER GENERATION
# ──────────────────────────────────────────────
system_template = r"""You are the official AI assistant of the DIEM - University of Salerno.
Your goal is to provide helpful, clear, and professional answers to students.

CRITICAL INSTRUCTIONS:

    EXCLUSIVE CONTEXT: Answer EXCLUSIVELY using the provided CONTEXT. DO NOT use any external knowledge.
    PROPER NOUNS: For names of people, universities, departments, buildings, and courses, copy them VERBATIM from the CONTEXT. NEVER substitute, translate, or deduce them from memory.
    ENTITY ISOLATION: If the question is about a SPECIFIC person by name, use ONLY the information explicitly linked to that exact person. NEVER mix data from different profiles found in the context.
    SPECIFIC FALLBACK (PEOPLE): If a requested person is not found in the context, state it clearly (in the user's language): "Non sono riuscito a trovare informazioni su [nome] nel database. Ti consiglio di visitare docenti.unisa.it per informazioni aggiornate."
    INLINE CITATION RULE (CRITICAL): Whenever you state a fact, list a course, or provide information from the CONTEXT, you MUST embed the source URL directly into the text as an inline Markdown link.
    - Format: [relevant words](exact_url_from_context)
    - Example GOOD: "Mario Vento tiene il corso di [Machine Learning](https://docenti.unisa.it/...)."
    - Example BAD: "Mario Vento tiene il corso (Fonte: URL)" or "Ecco il link: URL".
    - You must ONLY use the URLs provided in the "URL:" field of the corresponding document in the CONTEXT. NEVER invent or hallucinate links.
    GENERAL FALLBACK: If the answer to a general question cannot be found in the context, clearly state that you do not possess that specific information and suggest contacting the relevant university office.
    NO RAG JARGON: Never use phrases like "Based on the context", "According to the document", or "The database says". Answer naturally.
    LANGUAGE AND FORMAT: Always answer in the same language used by the user. Use markdown (bullet points, bold text) to make complex procedures or lists easy to read.
    UNCERTAINTY: If you are uncertain about ANY detail, omit it rather than guessing.
    AMBIGUITY RULE: If a question depends on a specific degree program (e.g., calculating the graduation grade) and the user has not specified it, DO NOT make assumptions. First, ask the user to specify their degree program.
    MATHEMATICAL RULE: NEVER invent numerical values (such as thesis points, CFUs, or bonus points) to complete a formula. If data is missing, explain the formula to the user and tell them what data is needed.
    MATHEMATICAL FORMATTING RULES:
        - When calculating the "Voto_Base" from formulas ending in "/110", the "/110" denotes the maximum grade scale, NOT a mathematical division.
        YOU MUST NEVER mathematically divide the result by 110.
        Example: If the calculation is 4.1 * 23.2 - 8.8 = 86.32, the base score is 86. DO NOT divide 86.32 by 110. Write the final answer as "86/110".
        - You MUST output all math equations strictly using dollar signs. Follow this exact template:
        For block equations, use double dollar signs on their own lines:
        $$
        Voto\_Base = \frac{{4,1 \times Media - 8,8}}{{110}}
        $$
        For inline equations, use a single dollar sign: $23.5 / 30$.
"""

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("placeholder", "{chat_history}"),
    ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:")
])

def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[DOCUMENTO {idx + 1}]\nURL: {doc.metadata.get('source', '')}\nTESTO: {doc.page_content}"
        for idx, doc in enumerate(docs)
    )


def build_context(docs: list[Document], scores: list[float]) -> str:
    """
    Assemble the generation context. With budget allocation enabled, each
    document is truncated proportionally to its reranker score; otherwise the
    concatenated context is simply cut at MAX_CONTEXT_CHARS.
    """
    if not docs:
        return "No relevant information found in the knowledge base."

    if config.ENABLE_CONTEXT_BUDGET_ALLOCATION and scores and len(scores) == len(docs):
        budgeted_docs = allocate_context_budget(
            docs=docs,
            scores=scores,
            total_budget=MAX_CONTEXT_CHARS,
            min_chars_per_doc=config.MIN_CHARS_PER_DOC,
            softmax_temperature=0.1,
            max_docs_hard_cap=10,
        )
        return _format_context(budgeted_docs)

    raw_context = _format_context(docs)
    if debug and len(raw_context) > MAX_CONTEXT_CHARS:
        print(f"\n[WARNING] Context cut: original: {len(raw_context)} -> Reduced to: {MAX_CONTEXT_CHARS}")
    return raw_context[:MAX_CONTEXT_CHARS]


def generate_answer(state: AgentState) -> AgentState:
    """Generate the final answer from the retrieved context."""
    docs = state.get("documents", [])
    context = build_context(docs, state.get("reranker_scores", []))

    if debug:
        print(f"\n[DEBUG GENERATION] Total characters in context: {len(context)}")
        print(f"[DEBUG GENERATION] Total documents injected: {len(docs)}")

    chain = generation_prompt | llm | StrOutputParser()
    try:
        answer = chain.invoke({
            "context": context,
            "question": state.get("rewritten_question") or state["question"],
            "chat_history": state.get("chat_history", [])
        })
        # Anti-babbling guard: very long outputs were observed to be degenerate loops.
        if len(answer) > 5000:
            print("[WARNING] LLM Babbling detected. Output was too long.")
            answer = "Mi dispiace, ho generato una risposta troppo complessa o imprecisa. Potresti riformulare la tua domanda?"
        return {**state, "answer": answer}
    except Exception as e:
        print(f"[ERROR] Answer generation failed: {e}")
        fallback_answer = "Si è verificato un errore durante la generazione della risposta. Riprova tra qualche istante."
        return {**state, "answer": fallback_answer}


def _clean_llm_output(text: str) -> str:
    """Normalize LaTeX delimiters, links and citation artifacts produced by the LLM."""
    if not text:
        return ""
    # 1. 【...url...】 citation markers -> inline Markdown link
    text = re.sub(r'【[^】]*?(https?://[a-zA-Z0-9_/\-?=&.%]+)[^】]*】?', r' ([Fonte](\1))', text)
    # 2-3. remaining 【...】 markers without a URL
    text = re.sub(r'【[^】]*】', '', text)
    text = text.replace('【', '').replace('】', '')
    # 4. "12†https://..." -> "https://..."
    text = re.sub(r'\b\d+†(https?://)', r'\1', text)
    # 5. \[ \] \( \) LaTeX delimiters -> $$ / $ (rendered by Chainlit)
    text = re.sub(r'\\+\[', '$$', text)
    text = re.sub(r'\\+\]', '$$', text)
    text = re.sub(r'\\+\(', '$', text)
    text = re.sub(r'\\+\)', '$', text)
    # 6. bare "[" / "]" lines used as block-math delimiters
    text = re.sub(r'(?m)^\s*\[\s*$', '$$', text)
    text = re.sub(r'(?m)^\s*\]\s*$', '$$', text)
    # 7. (... \command ...) -> inline math
    text = re.sub(r'\(([^)]*\\[a-zA-Z]+[^)]*)\)', r'$\1$', text)

    return text.strip()


# ──────────────────────────────────────────────
# 8. GUARDRAIL OUT – hallucination checker
# ──────────────────────────────────────────────
hallucination_prompt = ChatPromptTemplate.from_template("""
You are a strict fact-checker for a university RAG system.
Your task is to verify whether the ANSWER introduces factual claims that are NOT supported by the CONTEXT.

IMPORTANT DISTINCTIONS — these are always "grounded", do NOT mark them as hallucinated:
- Admissions of ignorance: "I don't have information about X", "X is not in my database"
- Redirects and suggestions: "I recommend visiting docenti.unisa.it", "please contact the department"
- Formatting and connective language that carries no factual claim

STEP-BY-STEP REASONING (follow this structure internally):
1. List every concrete factual claim in the ANSWER (names, dates, numbers, course titles, roles, locations).
2. For each claim, check if it is explicitly supported by the CONTEXT.
3. If ANY claim is not supported and is not an admission of ignorance or a redirect, conclude "hallucinated".
4. If ALL claims are supported (or are admissions/redirects), conclude "grounded".

Reply ONLY with "grounded" or "hallucinated" on the last line. No other text.

Context Documents:
{context}

Assistant's Answer:
{answer}

Grade:""")

def check_hallucination(state: AgentState) -> AgentState:
    """Output guardrail: check for hallucinations."""
    docs = state.get("documents", [])

    if not docs:
        print("[GUARDRAIL OUT] No source documents — marking as hallucinated (fail-safe).")
        return {**state, "hallucination_grade": "hallucinated"}

    context = "\n\n".join([doc.page_content for doc in docs])

    chain = hallucination_prompt | llm_judge | StrOutputParser()
    try:
        print("\n[GUARDRAIL OUT] Checking for hallucinations...")
        raw_grade = chain.invoke({
            "context": context[:MAX_CONTEXT_CHARS],  # limit for performance
            "answer": state["answer"]
        }).strip().lower()

        # Extract only the last non-empty line to handle chain-of-thought output
        # where the model reasons before emitting the final verdict.
        lines = [l.strip() for l in raw_grade.splitlines() if l.strip()]
        verdict_line = lines[-1] if lines else ""

        hallucination_grade = "hallucinated" if "hallucinated" in verdict_line else "grounded"
        print(f"[GUARDRAIL OUT] Hallucination check: {hallucination_grade}")

        return {**state, "hallucination_grade": hallucination_grade}
    except Exception as e:
        print(f"[ERROR] Hallucination check failed: {e}. Defaulting to 'hallucinated' (fail-safe).")
        return {**state, "hallucination_grade": "hallucinated"}


def route_after_hallucination_check(state: AgentState) -> Literal["end", "regenerate"]:
    """Route after hallucination check: if grounded, end; if hallucinated, decide whether to regenerate based on retry count."""
    hallucination_retry_count = state.get("hallucination_retry_count", 0)
    max_retries = config.MAX_HALLUCINATION_RETRIES
    if state["hallucination_grade"] == "hallucinated" and hallucination_retry_count < max_retries:
        return "regenerate"
    return "end"


# ──────────────────────────────────────────────
# 9. SPECIAL NODES
# ──────────────────────────────────────────────
def out_of_domain_response(state: AgentState) -> AgentState:
    """Answer for out-of-domain questions."""
    answer = ("Sono l'assistente del DIEM e posso rispondere solo a domande sul "
              "Dipartimento di Ingegneria dell'Informazione ed Elettrica e Matematica Applicata "
              "dell'Università degli Studi di Salerno. La tua domanda sembra fuori dal mio "
              "ambito: prova a chiedermi qualcosa sul DIEM!")
    return {**state, "answer": answer}


def regenerate_answer(state: AgentState) -> AgentState:
    """Regenerate the answer more conservatively."""
    if debug:
        print("[REGENERATE] Hallucinated answer; regenerating with stricter instructions...")
        print(f"Hallucinated answer text: {state.get('answer', '')[:1000]}...")  # Print a preview of the hallucinated answer for debugging
    # Same context format as generate_answer, so the inline-citation rule can still be honoured.
    context = build_context(state.get("documents", []), state.get("reranker_scores", []))

    strict_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template + "\n\nEXTRA: Be very conservative. "
        "If you are not 100% sure the information is in the context, "
        "say you don't have that specific information."),
        ("placeholder", "{chat_history}"),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:")
    ])

    chain = strict_prompt | llm | StrOutputParser()

    current_retry_count = state.get("hallucination_retry_count", 0)

    try:
        answer = chain.invoke({
            "context": context,
            "question": state.get("rewritten_question", state["question"]),
            "chat_history": state.get("chat_history", [])
        })
        return {
            **state,
            "answer": answer,
            "hallucination_retry_count": current_retry_count + 1
        }
    except Exception as e:
        print(f"[ERROR] Answer regeneration failed: {e}")
        return {
            **state,
            "answer": "Si è verificato un errore durante la generazione della risposta. Riprova tra qualche istante.",
            "hallucination_retry_count": current_retry_count + 1
        }


# ──────────────────────────────────────────────
# 10. GRAPH CONSTRUCTION (LangGraph)
# ──────────────────────────────────────────────
workflow = StateGraph(AgentState)

# 1. Core nodes (always active)
workflow.add_node("retrieve", lambda state: retrieve_and_rerank(state, n_queries=N_QUERIES))
workflow.add_node("generate", generate_answer)

# 2. Optional query rewriting: when disabled, retrieval uses the raw question.
first_after_guardrail = "rewrite_query" if ENABLE_QUERY_REWRITING else "retrieve"
if ENABLE_QUERY_REWRITING:
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_edge("rewrite_query", "retrieve")

# 3. Optional input guardrail (domain classifier).
if ENABLE_DOMAIN_CHECK:
    workflow.add_node("guardrail_input", guardrail_input)
    workflow.add_node("out_of_domain_response", out_of_domain_response)
    workflow.set_entry_point("guardrail_input")
    workflow.add_conditional_edges(
        "guardrail_input",
        route_after_domain_check,
        {
            "rewrite_query": first_after_guardrail,
            "out_of_domain_response": "out_of_domain_response",
        },
    )
    workflow.add_edge("out_of_domain_response", END)
else:
    workflow.set_entry_point(first_after_guardrail)

# 4. Retrieval always feeds generation.
workflow.add_edge("retrieve", "generate")

# 5. Optional output guardrail (hallucination check with bounded regeneration).
if ENABLE_HALLUCINATION_CHECK:
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("regenerate", regenerate_answer)

    workflow.add_edge("generate", "check_hallucination")
    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {
            "end": END,
            "regenerate": "regenerate"
        }
    )
    workflow.add_edge("regenerate", "check_hallucination")
else:
    # If hallucination check is disabled, the generation is the final step
    workflow.add_edge("generate", END)

# Compile the graph.
app = workflow.compile()


# ──────────────────────────────────────────────
# 11. CONVERSATION HELPERS (shared by the CLI, the Chainlit UI and batch_test.py)
# ──────────────────────────────────────────────
def new_state(question: str, chat_history: list) -> AgentState:
    """Initial graph state for a user turn."""
    return AgentState(
        question=question,
        rewritten_question=question,
        chat_history=list(chat_history),
        documents=[],
        reranker_scores=[],
        answer="",
        domain_check="",
        hallucination_grade="",
        hallucination_retry_count=0,
    )


def truncate_history(chat_history: list) -> list:
    """
    Keep at most MAX_HISTORY_TURNS user/assistant pairs, then drop the oldest
    pairs until the history fits in MAX_HISTORY_CHARS characters.
    """
    chat_history = list(chat_history)[-(MAX_HISTORY_TURNS * 2):]
    while sum(len(m.content) for m in chat_history) > MAX_HISTORY_CHARS and len(chat_history) > 2:
        chat_history = chat_history[2:]
    return chat_history


def run_turn(question: str, chat_history: list) -> tuple[str, list, dict]:
    """
    Run the agent on one user message.

    Returns (clean_answer, updated_history, final_state).
    """
    final_state = app.invoke(new_state(question, chat_history))
    answer = _clean_llm_output(final_state["answer"])
    updated_history = truncate_history(
        [*chat_history, HumanMessage(content=question), AIMessage(content=answer)]
    )
    return answer, updated_history, final_state


# ──────────────────────────────────────────────
# 12. MAIN LOOP (terminal chat)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("--- DIEM Agentic RAG Chatbot ---")
    main_model_name = getattr(llm, "model_name", getattr(llm, "model", "Unknown"))
    judge_model_name = getattr(llm_judge, "model_name", getattr(llm_judge, "model", "Unknown"))
    print(f"[SYSTEM] Main LLM: {main_model_name} | Judge LLM: {judge_model_name}")
    print(f"[SYSTEM] K_CHILDREN={K_CHILDREN} TOP_K={TOP_K} N_QUERIES={N_QUERIES} "
          f"MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS}")
    print(f"[SYSTEM] DOMAIN_CHECK={ENABLE_DOMAIN_CHECK} QUERY_REWRITING={ENABLE_QUERY_REWRITING} "
          f"TITLE_FILTER={ENABLE_TITLE_FILTER} HALLUCINATION_CHECK={ENABLE_HALLUCINATION_CHECK} "
          f"BUDGET_ALLOCATION={config.ENABLE_CONTEXT_BUDGET_ALLOCATION} "
          f"RETRIEVAL_THRESHOLD={config.APPLY_RETRIEVAL_THRESHOLD}")
    print("Type 'exit' to quit.\n")

    chat_history = []
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break

        print("-" * 50)
        start_time = time.time()
        answer, chat_history, _ = run_turn(user_input, chat_history)
        print("-" * 50)

        if debug:
            print(f"\n[DEBUG] Total generation time: {time.time() - start_time:.2f} seconds")
        print(f"\nBot: {answer}\n")
