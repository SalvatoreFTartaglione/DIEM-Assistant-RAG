"""Hybrid retrieval (semantic passes + dual BM25, fused with RRF) and cross-encoder re-ranking."""
import json
import re

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from diem_rag import config
from .resources import (
    BM25_AVAILABLE,
    bm25_lemma,
    bm25_meta,
    bm25_raw,
    child_db,
    llm_judge,
    nlp_bm25,
    parent_store,
    reranker,
)
from .state import AgentState

K_CHILDREN = config.K_CHILDREN
TOP_K = config.TOP_K
debug = config.DEBUG
ENABLE_TITLE_FILTER = config.ENABLE_TITLE_FILTER
K_BM25 = config.K_CHILDREN_BM25
RELATIVE_MULTIPLIER = config.RELATIVE_MULTIPLIER
MIN_ABSOLUTE_THRESHOLD = config.MIN_ABSOLUTE_THRESHOLD
RRF_K = config.RRF_K


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
