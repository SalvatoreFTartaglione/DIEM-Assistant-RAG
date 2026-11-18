"""Score-aware context budget allocation and context formatting for generation."""
import numpy as np
from langchain_core.documents import Document

from diem_rag import config

MAX_CONTEXT_CHARS = config.MAX_CONTEXT_CHARS
debug = config.DEBUG


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
