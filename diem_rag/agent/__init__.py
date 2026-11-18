"""
diem_rag.agent

Agentic RAG chatbot for the DIEM department (University of Salerno), built as a
LangGraph state machine:

    guardrail_input -> rewrite_query -> retrieve (hybrid + rerank) -> generate
          |                                                              |
          +--> out_of_domain_response              (optional) check_hallucination

Retrieval combines multi-pass semantic search on child chunks (ChromaDB, with
LLM-extracted metadata filters), dual BM25 indices and Reciprocal Rank Fusion,
followed by cross-encoder re-ranking and a score-aware context budget.

Modules:
    resources     models, vector store, parent store and BM25 indices (loaded once)
    state         graph state and chat-history window
    prompts       prompt templates
    retrieval     hybrid retrieval and cross-encoder re-ranking
    context       score-aware context budget and context formatting
    nodes         LangGraph nodes and routing functions
    graph         graph construction and compilation
    conversation  run_turn() and history helpers (used by the UI, the CLI and the tests)
    cli           terminal chat

Run `python -m diem_rag.agent.cli` for a terminal chat, or
`chainlit run DIEM_assistant.py` for the web UI. Every optional node can be
toggled from diem_rag/config.py.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
