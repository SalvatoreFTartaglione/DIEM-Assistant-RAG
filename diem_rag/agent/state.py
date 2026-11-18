"""LangGraph state definition and chat-history window."""
from typing import List, TypedDict

from diem_rag import config

MAX_HISTORY_TURNS = config.MAX_HISTORY_TURNS


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
