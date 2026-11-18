"""Conversation helpers shared by the terminal chat, the Chainlit UI and batch_test.py."""
import re

from langchain_core.messages import AIMessage, HumanMessage

from diem_rag import config
from .graph import app
from .state import AgentState

MAX_HISTORY_TURNS = config.MAX_HISTORY_TURNS
MAX_HISTORY_CHARS = config.MAX_HISTORY_CHARS


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
