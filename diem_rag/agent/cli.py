"""Terminal chat with full debug traces. Run with: python -m diem_rag.agent.cli"""
import time

from diem_rag import config
from .conversation import run_turn
from .resources import llm, llm_judge

K_CHILDREN = config.K_CHILDREN
TOP_K = config.TOP_K
N_QUERIES = config.N_QUERIES
MAX_CONTEXT_CHARS = config.MAX_CONTEXT_CHARS
debug = config.DEBUG
ENABLE_TITLE_FILTER = config.ENABLE_TITLE_FILTER
ENABLE_DOMAIN_CHECK = config.ENABLE_DOMAIN_CHECK
ENABLE_QUERY_REWRITING = config.ENABLE_QUERY_REWRITING
ENABLE_HALLUCINATION_CHECK = config.ENABLE_HALLUCINATION_CHECK


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
