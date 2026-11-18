"""LangGraph nodes and routing functions (guardrails, rewriting, generation, regeneration)."""
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from diem_rag import config
from .context import build_context
from .prompts import (
    domain_check_prompt,
    generation_prompt,
    hallucination_prompt,
    rewrite_prompt,
    system_template,
)
from .resources import llm, llm_judge
from .state import AgentState, _history_window

MAX_CONTEXT_CHARS = config.MAX_CONTEXT_CHARS
debug = config.DEBUG


# ──────────────────────────────────────────────
# 3. GUARDRAIL IN – Domain Classifier
# ──────────────────────────────────────────────

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
# 7. ANSWER GENERATION
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# 8. GUARDRAIL OUT – hallucination checker
# ──────────────────────────────────────────────
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
