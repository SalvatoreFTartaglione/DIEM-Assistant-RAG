"""Construction and compilation of the LangGraph workflow (nodes are toggled by config flags)."""
from langgraph.graph import END, StateGraph

from diem_rag import config
from .nodes import (
    check_hallucination,
    generate_answer,
    guardrail_input,
    out_of_domain_response,
    regenerate_answer,
    rewrite_query,
    route_after_domain_check,
    route_after_hallucination_check,
)
from .retrieval import retrieve_and_rerank
from .state import AgentState

N_QUERIES = config.N_QUERIES
ENABLE_DOMAIN_CHECK = config.ENABLE_DOMAIN_CHECK
ENABLE_QUERY_REWRITING = config.ENABLE_QUERY_REWRITING
ENABLE_HALLUCINATION_CHECK = config.ENABLE_HALLUCINATION_CHECK


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
