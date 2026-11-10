"""
Chainlit web interface for the DIEM Agentic RAG chatbot.

Run with:  chainlit run DIEM_assistant.py
      or:  python DIEM_assistant.py   (starts the same Chainlit server)

This module exposes the LangGraph agent defined in chatbot.py through a
Chainlit chat interface. Each user message is processed by chatbot.run_turn
(the same function used by the terminal chat and by batch_test.py), and a
per-session chat history is kept in the Chainlit user session.

The heavy initialization performed at import time in chatbot.py (embedding
model, vector store connection, BM25 indices, LLM clients, etc.) runs once
when this module is loaded, and is shared across all chat sessions.
"""

import os
import sys

if __name__ == "__main__":
    # `python DIEM_assistant.py` -> start the Chainlit server, which re-imports
    # this file as a module. Done before importing chatbot so the models are
    # loaded only once, inside the server.
    from chainlit.cli import run_chainlit

    run_chainlit(os.path.abspath(__file__))
    sys.exit(0)

import asyncio
import math
import time

import chainlit as cl

import config
from chatbot import run_turn

CHAT_HISTORY_KEY = "chat_history"

# The answer is generated all at once by the agent; it is then revealed
# progressively in the UI. The reveal never takes longer than
# TYPING_EFFECT_MAX_SECONDS, whatever the answer length.
TYPING_EFFECT_DELAY_SECONDS = 0.03
TYPING_EFFECT_MAX_SECONDS = 3.0


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize an empty chat history for the new session."""
    cl.user_session.set(CHAT_HISTORY_KEY, [])


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    """
    Define the starter buttons shown below the input bar at the beginning
    of a new chat. Each starter's label is the short text shown on the
    button; its message is the actual question sent to the agent when the
    button is clicked.
    """
    return [
        cl.Starter(
            label="🔬 Ricerca",
            message="Quali sono i progetti di ricerca attivi al DIEM?",
        ),
        cl.Starter(
            label="🏛️ DIEM",
            message="Informazioni sul DIEM",
        ),
        cl.Starter(
            label="📝 Ammissione",
            message="Requisiti ammissione al corso di laurea Magistrale in Ingegneria Informatica",
        ),
        cl.Starter(
            label="🌍 Erasmus",
            message="Opportunità di internazionalizzazione offerte dal DIEM",
        ),
        cl.Starter(
            label="🎓 Lauree",
            message="Corsi laurea offerti dal DIEM",
        ),
        cl.Starter(
            label="📅 Esami",
            message="Quando si svolgono gli appelli d'esame di Ingegneria Informatica triennale?",
        ),
        cl.Starter(
            label="📚 Corsi",
            message="Quali corsi sono offerti dal DIEM per l'anno accademico 2025/2026?",
        ),
        cl.Starter(
            label="👩‍🏫 Direttore",
            message="chi è il direttore del DIEM?",
        ),
    ]


async def _stream_answer_with_typing_effect(msg: cl.Message, answer: str) -> None:
    """
    Reveal an already-complete answer with a simulated typing effect on an
    existing message (purely visual: the answer is generated in one shot).
    A cursor is shown while the text is being revealed. Words are revealed
    in groups so that the animation never exceeds TYPING_EFFECT_MAX_SECONDS.
    """
    cursor = "▌"
    words = answer.split(" ")
    max_steps = max(1, int(TYPING_EFFECT_MAX_SECONDS / TYPING_EFFECT_DELAY_SECONDS))
    words_per_step = max(1, math.ceil(len(words) / max_steps))

    for end in range(words_per_step, len(words) + words_per_step, words_per_step):
        msg.content = " ".join(words[:end]) + cursor
        await msg.update()
        await asyncio.sleep(TYPING_EFFECT_DELAY_SECONDS)

    msg.content = answer
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """
    Handle an incoming user message: run the LangGraph agent, show a
    loading indicator while waiting, then reveal the answer with a
    simulated typing effect and persist it in the conversation history.
    """
    chat_history = cl.user_session.get(CHAT_HISTORY_KEY, [])

    thinking_msg = cl.Message(content="Sto pensando...")
    await thinking_msg.send()

    try:
        start_time = time.time()
        # The graph is synchronous: run it in a worker thread so the event loop stays responsive.
        answer, chat_history, _ = await cl.make_async(run_turn)(message.content, chat_history)
        if config.DEBUG:
            print(f"[DEBUG] Total generation time: {time.time() - start_time:.2f} seconds")
        # Persist the history only for successful turns.
        cl.user_session.set(CHAT_HISTORY_KEY, chat_history)
    except Exception as exc:
        print(f"[ERROR] Agent invocation failed: {exc}")
        answer = (
            "Si è verificato un errore durante l'elaborazione della "
            "risposta. Riprova tra qualche istante."
        )

    await _stream_answer_with_typing_effect(thinking_msg, answer)
