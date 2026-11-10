"""
batch_test.py

Automated batch testing for the DIEM Agentic RAG chatbot.
Runs a list of questions as a single conversation session, captures all
stdout output (including debug traces) for each turn, and saves the full
results to a Markdown file. The resulting log (with DEBUG=True) is the input
format expected by RAGAS.py.

Usage:
    python batch_test.py --set 2
"""

import argparse
import io
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from chatbot import run_turn

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

QUESTIONS_1 = [
    # Professors' tests
    "Quali sono i componenti della commisione paritetica Studenti-Docenti del DIEM?",
    "Descrivi brevemente gli obiettivi dei corsi di dottorato erogato dal DIEM",
    "Con una media di 26.4 a che voto di laurea posso aspirare in ingegneria informatica magistrale?",
    "Quali libri di testo dovrei comprare per seguire Ingegneria del Software?",
    "Quali sono i corsi insegnati da Alessia Saggese?",
    "E il prof. Addeo?",   # tests coreference resolution
    "Quali sono le attività del laboratorio di robotica del DIEM?",
    "Quali attrezzature ci sono nel laboratorio?",
    "Come viene calcolato il voto finale di laurea magistrale di Ingegneria Informatica?",
    "Ho la media del 28, a che voto posso aspirare?",
    "In cosa consiste il programma Horizon Europe della commissione europea?",
    "Quali sono le date dei prossimi appelli di Artificial Intelligence for Cybersecurity?",
    # From project assignment
    "Dove si trova la stanza 126?",
    "Cosa insegna Diego Gragnaniello?",
    "Dove fa ricevimento?",
    "Chi è responsabile per l'internazionallizazione al DIEM?",
    "Quali sono le opportunità di mobilizzazione internazionale disponibili?",
    "Dove si trova il DIEM?",
    "Quali sono le aree di ricerca attive del DIEM?",
    "Quali sono i requisiti di ammissioni per il corso di laurea in ingegneria informatica magistrale?",
    # Personal Tests
    "Cos'è il DIEM?",
    "Quali sono i corsi insegnati da Matteo Gaeta",
    "Quali sono i corsi di Vento del 2024/2025?",
    "Quando sono gli appelli di Artificial Vision?",
    "Qual è l'orario delle lezioni di ingegneria informatica?",


]
QUESTIONS_2 = [
    # ── Dipartimento – struttura e governance ──────────────────────────────
    "Chi compone la Giunta del DIEM?",
    "Quali sono le commissioni interne del DIEM e i rispettivi delegati?",
    "Chi è il delegato all'internazionalizzazione del DIEM?",
    "Qual è la funzione della Commissione Ricerca del DIEM?",
    "In quale edificio e piano si trova il Laboratorio Didattico di Informatica di Base del DIEM?",
    "Quante postazioni studente ha il laboratorio T25 del DIEM?",
    "Quali aree CUN sono rappresentate dai docenti del DIEM?",
    "Qual è l'indicatore ISPD ottenuto dal DIEM nella valutazione ANVUR 2015-2019?",

    # ── Offerta formativa – corsi ──────────────────────────────────────────
    "Da quale anno accademico è attivo il corso di laurea magistrale in Electrical Engineering for Digital Energy?",
    "Qual è la classe di laurea del corso magistrale in Information Engineering for Digital Medicine?",
    "Qual è la classe di laurea del corso magistrale in Electrical Engineering for Digital Energy?",
    "I corsi di laurea in Ingegneria Informatica del DIEM hanno una certificazione di qualità europea? Quale?",
    "Quali sono i due curricula del corso di laurea magistrale in Ingegneria Informatica?",
    "Cos'è il progetto IT4DiMe e a quale programma ministeriale è legato?",
    "Il DIEM gestisce dottorati di ricerca nazionali? Se sì, quali?",
    "In cosa consiste il Dottorato Nazionale in Photovoltaics del DIEM?",

    # ── Ammissione e carriera – domande matematiche / edge case ───────────
    "Ho 59 punti nella prova di ammissione alla magistrale in Ingegneria Informatica. Sono ammesso?",
    "Quali argomenti di Fisica sono previsti nel syllabus della prova di ammissione alla magistrale in Ingegneria Informatica?",
    "Quali argomenti di Matematica sono previsti nel syllabus della prova di ammissione alla magistrale?",
    "Come si contatta la segreteria per partecipare a una sessione della prova di ammissione alla magistrale?",
    "Ho una laurea triennale in Ingegneria con voto 106. Posso iscrivermi alla magistrale in Ingegneria Informatica?",
    "Quanti punti valgono al massimo i titoli della carriera universitaria pregressa nella prova di ammissione alla magistrale?",
    "Ho la media del 24 nella triennale. Ho diritto all'esonero dalla prova di ammissione alla magistrale in Ingegneria Informatica?",
    "Con una media di 29.5 nella triennale, qual è il mio voto base di laurea?",
    "Con una media di 22 nella triennale in Ingegneria Informatica, qual è il voto base di laurea?",

    # ── Internazionalizzazione ─────────────────────────────────────────────
    "Con quale università francese il DIEM ha un accordo per la laurea magistrale a doppio titolo?",
    "Quanti mesi al massimo può durare una mobilità Erasmus+ per studio per un corso magistrale?",
    "Cos'è un Learning Agreement e quando va firmato per la mobilità Erasmus?",
    "Come faccio a candidarmi per il programma Erasmus+ al DIEM?",
    "Quante università estere devo indicare nella domanda Erasmus+ e in quale ordine?",
    "Gli esami sostenuti all'estero durante l'Erasmus vengono riconosciuti nel mio piano di studio?",
    "Cosa sono i Blended Intensive Programmes (BIP) offerti nell'ambito di Erasmus+?",
    "Il DIEM ha accordi di cooperazione internazionale per il dottorato in co-tutela?",

    # ── Ricerca ────────────────────────────────────────────────────────────
    "Quali temi di ricerca sono attivi al DIEM nell'ambito dell'intelligenza artificiale?",
    "Il DIEM svolge ricerca nel campo della guida autonoma?",
    "Cosa studia il gruppo di ricerca del DIEM che si occupa di elaborazione di segnali?",
    "Il DIEM partecipa a dottorati nazionali nel settore dell'intelligenza artificiale applicata alla salute?",

    # ── Inserimento lavorativo ─────────────────────────────────────────────
    "Cos'è il programma di accompagnamento al lavoro previsto dal DIEM per gli studenti triennali?",
    "In quale anno del corso magistrale in Ingegneria Informatica si svolge l'attività di accompagnamento al lavoro?",
    "Quali aziende hanno partecipato alle tavole rotonde di accompagnamento al lavoro del DIEM?",

    "Cos'è il programma EUR-ACE e chi lo gestisce a livello europeo?",
    "Qual è la differenza tra Erasmus+ per studio ed Erasmus+ per traineeship?",

        # ── Terza missione e trasferimento tecnologico ─────────────────────────
    "Cos'è AI4Health e in cosa è specializzata?",                                          # Research, multiple pages
    "A quali programmi europei ha partecipato il DIEM come partner diretto?",              # Research, multiple pages
    "Quanti milioni di euro ha ottenuto il DIEM da progetti europei negli ultimi dieci anni?",  # Research, math
    "In cosa consiste il Dottorato Nazionale in Autonomous Systems al quale partecipa il DIEM?",  # Courses, multiple pages
    "Il DIEM partecipa al Dottorato Nazionale in Learning Sciences and Digital Technologies?",    # Courses

    # ── Bandi e opportunità per studenti ──────────────────────────────────
    "Cos'è il progetto Help Teaching del DIEM e chi può parteciparvi?",                   # Department, bandi
    "Cos'è il bando Swift App Development Bootcamp del DIEM?",                            # Department, bandi
    "Come posso candidarmi per un assegno di tutorato didattico al DIEM?",                # Department, bandi
    "Il DIEM offre tutorato specializzato per studenti con disabilità o DSA?",            # Department, bandi

    # ── Orientamento e carriera studenti ──────────────────────────────────
    "A cosa servono i seminari di orientamento in itinere organizzati dal Consiglio Didattico di Ingegneria Informatica?",  # Teaching
    "Il DIEM organizza seminari per aiutare gli studenti del terzo anno a scegliere la laurea magistrale?",                 # Teaching
    "Come funziona il tutorato didattico per gli studenti del primo e secondo anno di Ingegneria Informatica?",             # Teaching

    # ── Piano di studi e regolamenti ──────────────────────────────────────
    "Entro quando devo presentare la domanda di tesi per la laurea magistrale in Ingegneria Informatica?",  # Regulations, PDF
    "Quali esami posso completare dopo la domanda di laurea e entro quale scadenza?",                        # Regulations, PDF
    "Come posso richiedere il riconoscimento di esami sostenuti in un altro corso di laurea UniSA?",         # Regulations, PDF

    # ── Domande fuorvianti / OOD / storia conversazionale ─────────────────
    "Cosa insegna il prof. Capuano?",                                                    # Faculty, misleading
    "E la prof.ssa Lo Nostro?",                                                          # Faculty, history, OOD
    "Cos'è il programma PNRR e quali sono le sue missioni principali?",                  # OOD
    "Quando sono gli appelli di fondamenti di programmazione?",            # Timetable, date/time, optional
    "Quali sono le date delle prossime sedute di laurea in Ingegneria Informatica magistrale?",  # Timetable, date/time, optional
]
QUESTIONS_3 = [
    "Chi è il delegato all'internazionalizzazione del DIEM?",
    "Qual è la funzione della Commissione Ricerca del DIEM?",
    "Quali aree CUN sono rappresentate dai docenti del DIEM?",
    "Qual è l'indicatore ISPD ottenuto dal DIEM nella valutazione ANVUR 2015-2019?",
    "Cos'è il progetto IT4DiMe e a quale programma ministeriale è legato?",
    "In cosa consiste il Dottorato Nazionale in Photovoltaics del DIEM?",
    "Come faccio a candidarmi per il programma Erasmus+ al DIEM?",
    "Il DIEM partecipa a dottorati nazionali nel settore dell'intelligenza artificiale applicata alla salute?",
    "Quali aziende hanno partecipato alle tavole rotonde di accompagnamento al lavoro del DIEM?",
    ]
QUESTIONS_4 = [
    "Quali sono i membri della commissione paritetica Studenti-Docenti del DIEM?",
    "Descrvi brevemente gli obiettivi dei corsi di dottorato erogati dal DIEM",
    "Ho ottenuto 18 punti al TOLC posso immatricolarmi?",
    "Cosa insegna il prof. Greco?",
    "E il prof Addeo?",
    "Quali sono le atività del laboratorio di robotica del DIEM?",
    "Che attrezzature ci sono nel laboratorio?",
    "Come viene calcolato il voto finale di laurea magistrale in Ingegneria Informatica?",
    "In cosa consiste il programma Horizon Europe della commissione europea?",
    "Quali sono le date dei prossimi appelli di Ingegneria del Software?",
]

QUESTION_SETS = {
    # "1": QUESTIONS_1,   # mixed functional tests (professors, coreference, assignment questions)
    "2": QUESTIONS_2,   # 62-question benchmark used for the RAGAS evaluation
    # "3": QUESTIONS_3,   # short regression subset
    # "4": QUESTIONS_4,   # typos / misleading questions
}

OUTPUT_DIR = Path("./test_results")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _run_turn(question: str, chat_history: list) -> tuple[str, str, str, list]:
    """
    Invoke the chatbot for a single question and return
    (answer, rewritten_question, debug_log, updated_history).

    stdout is captured for the duration of the call so that all debug
    print statements emitted by the graph nodes are included in the log.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        answer, updated_history, final_state = run_turn(question, chat_history)

    rewritten = final_state.get("rewritten_question") or question
    return answer, rewritten, buffer.getvalue(), updated_history


def _format_turn(
    index:      int,
    question:   str,
    rewritten:  str,
    answer:     str,
    debug_log:  str,
    elapsed_s:  float,
) -> str:
    """Render a single conversation turn as a Markdown section."""
    lines = [
        f"## Turn {index}",
        "",
        f"**Question:** {question}",
        "",
    ]

    if rewritten.strip() != question.strip():
        lines += [f"**Rewritten query:** {rewritten}", ""]

    lines += [
        f"**Elapsed:** {elapsed_s:.2f}s",
        "",
        "### Answer",
        "",
        answer,
        "",
        "### Debug log",
        "",
        "```",
        debug_log.rstrip(),
        "```",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of questions through the DIEM chatbot.")
    parser.add_argument("--set", choices=sorted(QUESTION_SETS), default="2",
                        help="question set to run (default: 2, the RAGAS benchmark)")
    args = parser.parse_args()
    questions = QUESTION_SETS[args.set]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"batch_test_{timestamp}.md"

    chat_history: list = []
    sections: list[str] = []

    header = "\n".join([
        "# DIEM Chatbot — Batch Test",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Questions:** {len(questions)}  ",
        "**Session:** single continuous conversation",
        "",
        "---",
        "",
    ])
    sections.append(header)

    print(f"Starting batch test: {len(questions)} questions.")
    print(f"Output: {output_path}\n")
    global_start_time = time.time()
    try:
        for i, question in enumerate(questions, start=1):
            print(f"[{i}/{len(questions)}] {question}")
            t0 = time.time()
            try:
                answer, rewritten, debug_log, chat_history = _run_turn(question, chat_history)
            except Exception as e:
                answer     = f"[ERROR] {e}"
                rewritten  = question
                debug_log  = f"Exception raised:\n{e}"
            elapsed = time.time() - t0
            sections.append(_format_turn(i, question, rewritten, answer, debug_log, elapsed))
            print(f"    Done in {elapsed:.2f}s\n")
            # Incremental save: write to disk after every turn
            # This protects against unexpected hard crashes of the OS
            output_path.write_text("\n".join(sections), encoding="utf-8")

    except KeyboardInterrupt:
        # Catches Ctrl+C from the terminal
        print("\n[WARNING] Batch test interrupted by the user (Ctrl+C).")
    except Exception as e:
        # Catches any fatal outer loop errors
        print(f"\n[FATAL ERROR] Unexpected crash in the test loop: {e}")
    finally:
        # The finally block ALWAYS executes, ensuring the final state is saved
        global_end_time = time.time()
        total_elapsed = global_end_time - global_start_time
        # Document footer with total testing time
        footer = "\n".join([
            "---",
            f"**Batch test ended (or interrupted) in:** {total_elapsed:.2f} seconds",
        ])
        sections.append(footer)
        # Final explicit save guarantees the footer is included
        output_path.write_text("\n".join(sections), encoding="utf-8")
        print(f"\nBatch test finished or interrupted after {total_elapsed:.2f} seconds.")
        print(f"Results safely saved to: {output_path}")


if __name__ == "__main__":
    main()