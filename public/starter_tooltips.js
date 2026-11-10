/**
 * Adds a native browser tooltip (HTML title attribute) to each starter
 * button, showing the full question that will be sent when the button is
 * clicked. Starter buttons are rendered client-side by Chainlit's React
 * frontend after the initial page load, so a MutationObserver is used to
 * detect when they appear in the DOM and attach the tooltip text.
 *
 * The label -> question mapping below must be kept in sync with the
 * starters defined in DIEM_assistant.py (the set_starters function).
 */
(function () {
  const STARTER_TOOLTIPS = {
    "🔬 Ricerca": "Quali sono i progetti di ricerca attivi al DIEM?",
    "🏛️ DIEM": "Informazioni sul DIEM",
    "📝 Ammissione": "Requisiti ammissione al corso di laurea Magistrale in Ingegneria Informatica",
    "🌍 Erasmus": "Opportunità di internazionalizzazione offerte dal DIEM",
    "🎓 Lauree": "Corsi laurea offerti dal DIEM",
    "📅 Esami": "Quando si svolgono gli appelli d'esame di Ingegneria Informatica triennale?",
    "📚 Corsi": "Quali corsi sono offerti dal DIEM per l'anno accademico 2025/2026?",
    "👩‍🏫 Direttore": "Chi è il direttore del DIEM?",
  };

  /**
   * Scan the document for starter buttons and set their title attribute
   * to the corresponding full question, if not already set.
   */
  function applyStarterTooltips() {
    const allElements = document.querySelectorAll("button, [role='button']");

    allElements.forEach((element) => {
      const label = element.textContent.trim();
      const question = STARTER_TOOLTIPS[label];

      if (question && element.getAttribute("title") !== question) {
        element.setAttribute("title", question);
      }
    });
  }

  // Starter buttons are not present at initial page load; observe DOM
  // mutations on the document body to catch them as soon as React renders
  // the starters section.
  const observer = new MutationObserver(applyStarterTooltips);
  observer.observe(document.body, { childList: true, subtree: true });

  // Also run once immediately in case the starters are already present.
  applyStarterTooltips();
})();