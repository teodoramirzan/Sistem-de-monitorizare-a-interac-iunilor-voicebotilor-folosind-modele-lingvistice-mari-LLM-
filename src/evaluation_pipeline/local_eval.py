from __future__ import annotations

import re
from typing import Any

from .prompting import normalize_turns


INTENT_KEYWORDS = [
    ("block_card", ["blochez", "blocare", "blocat card", "pierdut card", "furat", "card compromis"]),
    ("unblock_card", ["deblochez", "deblocare", "card blocat"]),
    ("open_account", ["deschid un cont", "cont nou", "deschidere cont", "să-mi fac cont"]),
    ("close_account", ["închid cont", "inchid cont", "închidere cont", "renunț la cont"]),
    ("check_balance", ["sold", "câți bani", "cati bani", "balanță", "balanta", "disponibil"]),
    ("get_account_statement", ["extras", "istoric", "tranzacții", "tranzactii", "statement"]),
    (
        "report_suspicious_transaction",
        ["tranzacție suspectă", "tranzactie suspecta", "fraud", "neautorizat", "nu recunosc plata"],
    ),
    ("update_personal_data", ["actualizez", "modific", "schimb adresa", "telefon", "email", "date personale"]),
    ("schedule_advisor_meeting", ["programare", "întâlnire", "intalnire", "consultant", "advisor"]),
    ("reset_or_recover_auth", ["parolă", "parola", "pin", "acces", "autentificare", "resetare"]),
    ("general_product_info", ["comision", "costă", "costa", "dobândă", "dobanda", "produs", "condiții", "conditii"]),
]

OUT_OF_SCOPE = [
    "vremea",
    "meci",
    "film",
    "rețetă",
    "reteta",
    "glumă",
    "gluma",
    "muzică",
    "muzica",
    "politică",
    "politica",
]

RESOLVED = ["am actualizat", "am blocat", "am deblocat", "am programat", "a fost efectuat", "confirmare"]
PARTIAL = ["trebuie să", "trebuie sa", "este necesar", "vă rog", "va rog", "accesați", "accesati"]
REDIRECT = ["operator", "consultant uman", "transfer", "redirecționez", "redirectionez"]
FAILED = ["nu pot", "nu se poate", "eroare", "nu am reușit", "nu am reusit"]
INTERRUPT = ["revin mai târziu", "revin mai tarziu", "închid", "inchid", "pa", "la revedere"]


def _user_text(turns: list[dict[str, str]]) -> str:
    return " ".join(turn["text"].lower() for turn in turns if turn["role"] == "user")


def _assistant_text(turns: list[dict[str, str]]) -> str:
    return " ".join(turn["text"].lower() for turn in turns if turn["role"] == "assistant")


def evaluate_intent(conversation: dict[str, Any] | list[dict[str, str]]) -> dict[str, Any]:
    turns = normalize_turns(conversation)
    text = _user_text(turns)
    if any(word in text for word in OUT_OF_SCOPE):
        return {
            "intent": "fallback",
            "confidence": "high",
            "reasoning": "Conversația conține o solicitare în afara domeniului bancar.",
        }
    for intent, keywords in INTENT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return {
                "intent": intent,
                "confidence": "medium",
                "reasoning": f"Mesajele utilizatorului conțin indicii pentru {intent}.",
            }
    return {
        "intent": "fallback",
        "confidence": "low",
        "reasoning": "Nu există indicii suficiente pentru o intenție bancară suportată.",
    }


def evaluate_final_status(conversation: dict[str, Any] | list[dict[str, str]]) -> dict[str, Any]:
    turns = normalize_turns(conversation)
    all_text = (_user_text(turns) + " " + _assistant_text(turns)).lower()
    last_user = next((t["text"].lower() for t in reversed(turns) if t["role"] == "user"), "")
    assistant = _assistant_text(turns)

    if any(word in last_user for word in INTERRUPT):
        status = "intrerupta"
        reason = "Utilizatorul întrerupe conversația înainte de finalizare."
    elif any(word in assistant for word in REDIRECT):
        status = "redirectionata"
        reason = "Dialogul menționează explicit redirecționarea către operator sau consultant."
    elif any(word in assistant for word in FAILED):
        status = "nerezolvata"
        reason = "Voicebotul indică faptul că nu poate finaliza cererea."
    elif any(word in assistant for word in RESOLVED) and not any(word in assistant for word in PARTIAL):
        status = "rezolvata"
        reason = "Voicebotul confirmă finalizarea cererii în conversație."
    elif any(word in all_text for word in PARTIAL):
        status = "partial_rezolvata"
        reason = "Conversația cere pași suplimentari sau informații înainte de finalizare."
    else:
        status = "partial_rezolvata"
        reason = "Conversația nu conține o confirmare fermă de finalizare."
    return {"final_status": status, "confidence": "medium", "reasoning": reason}


def evaluate_incongruities(conversation: dict[str, Any] | list[dict[str, str]]) -> dict[str, Any]:
    turns = normalize_turns(conversation)
    for idx, turn in enumerate(turns[:-1]):
        if turn["role"] != "user":
            continue
        user = turn["text"].lower()
        nxt = turns[idx + 1]
        if nxt["role"] != "assistant":
            continue
        assistant = nxt["text"].lower()
        if "nu" in user and "despre vreme" in user and ("card" in assistant or "cont" in assistant):
            return {
                "has_incongruity": True,
                "incongruity_type": "nealiniat_context",
                "confidence": "medium",
                "reasoning": "Voicebotul continuă pe o interpretare bancară după o corecție explicită a utilizatorului.",
            }
        user_numbers = re.findall(r"\b\d{2,}\b", user)
        assistant_numbers = re.findall(r"\b\d{2,}\b", assistant)
        if user_numbers and assistant_numbers and not set(user_numbers).intersection(assistant_numbers):
            return {
                "has_incongruity": True,
                "incongruity_type": "nealiniat_context",
                "confidence": "medium",
                "reasoning": "Voicebotul folosește o valoare numerică diferită de cea oferită de utilizator.",
            }
    return {
        "has_incongruity": False,
        "incongruity_type": None,
        "confidence": "medium",
        "reasoning": "Nu apare o neconcordanță explicită în răspunsurile voicebotului.",
    }


def evaluate_locally(task: str, conversation: dict[str, Any] | list[dict[str, str]]) -> dict[str, Any]:
    if task == "intent":
        return evaluate_intent(conversation)
    if task == "final_status":
        return evaluate_final_status(conversation)
    if task == "incongruities":
        return evaluate_incongruities(conversation)
    raise ValueError(f"Task necunoscut: {task}")
