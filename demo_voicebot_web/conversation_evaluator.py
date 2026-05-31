import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


Turn = Dict[str, str]


INTENT_LABELS = [
    "blocare_card",
    "deblocare_card",
    "deschidere_cont",
    "inchidere_cont",
    "verificare_sold",
    "extras_de_cont",
    "tranzactie_suspicioasa",
    "actualizare_date_personale",
    "programare_consultant",
    "resetare_autentificare",
    "informatii_produse",
    "fallback",
]

FINAL_STATUS_LABELS = [
    "rezolvata",
    "partial_rezolvata",
    "nerezolvata",
    "redirectionata",
    "intrerupta",
]

INCONGRUITY_LABELS = [
    "incomplet",
    "irelevant",
    "contradictoriu",
    "nealiniat_context",
    "halucinatie",
]


@dataclass
class EvaluationResult:
    intent: Dict[str, Any]
    final_status: Dict[str, Any]
    incongruities: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "final_status": self.final_status,
            "incongruities": self.incongruities,
        }


class ConversationEvaluator:
    """Evaluator local pentru demo.

    In repo-ul dizertatiei taskurile sunt evaluate cu LLM-uri si prompturi.
    Pentru demo, acest evaluator pastreaza aceeasi suprafata de output, dar
    poate rula offline, fara chei API. Regulile sunt intentionat transparente.
    """

    def evaluate(self, conversation: List[Turn]) -> EvaluationResult:
        return EvaluationResult(
            intent=self.extract_intent(conversation),
            final_status=self.classify_final_status(conversation),
            incongruities=self.detect_incongruities(conversation),
        )

    def extract_intent(self, conversation: List[Turn]) -> Dict[str, Any]:
        user_text = " ".join(turn["text"] for turn in conversation if turn.get("role") == "user")
        normalized = _normalize(user_text)

        if _has_any(normalized, ["vreme", "restaurant", "gluma", "reteta", "taxi", "hotel"]):
            return _result("intent", "fallback", "high", "Utilizatorul a inclus o solicitare in afara domeniului bancar.")

        patterns = [
            ("deblocare_card", ["debloc", "deblochez", "card blocat"]),
            ("blocare_card", ["blochez", "blocare card", "pierdut", "furat", "retinut", "compromis", "mi au luat cardul"]),
            (
                "deschidere_cont",
                [
                    "deschid un cont",
                    "deschizi un cont",
                    "deschis un cont",
                    "deschid cont",
                    "cont nou",
                    "client nou",
                    "vreau cont",
                    "deschidere cont",
                ],
            ),
            ("inchidere_cont", ["inchid cont", "inchidere cont", "renunt la cont"]),
            ("verificare_sold", ["sold", "cati bani", "cati lei", "bani am in cont", "suma disponibila", "balanta"]),
            ("extras_de_cont", ["extras", "istoric tranzact", "tranzactiile", "statement"]),
            ("tranzactie_suspicioasa", ["tranzactie suspecta", "tranzactie ciudata", "raportez", "frauda", "frauduloasa", "neautorizata", "nu recunosc"]),
            (
                "actualizare_date_personale",
                [
                    "actualizez",
                    "actualizare",
                    "modific adresa",
                    "modific telefon",
                    "modific numarul",
                    "modific email",
                    "schimb adresa",
                    "schimb telefon",
                    "schimb numarul",
                    "schimb email",
                    "date personale",
                    "email nou",
                    "noua adresa",
                    "noul numar",
                ],
            ),
            ("programare_consultant", ["consultant", "programare", "intalnire", "discutie cu un consilier"]),
            ("resetare_autentificare", ["reset", "parola", "pin", "nu pot intra", "autentificare", "acces aplicatie"]),
            ("informatii_produse", ["cat costa", "cost", "pret", "taxa", "comision", "dobanda", "conditii", "informatii", "aflu mai multe", "conturi de economii", "produs", "card de credit"]),
        ]

        for label, keywords in patterns:
            if _has_any(normalized, keywords):
                return _result("intent", label, "high", f"Solicitarea utilizatorului corespunde intentiei {label}.")

        return _result("intent", "fallback", "medium", "Solicitarea nu se incadreaza clar in intentiile bancare suportate.")

    def classify_final_status(self, conversation: List[Turn]) -> Dict[str, Any]:
        assistant_text = " ".join(turn["text"] for turn in conversation if turn.get("role") == "assistant")
        user_text = " ".join(turn["text"] for turn in conversation if turn.get("role") == "user")
        full_text = _normalize(f"{assistant_text} {user_text}")

        if _has_any(full_text, ["transfer catre un operator", "operator uman", "redirectionez", "te transfer"]):
            return _status("redirectionata", "high", "Conversatia mentioneaza explicit transferul sau redirectionarea catre un operator.")

        interruption_turn = _last_user_interruption_turn(conversation)
        resolution_turn = _last_assistant_resolution_turn(conversation)
        if interruption_turn and interruption_turn > resolution_turn:
            return _status("intrerupta", "high", "Utilizatorul a intrerupt dialogul inainte de o rezolutie explicita.")

        if _has_any(full_text, ["nu pot finaliza", "nu am reusit", "eroare", "nu se poate procesa", "nu putem continua"]):
            return _status("nerezolvata", "high", "Voicebotul indica explicit ca solicitarea nu a putut fi finalizata.")

        if _has_any(full_text, ["trebuie sa mergeti", "semnati", "in aplicatie", "la sucursala", "link de resetare", "pas suplimentar"]):
            return _status("partial_rezolvata", "medium", "Voicebotul ofera un pas urmator, dar utilizatorul mai are actiuni de facut dupa apel.")

        if _has_any(
            full_text,
            [
                "am blocat",
                "a fost blocat",
                "am deblocat",
                "a fost deblocat",
                "am actualizat",
                "am programat",
                "am trimis",
                "a fost confirmata",
                "am inregistrat",
                "soldul disponibil",
            ],
        ):
            return _status("rezolvata", "high", "Voicebotul confirma finalizarea cererii in cadrul conversatiei.")

        return _status("partial_rezolvata", "low", "Conversatia are raspunsuri relevante, dar nu contine o confirmare ferma de finalizare.")

    def detect_incongruities(self, conversation: List[Turn]) -> Dict[str, Any]:
        turns = list(conversation)
        assistant_turns = [(idx, turn["text"]) for idx, turn in enumerate(turns, start=1) if turn.get("role") == "assistant"]

        hallucination = self._detect_hallucination(assistant_turns)
        if hallucination:
            return hallucination

        misalignment = self._detect_context_misalignment(turns)
        if misalignment:
            return misalignment

        ignored_correction = self._detect_ignored_user_correction(turns)
        if ignored_correction:
            return ignored_correction

        contradiction = self._detect_contradiction(assistant_turns)
        if contradiction:
            return contradiction

        incomplete = self._detect_incomplete_answer(turns)
        if incomplete:
            return incomplete

        return {
            "has_incongruity": False,
            "incongruity_type": None,
            "confidence": "medium",
            "reasoning": "Raspunsurile voicebotului sunt relevante si nu contin o neconcordanta explicita conform regulilor.",
        }

    def _detect_hallucination(self, assistant_turns: Iterable[tuple[int, str]]) -> Optional[Dict[str, Any]]:
        suspicious_patterns = [
            r"penalizare",
            r"amenda",
            r"garantat in \d+ secunde",
            r"contul va fi sters definitiv",
        ]
        for idx, text in assistant_turns:
            normalized = _normalize(text)
            if any(re.search(pattern, normalized) for pattern in suspicious_patterns):
                return _inc("halucinatie", "medium", f"ASSISTANT turn {idx} introduce o suma, penalizare sau consecinta specifica fara baza in dialog.")
        return None

    def _detect_context_misalignment(self, turns: List[Turn]) -> Optional[Dict[str, Any]]:
        last_user_digits = None
        last_user_email = None
        for idx, turn in enumerate(turns, start=1):
            text = turn.get("text", "")
            if turn.get("role") == "user":
                digits = re.findall(r"\b\d{4,16}\b", text)
                emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
                if digits:
                    last_user_digits = digits[-1]
                if emails:
                    last_user_email = emails[-1].lower()
            elif turn.get("role") == "assistant":
                assistant_digits = re.findall(r"\b\d{4,16}\b", text)
                assistant_emails = [m.lower() for m in re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)]
                if last_user_digits and assistant_digits and last_user_digits not in assistant_digits:
                    return _inc("nealiniat_context", "medium", f"ASSISTANT turn {idx} foloseste un numar diferit fata de cel oferit de utilizator.")
                if last_user_email and assistant_emails and last_user_email not in assistant_emails:
                    return _inc("nealiniat_context", "medium", f"ASSISTANT turn {idx} foloseste un email diferit fata de cel oferit de utilizator.")
        return None

    def _detect_contradiction(self, assistant_turns: Iterable[tuple[int, str]]) -> Optional[Dict[str, Any]]:
        has_can = None
        has_cannot = None
        for idx, text in assistant_turns:
            normalized = _normalize(text)
            if _has_any(normalized, ["pot rezolva", "am rezolvat", "se poate"]):
                has_can = idx
            if _has_any(normalized, ["nu pot rezolva", "nu se poate", "nu putem procesa"]):
                has_cannot = idx
        if has_can and has_cannot:
            return _inc("contradictoriu", "medium", f"ASSISTANT turn {has_can} si turn {has_cannot} contin afirmatii incompatibile despre posibilitatea rezolvarii.")
        return None

    def _detect_ignored_user_correction(self, turns: List[Turn]) -> Optional[Dict[str, Any]]:
        for idx in range(len(turns) - 1):
            user = turns[idx]
            assistant = turns[idx + 1]
            if user.get("role") != "user" or assistant.get("role") != "assistant":
                continue
            user_text = _normalize(user.get("text", ""))
            assistant_text = _normalize(assistant.get("text", ""))
            if (
                _has_any(user_text, ["nu despre card", "nu card", "despre vreme", "te-am intrebat despre vreme"])
                and _has_any(assistant_text, ["card", "cifre", "blocati", "blocarea"])
            ):
                return _inc(
                    "nealiniat_context",
                    "high",
                    f"ASSISTANT turn {idx + 2} continua fluxul despre card dupa ce utilizatorul corectase explicit subiectul.",
                )
        return None

    def _detect_incomplete_answer(self, turns: List[Turn]) -> Optional[Dict[str, Any]]:
        for idx in range(len(turns) - 1):
            user = turns[idx]
            assistant = turns[idx + 1]
            if user.get("role") != "user" or assistant.get("role") != "assistant":
                continue
            user_text = _normalize(user.get("text", ""))
            assistant_text = _normalize(assistant.get("text", ""))
            multi_request = " si " in user_text and _has_any(user_text, ["sold", "extras", "bloc", "programare", "parola", "comision"])
            if multi_request and not any(keyword in assistant_text for keyword in ["sold", "extras", "bloc", "programare", "parola", "comision", "mai intai"]):
                return _inc("incomplet", "low", f"ASSISTANT turn {idx + 2} nu adreseaza cererea multipla formulata anterior.")
        return None


def format_conversation(conversation: List[Turn]) -> str:
    return "\n".join(f"{turn['role'].upper()}: {turn['text']}" for turn in conversation)


def dump_evaluation(evaluation: EvaluationResult) -> str:
    return json.dumps(evaluation.to_dict(), ensure_ascii=False, indent=2)


def _normalize(text: str) -> str:
    replacements = {
        "ă": "a",
        "â": "a",
        "î": "i",
        "ș": "s",
        "ş": "s",
        "ț": "t",
        "ţ": "t",
    }
    lowered = text.lower()
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return lowered


def _has_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _result(field_name: str, value: str, confidence: str, reasoning: str) -> Dict[str, str]:
    return {field_name: value, "confidence": confidence, "reasoning": reasoning}


def _status(value: str, confidence: str, reasoning: str) -> Dict[str, str]:
    return _result("final_status", value, confidence, reasoning)


def _last_user_interruption_turn(conversation: List[Turn]) -> int:
    interruption_markers = [
        "revin",
        "gata",
        "inchid",
        "nu mai pot",
        "mai tarziu",
        "opresc",
        "nu acum",
        "alta data",
        "intrerup",
    ]
    last_turn = 0
    for idx, turn in enumerate(conversation, start=1):
        if turn.get("role") == "user" and _has_any(_normalize(turn.get("text", "")), interruption_markers):
            last_turn = idx
    return last_turn


def _last_assistant_resolution_turn(conversation: List[Turn]) -> int:
    resolution_markers = [
        "am blocat",
        "am actualizat",
        "am programat",
        "am trimis",
        "a fost confirmata",
        "am inregistrat",
        "soldul disponibil",
    ]
    last_turn = 0
    for idx, turn in enumerate(conversation, start=1):
        if turn.get("role") == "assistant" and _has_any(_normalize(turn.get("text", "")), resolution_markers):
            last_turn = idx
    return last_turn


def _inc(kind: str, confidence: str, reasoning: str) -> Dict[str, Any]:
    return {
        "has_incongruity": True,
        "incongruity_type": kind,
        "confidence": confidence,
        "reasoning": reasoning,
    }
