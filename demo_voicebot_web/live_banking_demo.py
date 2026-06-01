import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from conversation_kb import ConversationKnowledgeBase
from conversation_evaluator import ConversationEvaluator, Turn, dump_evaluation, format_conversation
from speech_normalizer import normalize_for_tts


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


END_COMMANDS = {"analiza", "analiza finala", "final", "stop", "exit", "iesire"}


@dataclass
class BankingSessionState:
    intent: Optional[str] = None
    authenticated: bool = False
    slots: Dict[str, str] = field(default_factory=dict)
    transcript: List[Turn] = field(default_factory=list)
    kb_examples: List[Dict[str, object]] = field(default_factory=list)


class BankingVoicebotDemo:
    def __init__(
        self,
        knowledge_base: Optional[ConversationKnowledgeBase] = None,
        llm_fallback: Optional[Callable[[List[Turn], str, Dict[str, object]], Optional[str]]] = None,
    ) -> None:
        self.state = BankingSessionState()
        self.evaluator = ConversationEvaluator()
        self.knowledge_base = knowledge_base or ConversationKnowledgeBase()
        self.llm_fallback = llm_fallback

    def start_message(self) -> str:
        text = (
            "Bună ziua! Sunt asistentul vocal demonstrativ pentru asistență bancară. "
            "Vă pot ajuta cu blocarea unui card, sold, extras de cont, tranzacții suspecte, date personale, "
            "programări cu un consultant sau resetarea accesului. Cu ce vă pot ajuta?"
        )
        self._add("assistant", text)
        return text

    def handle_user_message(self, user_text: str) -> str:
        self._add("user", user_text)
        response = self._handle_user_message_core(user_text, allow_llm=True)
        self._add("assistant", response)
        return response

    def handle_user_message_dataset_first(self, user_text: str) -> str:
        self._add("user", user_text)
        self._refresh_kb_context(user_text)
        response = self._try_llm_response(user_text)
        if not response:
            response = self._handle_user_message_core(user_text, allow_llm=False)
        self._add("assistant", response)
        return response

    def handle_user_message_llm_first(self, user_text: str) -> str:
        # Compatibilitate cu serverul existent: comportamentul este acum RAG dataset-first.
        return self.handle_user_message_dataset_first(user_text)

    def _handle_user_message_core(self, user_text: str, allow_llm: bool) -> str:
        normalized = _normalize(user_text)
        self._refresh_kb_context(user_text)

        if _is_interruption(normalized):
            return "Am înțeles. Oprim conversația aici și puteți reveni oricând aveți timp."

        if _is_small_talk(normalized):
            return "Bună! Sunt aici și vă pot ajuta cu operațiuni bancare sau cu un scenariu de test pentru demo."

        if _is_ambiguous_banking_fragment(normalized):
            default = _clarify_card_or_account(normalized)
            return self._fallback_response(user_text, default=default) if allow_llm else default

        current_turn_intent = self.evaluator.extract_intent([{"role": "user", "text": user_text}])["intent"]
        if current_turn_intent != "fallback" and _should_switch_intent(self.state.intent, current_turn_intent, normalized):
            self._switch_intent(current_turn_intent)

        if self.state.intent is None or self.state.intent == "fallback":
            evaluator_intent = current_turn_intent
            if evaluator_intent != "fallback":
                detected = evaluator_intent
            elif _looks_banking_related(normalized):
                detected = self.knowledge_base.suggest_intent(user_text) or evaluator_intent
            else:
                detected = "fallback"
            self.state.intent = detected

        if _has_any(normalized, ["operator", "persoana", "om", "agent uman"]):
            response = "Sigur, va transfer catre un operator uman pentru continuarea solicitarii."
        elif self.state.intent == "blocare_card":
            response = self._handle_block_card(normalized)
        elif self.state.intent == "deblocare_card":
            response = self._handle_unblock_card(normalized)
        elif self.state.intent == "deschidere_cont":
            response = self._handle_open_account(normalized)
        elif self.state.intent == "inchidere_cont":
            response = self._handle_close_account(normalized)
        elif self.state.intent == "verificare_sold":
            response = self._handle_balance(normalized)
        elif self.state.intent == "extras_de_cont":
            response = self._handle_statement(normalized)
        elif self.state.intent == "tranzactie_suspicioasa":
            response = self._handle_suspicious_transaction(normalized)
        elif self.state.intent == "actualizare_date_personale":
            response = self._handle_update_personal_data(user_text, normalized)
        elif self.state.intent == "programare_consultant":
            response = self._handle_advisor_meeting(user_text, normalized)
        elif self.state.intent == "resetare_autentificare":
            response = self._handle_auth_reset(normalized)
        elif self.state.intent == "informatii_produse":
            response = self._handle_product_info(normalized)
        else:
            response = self._fallback_response(user_text) if allow_llm else _gentle_fallback_message()

        return response

    def final_evaluation(self) -> str:
        return dump_evaluation(self.evaluator.evaluate(self.state.transcript))

    def knowledge_base_context(self) -> Dict[str, object]:
        query = " ".join(turn["text"] for turn in self.state.transcript if turn.get("role") == "user")
        context = self.knowledge_base.explain(query, intent_hint=self.state.intent)
        if self.state.kb_examples:
            context["examples"] = self.state.kb_examples
        return context

    def _handle_block_card(self, normalized: str) -> str:
        if "card_digits" not in self.state.slots:
            digits = _last_digits(normalized)
            if digits:
                self.state.slots["card_digits"] = digits
            else:
                return "Îmi pare rău pentru situație. Pentru siguranță, spuneți ultimele 4 cifre ale cardului pe care doriți să îl blocați."

        if "confirmation" not in self.state.slots and not _has_any(normalized, ["da", "confirm", "sigur"]):
            spoken_digits = _spell_digits(self.state.slots["card_digits"])
            return f"Confirmați blocarea cardului terminat în {spoken_digits}? Răspundeți cu da sau nu."

        self.state.slots["confirmation"] = "da"
        spoken_digits = _spell_digits(self.state.slots["card_digits"])
        return f"Am blocat cardul terminat în {spoken_digits}. Veți primi o confirmare prin SMS."

    def _handle_unblock_card(self, normalized: str) -> str:
        if not self.state.authenticated:
            if _has_any(normalized, ["confirm", "da", "cod", "autentificat"]):
                self.state.authenticated = True
            else:
                return "Pentru deblocarea cardului este necesară autentificarea. Confirmați codul primit prin SMS?"
        return "Am înregistrat cererea de deblocare. Din motive de securitate, un operator uman va valida solicitarea."

    def _handle_open_account(self, normalized: str) -> str:
        if _has_any(normalized, ["cat costa", "cost", "pret", "taxa", "comision"]):
            return "Deschiderea unui cont curent standard nu are cost de deschidere în acest demo. Comisioanele pot depinde de pachetul ales."
        if "phone" not in self.state.slots:
            phone = _last_digits(normalized)
            if phone and len(phone) >= 10:
                self.state.slots["phone"] = phone
            else:
                return "Vă pot ajuta cu deschiderea unui cont. Spuneți-mi un număr de telefon pentru programarea discuției cu un consultant."
        return f"Am programat un consultant să vă contacteze la {self.state.slots['phone']} pentru deschiderea contului."

    def _handle_close_account(self, normalized: str) -> str:
        if not self.state.authenticated:
            if _has_any(normalized, ["da", "confirm", "autentificat"]):
                self.state.authenticated = True
            else:
                return "Pentru închiderea contului trebuie să confirmați identitatea. Doriți să continui verificarea?"
        return "Am înregistrat cererea de închidere a contului. Pentru finalizare, trebuie să semnați documentele în aplicație."

    def _handle_balance(self, normalized: str) -> str:
        if not self.state.authenticated:
            if _has_any(normalized, ["da", "confirm", "cod", "autentificat"]):
                self.state.authenticated = True
            else:
                return "Pentru sold trebuie să confirmați identitatea. Puteți confirma codul primit prin SMS?"
        return "Soldul disponibil pentru contul principal este 2.450 de lei."

    def _handle_statement(self, normalized: str) -> str:
        if "period" not in self.state.slots:
            if _has_any(normalized, ["luna", "ultimele", "ianuarie", "februarie", "martie", "aprilie", "mai"]):
                self.state.slots["period"] = "perioada solicitată"
            else:
                return "Pentru ce perioadă doriți extrasul de cont?"
        return "Am trimis extrasul de cont pentru perioada solicitată pe adresa de email asociată contului."

    def _handle_suspicious_transaction(self, normalized: str) -> str:
        if "details" not in self.state.slots:
            if _has_any(normalized, ["lei", "ron", "euro", "ieri", "azi"]) or _last_digits(normalized):
                self.state.slots["details"] = "primite"
            else:
                return "Îmi pare rău pentru această situație. Spuneți-mi suma sau data tranzacției pe care nu o recunoașteți."
        return "Am înregistrat raportarea tranzacției suspecte și am trimis cazul către echipa de securitate."

    def _handle_update_personal_data(self, user_text: str, normalized: str) -> str:
        if "new_value" not in self.state.slots:
            email = _extract_email(user_text)
            phone = _last_digits(normalized)
            if email:
                self.state.slots["new_value"] = email
            elif phone and len(phone) >= 10:
                self.state.slots["new_value"] = phone
            else:
                return "Ce date doriți să actualizați? Puteți spune noul telefon, noua adresă sau noul email."
        return f"Am actualizat datele personale cu valoarea: {self.state.slots['new_value']}."

    def _handle_advisor_meeting(self, user_text: str, normalized: str) -> str:
        if "meeting_time" not in self.state.slots:
            if _has_any(normalized, ["luni", "marti", "miercuri", "joi", "vineri", "maine", "ora"]):
                self.state.slots["meeting_time"] = user_text
            else:
                return "Pentru când doriți programarea cu un consultant?"
        return f"Am programat discuția cu un consultant pentru: {self.state.slots['meeting_time']}."

    def _handle_auth_reset(self, normalized: str) -> str:
        if not self.state.authenticated:
            if _has_any(normalized, ["da", "confirm", "cod", "autentificat"]):
                self.state.authenticated = True
            else:
                return "Pentru resetarea accesului trebuie să confirmați codul primit prin SMS. Îl confirmați?"
        return "Am trimis linkul de resetare pentru accesul în aplicație. Linkul este valabil 15 minute."

    def _handle_product_info(self, normalized: str) -> str:
        if _has_any(normalized, ["cat costa", "cost", "pret", "taxa"]) and _has_any(normalized, ["cont", "deschidere"]):
            return "Pentru deschiderea unui cont, costurile depind de pachetul ales. În acest demo, deschiderea contului standard este tratată ca gratuită."
        if "card credit" in normalized:
            return "Pentru cardul de credit, limita și comisioanele depind de profilul clientului. Pot programa o discuție cu un consultant."
        if "comision" in normalized:
            return "Comisioanele diferă în funcție de pachetul de cont. Vă pot trimite sumarul condițiilor pe email."
        return "Vă pot oferi informații despre conturi, carduri, credite, comisioane și condiții generale."

    def _add(self, role: str, text: str) -> None:
        self.state.transcript.append({"role": role, "text": text})

    def _refresh_kb_context(self, latest_user_text: str) -> None:
        if not self.knowledge_base.available:
            self.state.kb_examples = []
            return
        current_turn_intent = self.evaluator.extract_intent([{"role": "user", "text": latest_user_text}])["intent"]
        if current_turn_intent != "fallback":
            self.state.intent = current_turn_intent
        query = " ".join(turn["text"] for turn in self.state.transcript if turn.get("role") == "user")
        if not query:
            query = latest_user_text
        self.state.kb_examples = [
            example.to_dict() for example in self.knowledge_base.search(query, top_k=4, intent_hint=self.state.intent)
        ]

    def _fallback_response(self, user_text: str, default: Optional[str] = None) -> str:
        generated = self._try_llm_response(user_text)
        if generated:
            return generated
        if default:
            return default
        return (
            "Pot continua pe solicitări bancare precum carduri, conturi, sold, extras, tranzacții suspecte, "
            "date personale, programări sau resetarea accesului. Spuneți-mi ce doriți să verificăm."
        )

    def _try_llm_response(self, user_text: str) -> Optional[str]:
        if not self.llm_fallback:
            return None
        try:
            return self.llm_fallback(self.state.transcript, user_text, self.knowledge_base_context())
        except Exception:
            return None

    def _switch_intent(self, new_intent: str) -> None:
        if self.state.intent != new_intent:
            self.state.intent = new_intent
            self.state.slots.clear()
            self.state.authenticated = False


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Demo live pentru evaluarea conversatiilor voicebot.")
    parser.add_argument("--show-transcript", action="store_true", help="Afiseaza transcriptul inainte de analiza finala.")
    parser.add_argument("--voice", action="store_true", help="Ruleaza demo-ul cu microfon si voce Zevo.")
    parser.add_argument("--zevo-api-key", default=os.getenv("ZEVO_API_KEY"), help="Cheia API Zevo. Alternativ, seteaza ZEVO_API_KEY.")
    parser.add_argument("--tts-voice", default=os.getenv("ZEVO_TTS_VOICE", "gia"), help="Vocea Zevo TTS folosita pentru raspunsuri.")
    args = parser.parse_args()

    demo = BankingVoicebotDemo()
    first_message = demo.start_message()
    print(f"VOICEBOT: {first_message}")

    if args.voice:
        run_voice_loop(demo, args, first_message)
    else:
        print("Scrieti 'analiza' cand doriti intentul, statusul final si incongruentele.\n")
        run_text_loop(demo)

    if args.show_transcript:
        print("\n=== Transcript ===")
        print(format_conversation(demo.state.transcript))

    print("\n=== Analiza finala ===")
    print(demo.final_evaluation())


def run_text_loop(demo: BankingVoicebotDemo) -> None:
    while True:
        user_text = input("USER: ").strip()
        if not user_text:
            continue
        if _normalize(user_text) in END_COMMANDS:
            break
        print(f"VOICEBOT: {demo.handle_user_message(user_text)}")


def run_voice_loop(demo: BankingVoicebotDemo, args: argparse.Namespace, first_message: str) -> None:
    if not args.zevo_api_key:
        raise SystemExit("Lipsește cheia Zevo. Setează ZEVO_API_KEY sau folosește --zevo-api-key.")

    try:
        from zevo_stt import DEFAULT_DOMAIN_STT_GENERAL, record_and_transcribe_from_mic
        from zevo_tts import speak_text_zevo
    except ImportError as exc:
        raise SystemExit(
            "Nu pot porni modul audio. Instalează dependențele pentru zevo_stt.py și zevo_tts.py "
            f"sau rulează fără --voice. Detaliu: {exc}"
        ) from exc

    print("Mod audio pornit. Vorbiți după mesajul 'Ascult...'. Spuneți 'analiza' pentru evaluarea finală.\n")
    speak_text_zevo(args.zevo_api_key, normalize_for_tts(first_message), voice=args.tts_voice)

    while True:
        print("Ascult...")
        user_text = record_and_transcribe_from_mic(args.zevo_api_key, domain=DEFAULT_DOMAIN_STT_GENERAL).strip()
        if not user_text:
            print("Nu am primit transcriere. Încercăm din nou.")
            continue
        print(f"USER: {user_text}")
        if _normalize(user_text) in END_COMMANDS:
            break
        bot_response = demo.handle_user_message(user_text)
        print(f"VOICEBOT: {bot_response}")
        speak_text_zevo(args.zevo_api_key, normalize_for_tts(bot_response), voice=args.tts_voice)


def _normalize(text: str) -> str:
    replacements = {"ă": "a", "â": "a", "î": "i", "ș": "s", "ş": "s", "ț": "t", "ţ": "t"}
    lowered = text.lower()
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return lowered


def _has_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_interruption(text: str) -> bool:
    return _has_any(
        text,
        [
            "revin",
            "mai tarziu",
            "nu mai pot",
            "inchid",
            "intrerup",
            "opresc",
            "gata",
            "nu acum",
            "alta data",
        ],
    )


def _is_small_talk(text: str) -> bool:
    cleaned = text.strip(" .,!?:;-")
    if cleaned.startswith(("hello", "helo", "hell", "buna", "bună", "salut", "hei", "ceau", "ciao")) and len(cleaned.split()) <= 2:
        return True
    return cleaned in {
        "buna",
        "salut",
        "hello",
        "hei",
        "ceau",
        "ciao",
        "ce faci",
        "cum esti",
        "cum ești",
        "mersi",
        "multumesc",
        "buna ziua",
        "bună",
        "mulțumesc",
    }


def _is_ambiguous_banking_fragment(text: str) -> bool:
    cleaned = text.strip(" .,!?:;-")
    words = cleaned.split()
    if len(words) > 3:
        return False
    ambiguous = {
        "card",
        "cardul",
        "carduri",
        "cont",
        "contul",
        "conturi",
        "credit",
        "credite",
        "banca",
        "bancă",
        "aplicatie",
        "aplicație",
    }
    if cleaned in ambiguous:
        return True
    return any(word in ambiguous for word in words) and not _has_any(
        cleaned,
        [
            "blochez",
            "blocare",
            "deblochez",
            "deblocare",
            "pierdut",
            "furat",
            "sold",
            "extras",
            "deschid",
            "inchid",
            "închid",
            "comision",
            "cost",
            "taxa",
            "taxă",
        ],
    )


def _clarify_card_or_account(text: str) -> str:
    if "card" in text:
        return (
            "Sigur. Despre card vă pot ajuta cu blocare, deblocare, card de credit, comisioane sau tranzacții. "
            "Ce doriți să facem mai exact?"
        )
    if "cont" in text:
        return (
            "Sigur. Pentru cont vă pot ajuta cu sold, extras, deschidere, închidere sau informații despre comisioane. "
            "Ce anume vă interesează?"
        )
    return "Sigur. Spuneți-mi puțin mai exact ce operațiune bancară doriți să verificăm."


def _should_switch_intent(current_intent: Optional[str], new_intent: str, text: str) -> bool:
    if current_intent in {None, "fallback"}:
        return True
    if current_intent == new_intent:
        return False
    if _has_any(text, ["nu", "alt", "alta", "schimb", "acum", "vreau", "as vrea", "aș vrea"]):
        return True
    return new_intent in {
        "verificare_sold",
        "extras_de_cont",
        "tranzactie_suspicioasa",
        "programare_consultant",
        "resetare_autentificare",
        "informatii_produse",
        "deschidere_cont",
        "inchidere_cont",
        "blocare_card",
        "deblocare_card",
    }


def _looks_banking_related(text: str) -> bool:
    return _has_any(
        text,
        [
            "card",
            "cont",
            "sold",
            "bani",
            "extras",
            "tranzactie",
            "parola",
            "pin",
            "iban",
            "credit",
            "comision",
            "dobanda",
            "consultant",
            "banca",
        ],
    )


def _gentle_fallback_message() -> str:
    return (
        "Pot continua pe solicitări bancare precum carduri, conturi, sold, extras, tranzacții suspecte, "
        "date personale, programări sau resetarea accesului. Spuneți-mi ce doriți să verificăm."
    )


def _last_digits(text: str) -> Optional[str]:
    import re

    matches = re.findall(r"\b\d{4,16}\b", text)
    if matches:
        return matches[-1]

    spaced_digit_matches = re.findall(r"(?:\b\d\b[\s,.-]*){4,16}", text)
    if spaced_digit_matches:
        digits = re.sub(r"\D", "", spaced_digit_matches[-1])
        if 4 <= len(digits) <= 16:
            return digits

    return None


def _spell_digits(value: str) -> str:
    return " ".join(value)


def _extract_email(text: str) -> Optional[str]:
    import re

    matches = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return matches[-1] if matches else None


if __name__ == "__main__":
    run_cli()
