import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from conversation_evaluator import Turn


INTENT_MAP = {
    "block_card": "blocare_card",
    "unblock_card": "deblocare_card",
    "open_account": "deschidere_cont",
    "close_account": "inchidere_cont",
    "check_balance": "verificare_sold",
    "get_account_statement": "extras_de_cont",
    "report_suspicious_transaction": "tranzactie_suspicioasa",
    "update_personal_data": "actualizare_date_personale",
    "schedule_advisor_meeting": "programare_consultant",
    "reset_or_recover_auth": "resetare_autentificare",
    "general_product_info": "informatii_produse",
    "fallback": "fallback",
}


DEFAULT_DATASET_CANDIDATES = [
    Path(__file__).resolve().parent.parent
    / "data"
    / "master_dataset_refined_180.json",
    Path(__file__).resolve().parent.parent
    / "Sistem-de-monitorizare-a-interac-iunilor-voicebotilor-folosind-modele-lingvistice-mari-LLM-"
    / "data"
    / "master_dataset_refined_180.json",
    Path.home()
    / "Documents"
    / "Sistem-de-monitorizare-a-interac-iunilor-voicebotilor-folosind-modele-lingvistice-mari-LLM-"
    / "data"
    / "master_dataset_refined_180.json",
]


@dataclass
class KBExample:
    conversation_id: str
    dataset_intent: str
    mapped_intent: str
    final_status: str
    score: float
    first_user_message: str
    assistant_resolution: str
    turns: List[Turn]

    def to_dict(self) -> Dict[str, object]:
        return {
            "conversation_id": self.conversation_id,
            "dataset_intent": self.dataset_intent,
            "mapped_intent": self.mapped_intent,
            "final_status": self.final_status,
            "score": round(self.score, 4),
            "first_user_message": self.first_user_message,
            "assistant_resolution": self.assistant_resolution,
            "turns": self.turns,
        }


class ConversationKnowledgeBase:
    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        self.dataset_path = dataset_path or find_default_dataset()
        self.conversations: List[Dict[str, object]] = []
        self.documents: List[Dict[str, object]] = []
        if self.dataset_path and self.dataset_path.exists():
            self._load(self.dataset_path)

    @property
    def available(self) -> bool:
        return bool(self.documents)

    def search(self, query: str, top_k: int = 3, intent_hint: Optional[str] = None) -> List[KBExample]:
        if not self.documents:
            return []
        query_vector = text_vector(query)
        if not query_vector:
            return []

        scored = []
        for doc in self.documents:
            score = cosine(query_vector, doc["vector"])
            if score > 0:
                conversation = doc["conversation"]
                mapped_intent = INTENT_MAP.get(str(conversation.get("intent", "")), "fallback")
                reranked_score = score * 1.25 if intent_hint and mapped_intent == intent_hint else score
                scored.append((reranked_score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [self._example_from_doc(doc, score) for score, doc in scored[:top_k]]

    def suggest_intent(self, query: str) -> Optional[str]:
        examples = self.search(query, top_k=5)
        if not examples:
            return None
        votes: Counter[str] = Counter()
        for example in examples:
            votes[example.mapped_intent] += example.score
        intent, score = votes.most_common(1)[0]
        return intent if score >= 0.08 else None

    def explain(self, query: str, top_k: int = 3, intent_hint: Optional[str] = None) -> Dict[str, object]:
        examples = self.search(query, top_k=top_k, intent_hint=intent_hint)
        return {
            "available": self.available,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "examples": [example.to_dict() for example in examples],
            "suggested_intent": self.suggest_intent(query),
        }

    def _load(self, dataset_path: Path) -> None:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        self.conversations = data.get("conversations", [])
        self.documents = []
        for conversation in self.conversations:
            turns = conversation.get("turns", [])
            text = conversation_text(turns)
            self.documents.append(
                {
                    "conversation": conversation,
                    "text": text,
                    "vector": text_vector(text),
                }
            )

    def _example_from_doc(self, doc: Dict[str, object], score: float) -> KBExample:
        conversation = doc["conversation"]
        turns = conversation.get("turns", [])
        return KBExample(
            conversation_id=str(conversation.get("conversation_id", "")),
            dataset_intent=str(conversation.get("intent", "")),
            mapped_intent=INTENT_MAP.get(str(conversation.get("intent", "")), "fallback"),
            final_status=str(conversation.get("final_status", "")),
            score=score,
            first_user_message=first_turn_text(turns, "user"),
            assistant_resolution=last_turn_text(turns, "assistant"),
            turns=[{"role": turn.get("role", ""), "text": turn.get("text", "")} for turn in turns],
        )


def find_default_dataset() -> Optional[Path]:
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def conversation_text(turns: Iterable[Turn]) -> str:
    return " ".join(turn.get("text", "") for turn in turns)


def first_turn_text(turns: Iterable[Turn], role: str) -> str:
    for turn in turns:
        if turn.get("role") == role:
            return turn.get("text", "")
    return ""


def last_turn_text(turns: Iterable[Turn], role: str) -> str:
    found = ""
    for turn in turns:
        if turn.get("role") == role:
            found = turn.get("text", "")
    return found


def text_vector(text: str) -> Counter[str]:
    tokens = [token for token in re.findall(r"[a-z0-9ăâîșşțţ]+", normalize(text)) if len(token) > 2]
    return Counter(tokens)


def normalize(text: str) -> str:
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


def cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
