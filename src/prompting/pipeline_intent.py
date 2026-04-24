# src/prompting/pipeline_intent.py

"""
Pipeline pentru extragerea intențiilor din conversații voicebot-utilizator.

Modele folosite:
─────────────────────────────────────────────────────────────────────────────
1. OpenAI o3
   Model de tip reasoning dezvoltat de OpenAI. Gândește pas cu pas înainte
   de a răspunde, ceea ce îl face potrivit pentru sarcini de clasificare unde
   contextul conversației trebuie interpretat cu atenție. Accesat via API
   oficial OpenAI. Folosit ca benchmark comercial de performanță maximă.

2. Gemini 2.5 Pro (gemini-2.5-pro-exp-03-25)
   Modelul flagship al Google DeepMind cu capacități de reasoning. Suportă
   un context extrem de mare (1M tokens) și performează bine pe sarcini
   multilingve. Accesat via Google Generative AI SDK. Folosit ca benchmark
   comercial pentru viteză și context mare.

3. Aya Expanse 8B
   Model open-weight dezvoltat de Cohere Labs, optimizat explicit pentru
   multilingvism (23 de limbi, inclusiv română). 8 miliarde de parametri.
   Rulat local via Ollama. Folosit ca referință open-source pentru limbi
   non-engleze.

4. RoLLaMA 2 7B Instruct (RoLlama2-7b-Instruct)
   Model generativ bazat pe LLaMA 2, fine-tunat pe date în limba română.
   7 miliarde de parametri. Rulat local via Ollama. Singurul model din
   pipeline antrenat nativ pe limba română, testat ca pilon pentru RO.

5. RoBERT (encoder)
   Model de tip encoder (BERT) pre-antrenat pe texte românești, dezvoltat
   de readerbench / UPB. Nu generează text — clasifică direct prin
   zero-shot-classification pe baza similarității semantice între textul
   conversației și etichetele de intenție. Nu folosește prompt Jinja;
   primește textul brut al conversației. Testat pentru a vedea dacă un
   clasificator clasic bate LLM-urile generative pe această sarcină.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import time
import re
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

PROMPTS_DIR = Path("prompts/intent_extraction")
CONFIGS_DIR = Path("configs")
OUTPUT_DIR  = Path(
    r"C:\Users\Matebook 14s\Documents"
    r"\Sistem-de-monitorizare-a-interac-iunilor-voicebotilor-folosind-modele-lingvistice-mari-LLM-"
    r"\outputs"
)

# ──────────────────────────────────────────────
# Valid labels (for output validation)
# ──────────────────────────────────────────────

with open(CONFIGS_DIR / "intent_definitions.json", encoding="utf-8") as f:
    _intent_definitions = json.load(f)

VALID_LABELS = {label["name"] for label in _intent_definitions["labels"]}

# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class PredictionResult:
    conversation_id:  str
    dataset_label:    str        # eticheta din dataset (fost "gold")
    model_name:       str
    prompt_lang:      str        # "en" | "ro"
    predicted_intent: str | None
    confidence:       str | None
    reasoning:        str | None
    latency_ms:       float
    parse_failed:     bool
    raw_response:     str

# ──────────────────────────────────────────────
# Prompt renderer
# ──────────────────────────────────────────────

env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))


def render_prompt(lang: str, conversation: list[dict]) -> str:
    template_name = "en_zero_shot.jinja" if lang == "en" else "ro_zero_shot.jinja"
    template = env.get_template(template_name)
    return template.render(conversation=conversation)


# ──────────────────────────────────────────────
# Response parser
# ──────────────────────────────────────────────

def parse_response(raw: str) -> tuple[str | None, str | None, str | None, bool]:
    """
    Parsează răspunsul modelului și extrage (intent, confidence, reasoning, parse_failed).
    Gestionează cazurile în care modelul înfășoară JSON-ul în ```json ... ```.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
        intent     = data.get("intent")
        confidence = data.get("confidence")
        reasoning  = data.get("reasoning")

        if intent not in VALID_LABELS:
            # Modelul a returnat o etichetă invalidă — parse_failed = True
            return intent, confidence, reasoning, True

        return intent, confidence, reasoning, False

    except json.JSONDecodeError:
        return None, None, None, True


# ──────────────────────────────────────────────
# Base model caller
# ──────────────────────────────────────────────

class BaseModelCaller(ABC):
    name: str

    def call(
        self,
        prompt: str,
        conversation_id: str,
        dataset_label: str,
        lang: str,
    ) -> PredictionResult:

        start = time.monotonic()
        raw = self._call_api(prompt)
        latency_ms = (time.monotonic() - start) * 1000

        intent, confidence, reasoning, parse_failed = parse_response(raw)

        return PredictionResult(
            conversation_id=conversation_id,
            dataset_label=dataset_label,
            model_name=self.name,
            prompt_lang=lang,
            predicted_intent=intent,
            confidence=confidence,
            reasoning=reasoning,
            latency_ms=latency_ms,
            parse_failed=parse_failed,
            raw_response=raw,
        )

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        pass


# ──────────────────────────────────────────────
# 1. OpenAI o3
# ──────────────────────────────────────────────

class OpenAICaller(BaseModelCaller):
    """
    OpenAI o3 — model reasoning accesat via API oficial.
    Citește OPENAI_API_KEY din variabilele de mediu.
    """
    name = "openai_o3"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        self.model  = "o3"

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


# ──────────────────────────────────────────────
# 2. Gemini 2.5 Pro
# ──────────────────────────────────────────────

class GeminiCaller(BaseModelCaller):
    """
    Gemini 2.5 Pro — model Google DeepMind accesat via google-generativeai SDK.
    Citește GEMINI_API_KEY din variabilele de mediu.
    """
    name = "gemini_2.5_pro"

    def __init__(self):
        import google.generativeai as genai
        import os
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")

    def _call_api(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


# ──────────────────────────────────────────────
# 3. Aya Expanse 8B
# ──────────────────────────────────────────────

class AyaExpanseCaller(BaseModelCaller):
    """
    Aya Expanse 8B — model open-weight Cohere Labs, rulat local via Ollama.
    Înainte de prima rulare: ollama pull aya-expanse:8b
    """
    name = "aya_expanse_8b"

    def __init__(self):
        import ollama
        self.client = ollama
        self.model  = "aya-expanse:8b"

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]


# ──────────────────────────────────────────────
# 4. RoLLaMA 2 7B Instruct
# ──────────────────────────────────────────────

class RoLlamaCaller(BaseModelCaller):
    """
    RoLlama2-7b-Instruct — model LLaMA 2 fine-tunat pe română, rulat local via Ollama.
    Înainte de prima rulare: ollama pull rollama2:7b-instruct
    Verifică tag-ul exact cu: ollama list
    """
    name = "rollama2_7b"

    def __init__(self):
        import ollama
        self.client = ollama
        self.model  = "rollama2:7b-instruct"  # ajustează dacă tag-ul diferă

    def _call_api(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]


# ──────────────────────────────────────────────
# 5. RoBERT (encoder, zero-shot classification)
# ──────────────────────────────────────────────

class RoBERTCaller(BaseModelCaller):
    """
    RoBERT — model BERT pre-antrenat pe texte românești (readerbench/ro-bert).
    Nu generează text și nu folosește prompt Jinja.
    Primește textul brut al conversației și îl clasifică direct prin
    zero-shot-classification (similaritate semantică între text și etichete).
    Rulat local via HuggingFace Transformers.
    """
    name = "robert_encoder"

    def __init__(self):
        from transformers import pipeline
        self.labels = list(VALID_LABELS)
        self.pipe = pipeline(
            "zero-shot-classification",
            model="readerbench/ro-bert",
            device=0,   # GPU; schimbă în -1 pentru CPU
        )

    def call(
        self,
        prompt: str,
        conversation_id: str,
        dataset_label: str,
        lang: str,
    ) -> PredictionResult:
        """
        Override complet al metodei call() — RoBERT nu parsează JSON.
        Extrage textul conversației din prompt și clasifică direct.
        """
        conv_text = self._extract_conversation_text(prompt)

        start  = time.monotonic()
        result = self.pipe(conv_text, candidate_labels=self.labels)
        latency_ms = (time.monotonic() - start) * 1000

        predicted = result["labels"][0]
        score     = result["scores"][0]
        confidence = "high" if score > 0.7 else "medium" if score > 0.4 else "low"

        return PredictionResult(
            conversation_id=conversation_id,
            dataset_label=dataset_label,
            model_name=self.name,
            prompt_lang=lang,
            predicted_intent=predicted,
            confidence=confidence,
            reasoning=f"zero-shot score: {score:.4f}",
            latency_ms=latency_ms,
            parse_failed=False,
            raw_response=str(result),
        )

    def _call_api(self, prompt: str) -> str:
        pass  # nu e folosit — call() este overridden complet

    def _extract_conversation_text(self, prompt: str) -> str:
        """
        Extrage replicile din prompt ca text simplu pentru RoBERT.
        Caută blocul dintre '## Conversation' / '## Conversație' și
        următorul header '##'.
        """
        lines = prompt.split("\n")
        conv_lines = []
        in_conv = False
        for line in lines:
            if line.strip().startswith("## Convers"):
                in_conv = True
                continue
            if in_conv and line.strip().startswith("##"):
                break
            if in_conv and line.strip():
                conv_lines.append(line.strip())
        return " ".join(conv_lines)


# ──────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────

def run_pipeline(
    dataset_path: str,
    langs: list[str] = ["ro", "en"],
    models: list[BaseModelCaller] = None,
) -> list[PredictionResult]:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "intent_predictions.json"

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    conversations = data["conversations"]
    results: list[PredictionResult] = []
    total = len(conversations) * len(langs) * len(models)
    done  = 0

    for conv in conversations:
        conv_id       = conv["conversation_id"]
        dataset_label = conv["intent"]        # eticheta din dataset
        turns         = conv["turns"]

        for lang in langs:
            prompt = render_prompt(lang, turns)

            for model in models:
                done += 1
                print(
                    f"[{done}/{total}] {model.name} | {lang} | {conv_id} ...",
                    end=" ",
                    flush=True,
                )
                try:
                    result = model.call(
                        prompt=prompt,
                        conversation_id=conv_id,
                        dataset_label=dataset_label,
                        lang=lang,
                    )
                    results.append(result)

                    status = "PARSE_FAIL" if result.parse_failed else result.predicted_intent
                    correct = "✓" if result.predicted_intent == dataset_label else "✗"
                    print(f"{correct} {status}  ({result.latency_ms:.0f}ms)")

                except Exception as e:
                    print(f"ERROR: {e}")
                    results.append(PredictionResult(
                        conversation_id=conv_id,
                        dataset_label=dataset_label,
                        model_name=model.name,
                        prompt_lang=lang,
                        predicted_intent=None,
                        confidence=None,
                        reasoning=None,
                        latency_ms=0.0,
                        parse_failed=True,
                        raw_response=str(e),
                    ))

    # Salvează rezultatele
    output = [vars(r) for r in results]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(results)} predicții salvate în:\n  {output_file}")
    return results


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    models = [
        OpenAICaller(),
        GeminiCaller(),
        AyaExpanseCaller(),
        RoLlamaCaller(),
        RoBERTCaller(),
    ]

    run_pipeline(
        dataset_path="data/master_dataset_refined_180.json",
        langs=["ro", "en"],
        models=models,
    )