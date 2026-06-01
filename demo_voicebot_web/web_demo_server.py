import argparse
import asyncio
import base64
import hashlib
import json
import os
import shutil
import sys
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
for import_path in (ROOT, PROJECT_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_file(PROJECT_ROOT / ".env")
load_dotenv_file(ROOT / ".env")

ENV_ALIASES = {
    "OPENAI_API_KEY": ["OPENAI_KEY", "OPENAI_TOKEN"],
    "GOOGLE_API_KEY": ["GEMINI_API_KEY", "GOOGLE_GENAI_API_KEY", "GOOGLE_AI_API_KEY"],
    "ZEVO_API_KEY": ["ZEVO_KEY", "ZEVO_LICENSE"],
}

for canonical, aliases in ENV_ALIASES.items():
    if not os.getenv(canonical):
        for alias in aliases:
            value = os.getenv(alias)
            if value:
                os.environ[canonical] = value
                break

from conversation_evaluator import ConversationEvaluator, Turn
from live_banking_demo import BankingVoicebotDemo
from speech_normalizer import normalize_for_tts
from zevo_stt import DEFAULT_DOMAIN_STT_GENERAL, speech_to_text_ws
from zevo_tts import (
    DEFAULT_AUDIO_FORMAT_TTS,
    DEFAULT_BITS_PER_SAMPLE_TTS,
    DEFAULT_SAMPLE_RATE_TTS,
    TTSRequestParams,
    ZEVO_TTS_API_URI,
    text_to_speech_ws,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


STATIC_ROOT = ROOT / "web_ui"
TTS_CACHE = ROOT / "tts_cache"
SESSIONS: Dict[str, BankingVoicebotDemo] = {}
LIVE_LLM_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-2.5-flash")

EVALUATION_MODELS = {
    "openai_o3": {"label": "OpenAI o3", "kind": "API", "provider": "openai", "supports_real": True},
    "gemini_2.5_flash": {"label": "Gemini 2.5 Flash", "kind": "API", "provider": "gemini", "supports_real": True},
    "aya_expanse_8b": {"label": "Aya Expanse 8B", "kind": "local", "provider": "ollama", "supports_real": True},
    "rollama2_7b": {"label": "RoLLaMA 2 7B", "kind": "local", "provider": "ollama", "supports_real": True},
    "roberta_encoder": {"label": "XLM-RoBERTa encoder", "kind": "local", "provider": "encoder", "supports_real": False},
    "mistral_7b": {"label": "Mistral 7B", "kind": "local", "provider": "ollama", "supports_real": True},
    "qwen2.5_3b": {"label": "Qwen2.5 3B", "kind": "local", "provider": "ollama", "supports_real": True},
}

TASK_RECOMMENDATIONS = {
    "intent": {
        "model": "openai_o3",
        "lang": "en",
        "prompt_version": "v4",
        "prompt_file": "prompts/intent_extraction/en_few_shot_v4.jinja",
        "definitions_file": "configs/intent_definitions.json",
        "few_shot_file": "configs/few_shot_examples_intent.json",
        "metric": "accuracy/F1 98.3%",
        "source": "evaluation_report_intent.txt",
    },
    "final_status": {
        "model": "openai_o3",
        "lang": "ro",
        "prompt_version": "v4",
        "prompt_file": "prompts/final_status/fs_ro_few_shot_v4.jinja",
        "definitions_file": "configs/final_status_definitions.json",
        "few_shot_file": "configs/few_shot_examples_final_status.json",
        "metric": "recomandare derivata din notebook; rezultate JSON lipsa",
        "source": "outputs_final_status/final_status_experiments.ipynb",
    },
    "incongruities": {
        "model": "gemini_2.5_flash",
        "lang": "ro",
        "prompt_version": "v4",
        "prompt_file": "prompts/incongruities/ro_incongruities_v4.jinja",
        "definitions_file": "configs/incongruities_definitions.json",
        "few_shot_file": "configs/few_shot_examples_incongruities.json",
        "metric": "binary F1 0.8219, type macro F1 0.8531",
        "source": "outputs_incongruities/exp_inc_gemini_2.5_flash__ro__v4.json",
    },
}

MODEL_TASK_PROMPTS = {
    "intent": {
        "openai_o3": {"lang": "en", "prompt_version": "v4", "note": "best/tie pentru OpenAI o3"},
        "gemini_2.5_flash": {"lang": "ro", "prompt_version": "v4", "note": "best pentru Gemini în raportul de intent"},
        "aya_expanse_8b": {"lang": "en", "prompt_version": "v4", "note": "best pentru Aya Expanse 8B"},
        "rollama2_7b": {"lang": "ro", "prompt_version": "v4", "note": "variantă compatibilă cu prompturile disponibile"},
        "mistral_7b": {"lang": "ro", "prompt_version": "v4", "note": "model adăugat pentru comparație"},
        "qwen2.5_3b": {"lang": "ro", "prompt_version": "v4", "note": "model adăugat pentru comparație"},
        "roberta_encoder": {"lang": "ro", "prompt_version": "v4", "note": "baseline encoder; fără apel generativ real"},
    },
    "final_status": {
        "default": {"lang": "ro", "prompt_version": "v4", "note": "v4 few-shot adăugat pentru demo"},
    },
    "incongruities": {
        "openai_o3": {"lang": "ro", "prompt_version": "v4", "note": "v4 few-shot disponibil"},
        "gemini_2.5_flash": {"lang": "ro", "prompt_version": "v4", "note": "best salvat pentru incongruențe"},
        "aya_expanse_8b": {"lang": "ro", "prompt_version": "v4", "note": "v4 few-shot disponibil"},
        "rollama2_7b": {"lang": "ro", "prompt_version": "v4", "note": "v4 few-shot disponibil"},
        "mistral_7b": {"lang": "ro", "prompt_version": "v4", "note": "model adăugat pentru comparație"},
        "qwen2.5_3b": {"lang": "ro", "prompt_version": "v4", "note": "model adăugat pentru comparație"},
        "roberta_encoder": {"lang": "ro", "prompt_version": "v4", "note": "baseline encoder; fără apel generativ real"},
    },
}


class BanutilHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.path = "/index.html"
            return super().do_GET()
        if parsed.path.startswith("/tts_cache/"):
            return self._serve_cache_file(parsed.path)
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        routes = {
            "/api/start": self._start_session,
            "/api/message": self._message,
            "/api/voice-message": self._voice_message,
            "/api/analyze": self._analyze,
            "/api/evaluation-options": self._evaluation_options,
            "/api/env-status": self._env_status,
            "/api/evaluate-conversation": self._evaluate_conversation,
            "/api/tts": self._tts,
            "/api/cache/clear": self._clear_cache,
        }
        handler = routes.get(parsed.path)
        if not handler:
            self.send_error(404, "Endpoint necunoscut")
            return None
        return handler()

    def _start_session(self):
        session_id = self._read_json().get("session_id", "default")
        demo = create_demo_session()
        greeting = demo.start_message()
        SESSIONS[session_id] = demo
        self._send_json(
            {
                "bot": greeting,
                "transcript": demo.state.transcript,
                "knowledge_base": demo.knowledge_base_context(),
                "name": "Banutel",
            }
        )

    def _message(self):
        payload = self._read_json()
        session_id = payload.get("session_id", "default")
        user_text = str(payload.get("message", "")).strip()
        demo = get_demo_session(session_id)
        if not demo.state.transcript:
            demo.start_message()
        if not user_text:
            return self._send_json({"error": "Mesaj gol"}, status=400)
        bot_text = demo.handle_user_message_llm_first(user_text)
        self._send_json(
            {
                "bot": bot_text,
                "transcript": demo.state.transcript,
                "knowledge_base": demo.knowledge_base_context(),
            }
        )

    def _voice_message(self):
        payload = self._read_json()
        session_id = payload.get("session_id", "default")
        key = str(payload.get("key") or os.getenv("ZEVO_API_KEY", "")).strip()
        audio_base64 = str(payload.get("audio_base64", "")).strip()
        if not key:
            return self._send_json({"error": "Lipseste ZEVO_API_KEY"}, status=400)
        if not audio_base64:
            return self._send_json({"error": "Lipseste audio_base64"}, status=400)

        audio_data = base64.b64decode(audio_base64)
        stt_raw = asyncio.run(speech_to_text_ws(audio_data, key, DEFAULT_DOMAIN_STT_GENERAL))
        try:
            stt_payload = json.loads(stt_raw)
        except json.JSONDecodeError:
            return self._send_json({"error": f"Raspuns STT invalid: {stt_raw}"}, status=502)
        if "error" in stt_payload:
            return self._send_json({"error": stt_payload.get("details") or stt_payload["error"]}, status=502)

        user_text = (stt_payload.get("text_pp") or stt_payload.get("text") or "").strip()
        if not user_text:
            return self._send_json({"error": "Zevo STT nu a returnat transcript"}, status=502)

        demo = get_demo_session(session_id)
        if not demo.state.transcript:
            demo.start_message()
        bot_text = demo.handle_user_message_llm_first(user_text)
        self._send_json(
            {
                "user": user_text,
                "bot": bot_text,
                "transcript": demo.state.transcript,
                "knowledge_base": demo.knowledge_base_context(),
            }
        )

    def _analyze(self):
        payload = self._read_json()
        session_id = payload.get("session_id", "default")
        model_config = payload.get("model_config") or {}
        demo = SESSIONS.get(session_id)
        if not demo:
            return self._send_json({"error": "Sesiune inexistenta"}, status=404)
        evaluation = build_pipeline_evaluation(demo.state.transcript, model_config)
        self._send_json(
            {
                "evaluation": evaluation["results"],
                "pipeline": evaluation,
                "transcript": demo.state.transcript,
                "knowledge_base": demo.knowledge_base_context(),
            }
        )

    def _evaluation_options(self):
        self._send_json(
            {
                "models": EVALUATION_MODELS,
                "recommendations": TASK_RECOMMENDATIONS,
                "env_status": env_status(),
                "execution_modes": {
                    "local": "Evaluator local, fără chei API, util pentru demo rapid.",
                    "real": "Trimite prompturile v4 către OpenAI/Gemini/Ollama, în funcție de modelul ales.",
                },
                "tasks": ["intent", "final_status", "incongruities"],
                "note": "Alege Local pentru demo fără chei sau Model real pentru apeluri OpenAI/Gemini/Ollama.",
            }
        )

    def _env_status(self):
        self._send_json(env_status())

    def _evaluate_conversation(self):
        payload = self._read_json()
        text = str(payload.get("conversation_text", "")).strip()
        model_config = payload.get("model_config") or {}
        if not text:
            return self._send_json({"error": "Lipseste conversatia de evaluat"}, status=400)
        try:
            transcript = parse_conversation_text(text)
        except (ValueError, json.JSONDecodeError) as exc:
            return self._send_json({"error": str(exc)}, status=400)
        evaluation = build_pipeline_evaluation(transcript, model_config)
        self._send_json(
            {
                "transcript": transcript,
                "evaluation": evaluation,
                "knowledge_base": knowledge_base_context_for(transcript),
            }
        )

    def _tts(self):
        payload = self._read_json()
        text = normalize_for_tts(str(payload.get("text", "")).strip())
        voice = str(payload.get("voice") or os.getenv("ZEVO_TTS_VOICE", "gia"))
        key = str(payload.get("key") or os.getenv("ZEVO_API_KEY", "")).strip()
        if not text:
            return self._send_json({"error": "Text gol"}, status=400)
        if not key:
            return self._send_json({"error": "Lipseste ZEVO_API_KEY"}, status=400)

        TTS_CACHE.mkdir(exist_ok=True)
        digest = hashlib.sha256(f"{voice}:{text}".encode("utf-8")).hexdigest()[:24]
        wav_path = TTS_CACHE / f"{digest}.wav"
        if not wav_path.exists():
            params = TTSRequestParams(
                key=key,
                text=text,
                voice=voice,
                audio_format=DEFAULT_AUDIO_FORMAT_TTS,
                sample_rate=DEFAULT_SAMPLE_RATE_TTS,
                bits_per_sample=DEFAULT_BITS_PER_SAMPLE_TTS,
            )
            audio_data = asyncio.run(text_to_speech_ws(ZEVO_TTS_API_URI, params))
            if not audio_data:
                return self._send_json({"error": "Zevo TTS nu a returnat audio"}, status=502)
            self._write_wav(wav_path, audio_data)
        self._send_json({"audio_url": f"/tts_cache/{wav_path.name}", "cached": True})

    def _clear_cache(self):
        if TTS_CACHE.exists():
            shutil.rmtree(TTS_CACHE)
        TTS_CACHE.mkdir(exist_ok=True)
        self._send_json({"ok": True, "message": "Cache-ul TTS a fost golit."})

    def _serve_cache_file(self, path: str):
        name = Path(path).name
        target = TTS_CACHE / name
        if not target.exists() or target.suffix.lower() != ".wav":
            self.send_error(404, "Fisier audio inexistent")
            return None
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def _send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _write_wav(path: Path, audio_data: bytes):
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(DEFAULT_BITS_PER_SAMPLE_TTS // 8)
            wav_file.setframerate(DEFAULT_SAMPLE_RATE_TTS)
            wav_file.writeframes(audio_data)


def build_pipeline_evaluation(transcript: List[Turn], model_config: Dict[str, str]) -> Dict[str, object]:
    execution_mode = str(model_config.get("execution_mode") or "local")
    local_results = ConversationEvaluator().evaluate(transcript).to_dict()
    raw_results = {}
    tasks = {}
    for task, local_result in local_results.items():
        recommendation = TASK_RECOMMENDATIONS[task]
        selected_model = model_config.get(task) or model_config.get("model") or recommendation["model"]
        if selected_model not in EVALUATION_MODELS:
            selected_model = recommendation["model"]
        task_config = resolve_task_prompt_config(task, selected_model, model_config)
        prompt_context = load_prompt_context(task, task_config)
        task_result = local_result
        raw_response = None
        error = None
        if execution_mode == "real":
            task_result, raw_response, error = evaluate_task_with_real_model(task, transcript, selected_model, task_config)
        raw_results[task] = task_result
        tasks[task] = {
            "model": selected_model,
            "model_label": EVALUATION_MODELS[selected_model]["label"],
            "provider": EVALUATION_MODELS[selected_model]["provider"],
            "execution_mode": execution_mode,
            "recommended_model": recommendation["model"],
            "is_recommended": selected_model == recommendation["model"],
            "lang": task_config["lang"],
            "prompt_version": task_config["prompt_version"],
            "prompt_selection_note": task_config.get("note"),
            "recommendation": recommendation,
            "prompt_context": prompt_context,
            "result": task_result,
        }
        if raw_response is not None:
            tasks[task]["raw_response"] = raw_response
        if error is not None:
            tasks[task]["error"] = error
    return {"results": raw_results, "tasks": tasks}


def create_demo_session() -> BankingVoicebotDemo:
    return BankingVoicebotDemo(llm_fallback=generate_live_llm_reply)


def get_demo_session(session_id: str) -> BankingVoicebotDemo:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = create_demo_session()
    return SESSIONS[session_id]


def format_kb_examples_for_live_prompt(examples: object) -> str:
    if not isinstance(examples, list):
        return ""
    formatted_examples = []
    for item in examples[:4]:
        if not isinstance(item, dict):
            continue
        turns = item.get("turns") if isinstance(item.get("turns"), list) else []
        turns_text = "\n".join(
            f"  {str(turn.get('role', '')).upper()}: {turn.get('text', '')}"
            for turn in turns[:12]
            if isinstance(turn, dict)
        )
        if not turns_text:
            turns_text = (
                f"  USER: {item.get('first_user_message', '')}\n"
                f"  ASSISTANT: {item.get('assistant_resolution', '')}"
            )
        formatted_examples.append(
            "\n".join(
                [
                    f"Exemplu {item.get('conversation_id', '')}",
                    f"intent={item.get('mapped_intent', 'necunoscut')}; "
                    f"status={item.get('final_status', 'necunoscut')}; "
                    f"scor_similaritate={item.get('score', 0)}",
                    turns_text,
                ]
            )
        )
    return "\n\n".join(formatted_examples)


def generate_live_llm_reply(transcript: List[Turn], user_text: str, knowledge_base: Dict[str, object]) -> Optional[str]:
    if not os.getenv("GOOGLE_API_KEY"):
        return None
    examples = knowledge_base.get("examples") if isinstance(knowledge_base, dict) else []
    examples_text = format_kb_examples_for_live_prompt(examples)
    recent_turns = "\n".join(
        f"{turn.get('role', '').upper()}: {turn.get('text', '')}"
        for turn in transcript[-8:]
    )
    prompt = f"""
Ești Bănuțel, un voicebot demonstrativ pentru asistență bancară în limba română.
Răspunzi printr-un flow RAG dataset-first: conversația live este ghidată mai întâi de exemplele similare din dataset, apoi de raționamentul tău general când datasetul nu acoperă complet cazul. Răspunde natural, politicos și concis, în maximum două propoziții.

Pipeline obligatoriu:
1. Uită-te la transcriptul recent și la exemplele similare din dataset.
2. Alege exemplul sau tiparul cel mai apropiat, dacă există unul relevant.
3. Continuă conversația cu următorul pas logic din acel tipar, nu cu o soluție inventată.
4. Dacă exemplele sunt slabe, conflictuale sau nu acoperă mesajul, cere o clarificare scurtă sau răspunde ca asistent bancar demonstrativ.

Reguli:
- Nu folosi reguli hardcodate pentru un singur caz; generalizează din exemplele similare.
- Dacă utilizatorul este într-un flux deja început, continuă acel flux și cere informația următoare necesară, fără să sari la recomandări externe.
- Dacă utilizatorul schimbă subiectul în aceeași conversație, continuă cu noul subiect și nu rămâne blocat în fluxul anterior.
- Dacă utilizatorul salută, întreabă „ce faci” sau mulțumește, răspunde firesc și invită-l să continue.
- Dacă întrebarea este bancară, ajută-l la nivel de demo: carduri, conturi, sold, extras, tranzacții suspecte, date personale, programări, resetare acces, comisioane sau produse.
- Dacă lipsește o informație necesară, cere exact acea informație, fără să inventezi.
- Dacă întrebarea nu este bancară, redirecționează blând spre ce poate face demo-ul, fără formula rigidă „pot răspunde doar”.
- Nu inventa date personale reale, solduri reale, coduri, decizii bancare reale sau politici bancare reale. Marchează răspunsul ca demo când este nevoie.
- Nu copia mecanic ultimul răspuns dintr-un exemplu dacă utilizatorul live este într-o etapă diferită; folosește exemplul ca flow de referință.
- Nu folosi Markdown.

Exemple similare din knowledge base:
{examples_text or "Nu există exemple relevante."}

Transcript recent:
{recent_turns}

Ultimul mesaj utilizator:
{user_text}

Răspuns Bănuțel:
""".strip()
    try:
        from google import genai

        client = genai.Client()
        response = client.models.generate_content(model=LIVE_LLM_MODEL, contents=prompt)
        text = (response.text or "").strip()
    except Exception:
        return None
    return text[:700] if text else None


def env_status() -> Dict[str, object]:
    dotenv_paths = [PROJECT_ROOT / ".env", ROOT / ".env"]
    return {
        "dotenv_paths_checked": [str(path) for path in dotenv_paths],
        "dotenv_found": [str(path) for path in dotenv_paths if path.exists()],
        "openai_api_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
        "google_api_key_loaded": bool(os.getenv("GOOGLE_API_KEY")),
        "supported_aliases": {
            "OPENAI_API_KEY": ENV_ALIASES["OPENAI_API_KEY"],
            "GOOGLE_API_KEY": ENV_ALIASES["GOOGLE_API_KEY"],
        },
    }


def resolve_task_prompt_config(task: str, model_key: str, model_config: Dict[str, str]) -> Dict[str, str]:
    recommendation = TASK_RECOMMENDATIONS[task]
    base = MODEL_TASK_PROMPTS.get(task, {}).get(model_key) or MODEL_TASK_PROMPTS.get(task, {}).get("default") or {}
    lang = model_config.get(f"{task}_lang") or base.get("lang") or recommendation["lang"]
    version = model_config.get(f"{task}_prompt_version") or base.get("prompt_version") or recommendation["prompt_version"]
    return {
        "lang": lang,
        "prompt_version": version,
        "note": base.get("note") or "configurație recomandată pentru task",
    }


def evaluate_task_with_real_model(task: str, transcript: List[Turn], model_key: str, task_config: Dict[str, str]):
    if not EVALUATION_MODELS[model_key].get("supports_real"):
        return (
            {
                "error": "model_nongenerativ",
                "message": "Acest model este baseline encoder și nu poate primi prompturi conversaționale în demo-ul web.",
            },
            None,
            "Modelul selectat nu suportă evaluare generativă reală în pagina web.",
        )
    try:
        from src.evaluation_pipeline.prompting import render_prompt
        from src.evaluation_pipeline.providers import evaluate_with_provider

        prompt, _prompt_meta = render_prompt(
            task=task,
            conversation=transcript,
            lang=task_config["lang"],
            version=task_config["prompt_version"],
        )
        result, raw_response = evaluate_with_provider(
            task=task,
            conversation=transcript,
            model_key=model_key,
            prompt=prompt,
            provider="auto",
        )
        return result, raw_response, None
    except Exception as exc:
        return (
            {
                "error": "model_call_failed",
                "message": str(exc),
            },
            None,
            str(exc),
        )


def load_prompt_context(task: str, task_config: Dict[str, str]) -> Dict[str, object]:
    definitions_file = TASK_RECOMMENDATIONS[task].get("definitions_file")
    few_shot_file = few_shot_file_for(task, task_config["lang"])
    prompt_file = prompt_file_for(task, task_config["lang"], task_config["prompt_version"])
    definitions = load_json_file(definitions_file)
    examples = load_json_file(few_shot_file)
    labels = definitions.get("labels", []) if isinstance(definitions, dict) else []
    return {
        "prompt_file": prompt_file,
        "prompt_exists": path_exists(prompt_file),
        "definitions_file": definitions_file,
        "class_labels": [item.get("name") for item in labels if isinstance(item, dict)],
        "class_count": len(labels),
        "few_shot_file": few_shot_file,
        "few_shot_examples_loaded": count_examples(examples),
    }


def few_shot_file_for(task: str, lang: str) -> str | None:
    if task == "intent":
        return "configs/few_shot_examples_intent.json"
    if task == "final_status":
        return "configs/few_shot_examples_final_status_en.json" if lang == "en" else "configs/few_shot_examples_final_status.json"
    if task == "incongruities":
        return "configs/few_shot_examples_incongruities.json"
    return None


def prompt_file_for(task: str, lang: str, version: str) -> str | None:
    try:
        from src.evaluation_pipeline.config import PROMPT_FILES

        return f"prompts/{PROMPT_FILES[task][(lang, version)]}"
    except Exception:
        return TASK_RECOMMENDATIONS.get(task, {}).get("prompt_file")


def path_exists(relative_path: str | None) -> bool:
    if not relative_path:
        return False
    return (PROJECT_ROOT / relative_path).exists()


def load_json_file(relative_path: str | None):
    if not relative_path:
        return None
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def count_examples(data) -> int:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict) and isinstance(data.get("examples"), list):
        return len(data["examples"])
    return 0


def parse_conversation_text(text: str) -> List[Turn]:
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        data = json.loads(text)
        turns = data.get("turns", []) if isinstance(data, dict) else data
        return normalize_turns(turns)

    turns: List[Turn] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError("Fiecare replica trebuie sa fie de forma USER: text sau ASSISTANT: text.")
        role, message = line.split(":", 1)
        message = message.strip()
        if message:
            turns.append({"role": normalize_role(role), "text": message})
    if not turns:
        raise ValueError("Nu am gasit replici valide in conversatie.")
    return turns


def normalize_turns(turns) -> List[Turn]:
    normalized = []
    for turn in turns:
        text = str(turn.get("text", "")).strip()
        if text:
            normalized.append({"role": normalize_role(str(turn.get("role", ""))), "text": text})
    if not normalized:
        raise ValueError("JSON-ul nu contine turns valide.")
    return normalized


def normalize_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized in {"user", "utilizator", "client", "tu"}:
        return "user"
    if normalized in {"assistant", "bot", "voicebot", "banutel", "bănuțel"}:
        return "assistant"
    raise ValueError(f"Rol necunoscut: {role}. Foloseste USER sau ASSISTANT.")


def knowledge_base_context_for(transcript: List[Turn]) -> Dict[str, object]:
    demo = BankingVoicebotDemo()
    demo.state.transcript = transcript
    return demo.knowledge_base_context()


def main():
    parser = argparse.ArgumentParser(description="Interfata web pentru demo-ul Banutel.")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    if not STATIC_ROOT.exists():
        raise SystemExit(f"Lipseste directorul UI: {STATIC_ROOT}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), BanutilHandler)
    print(f"Banutel porneste la http://127.0.0.1:{args.port}")
    print("Apasa Ctrl+C ca sa opresti serverul.")
    server.serve_forever()


if __name__ == "__main__":
    main()
