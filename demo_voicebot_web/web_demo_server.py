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
from typing import Dict, List
from urllib.parse import urlparse

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


ROOT = Path(__file__).resolve().parent
STATIC_ROOT = ROOT / "web_ui"
TTS_CACHE = ROOT / "tts_cache"
SESSIONS: Dict[str, BankingVoicebotDemo] = {}

EVALUATION_MODELS = {
    "openai_o3": {"label": "OpenAI o3", "kind": "API"},
    "gemini_2.5_flash": {"label": "Gemini 2.5 Flash", "kind": "API"},
    "aya_expanse_8b": {"label": "Aya Expanse 8B", "kind": "local"},
    "rollama2_7b": {"label": "RoLLaMA 2 7B", "kind": "local"},
    "roberta_encoder": {"label": "XLM-RoBERTa encoder", "kind": "local"},
    "mistral_7b": {"label": "Mistral 7B", "kind": "local"},
    "qwen2.5_3b": {"label": "Qwen2.5 3B", "kind": "local"},
}

TASK_RECOMMENDATIONS = {
    "intent": {
        "model": "openai_o3",
        "lang": "en",
        "prompt_version": "v4",
        "metric": "accuracy/F1 98.3%",
        "source": "evaluation_report_intent.txt",
    },
    "final_status": {
        "model": "openai_o3",
        "lang": "ro",
        "prompt_version": "v4",
        "metric": "recomandare derivata din notebook; rezultate JSON lipsa",
        "source": "outputs_final_status/final_status_experiments.ipynb",
    },
    "incongruities": {
        "model": "gemini_2.5_flash",
        "lang": "ro",
        "prompt_version": "v4",
        "metric": "binary F1 0.8219, type macro F1 0.8531",
        "source": "outputs_incongruities/exp_inc_gemini_2.5_flash__ro__v4.json",
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
            "/api/evaluate-conversation": self._evaluate_conversation,
            "/api/tts": self._tts,
            "/api/cache/clear": self._clear_cache,
            "/api/telephony/inbound": self._telephony_inbound,
            "/api/telephony/status": self._telephony_status,
        }
        handler = routes.get(parsed.path)
        if not handler:
            self.send_error(404, "Endpoint necunoscut")
            return None
        return handler()

    def _start_session(self):
        session_id = self._read_json().get("session_id", "default")
        demo = BankingVoicebotDemo()
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
        demo = SESSIONS.setdefault(session_id, BankingVoicebotDemo())
        if not demo.state.transcript:
            demo.start_message()
        if not user_text:
            return self._send_json({"error": "Mesaj gol"}, status=400)
        bot_text = demo.handle_user_message(user_text)
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

        demo = SESSIONS.setdefault(session_id, BankingVoicebotDemo())
        if not demo.state.transcript:
            demo.start_message()
        bot_text = demo.handle_user_message(user_text)
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
                "tasks": ["intent", "final_status", "incongruities"],
                "note": "In pagina web, evaluatorul ruleaza local pentru demo. Selectia de model arata configuratia si recomandarea pe task.",
            }
        )

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

    def _telephony_status(self):
        phone_number = os.getenv("ZEVO_PHONE_NUMBER", "").strip()
        public_url = os.getenv("PUBLIC_WEBHOOK_URL", "").strip()
        configured = bool(phone_number)
        self._send_json(
            {
                "configured": configured,
                "phone_number": phone_number,
                "public_webhook_url": public_url,
                "webhook_url": "/api/telephony/inbound",
                "note": "Pentru apeluri reale, numarul Zevo trebuie sa trimita transcriptul catre webhook-ul public al acestui server.",
            }
        )

    def _telephony_inbound(self):
        payload = self._read_json()
        session_id = str(payload.get("call_id") or payload.get("session_id") or "phone-default")
        user_text = str(payload.get("transcript") or payload.get("message") or "").strip()
        demo = SESSIONS.setdefault(session_id, BankingVoicebotDemo())
        if not demo.state.transcript:
            demo.start_message()
        if not user_text:
            return self._send_json({"error": "Lipseste transcriptul apelului"}, status=400)
        bot_text = demo.handle_user_message(user_text)
        self._send_json(
            {
                "reply_text": bot_text,
                "transcript": demo.state.transcript,
                "knowledge_base": demo.knowledge_base_context(),
            }
        )

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
    raw_results = ConversationEvaluator().evaluate(transcript).to_dict()
    tasks = {}
    for task, task_result in raw_results.items():
        recommendation = TASK_RECOMMENDATIONS[task]
        selected_model = model_config.get(task) or model_config.get("model") or recommendation["model"]
        if selected_model not in EVALUATION_MODELS:
            selected_model = recommendation["model"]
        tasks[task] = {
            "model": selected_model,
            "model_label": EVALUATION_MODELS[selected_model]["label"],
            "recommended_model": recommendation["model"],
            "is_recommended": selected_model == recommendation["model"],
            "lang": model_config.get(f"{task}_lang") or recommendation["lang"],
            "prompt_version": model_config.get(f"{task}_prompt_version") or recommendation["prompt_version"],
            "recommendation": recommendation,
            "result": task_result,
        }
    return {"results": raw_results, "tasks": tasks}


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
