# zevo_stt.py
import os
import sys
import json
import asyncio
import websockets
import speech_recognition as sr
from datetime import datetime # Pentru timestamp-uri

# Constante
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_DOMAIN_STT_GENERAL = 'ro-RO_general-2025.1' # Domeniu general
AMBIENT_NOISE_DURATION = 1 
LISTEN_TIMEOUT = 7 
LISTEN_PHRASE_TIMEOUT = 15 # Poate fi ajustat sau eliminat dacă nu e necesară o limită strictă de frază

def suppress_alsa_warnings():
    if os.name != 'posix':
        return None
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        stderr_fd_original = os.dup(sys.stderr.fileno())
        sys.stderr.flush()
        os.dup2(null_fd, sys.stderr.fileno())
        os.close(null_fd)
        return stderr_fd_original
    except OSError as e:
        print(f"Atenție: Nu s-a putut suprima output-ul ALSA: {e}")
        return None

def restore_stderr(original_stderr_fd):
    if original_stderr_fd is not None and os.name == 'posix':
        try:
            sys.stderr.flush()
            os.dup2(original_stderr_fd, sys.stderr.fileno())
            os.close(original_stderr_fd)
        except OSError as e:
            print(f"Atenție: Nu s-a putut restaura stderr: {e}")

async def speech_to_text_ws(audio_data, api_key, domain, 
                            phrases_list=None, # Păstrăm parametrul, dar nu îl vom folosi în testul default
                            sample_rate=DEFAULT_SAMPLE_RATE, 
                            chunk_size=16000,
                            server_uri="wss://live-transcriber.zevo-tech.com:2053"):
    """Trimite date audio către API-ul Zevo STT WebSocket și returnează răspunsul JSON."""
    config_data = {
        "key": api_key,
        "sample_rate": str(sample_rate),
        "domain": domain
    }
    
    # Logica pentru phrases_list rămâne, pentru flexibilitate, chiar dacă nu o folosim în testul principal
    if domain == 'ro-RO_phrases' and phrases_list and isinstance(phrases_list, list): # Folosim direct stringul 'ro-RO_phrases'
        config_data["phrases"] = phrases_list 
        print(f"({datetime.now().strftime('%H:%M:%S')}) STT Config: Se folosește domeniul de fraze cu lista: {phrases_list}")

    config_payload = {"config": config_data}

    try:
        async with websockets.connect(server_uri, ping_interval=20, ping_timeout=20, close_timeout=10) as websocket:
            await websocket.send(json.dumps(config_payload))
            initial_response_str = await websocket.recv() 
            
            if isinstance(initial_response_str, str):
                try:
                    init_resp_json = json.loads(initial_response_str)
                    if init_resp_json.get("status") == "error":
                        print(f"({datetime.now().strftime('%H:%M:%S')}) STT Configurare Zevo eșuată (status: error de la server): {initial_response_str}")
                        return json.dumps({"error": "Configurare Zevo eșuată de server", "details": initial_response_str})
                    
                    message_content = init_resp_json.get("message")
                    if isinstance(message_content, str) and message_content.lower() not in ["waiting for audio", "ok", "processing"]:
                        print(f"({datetime.now().strftime('%H:%M:%S')}) STT Configurare Zevo eșuată (mesaj text neașteptat de la server): {initial_response_str}")
                        return json.dumps({"error": "Mesaj de configurare neașteptat de la Zevo", "details": initial_response_str})
                    
                    if not (init_resp_json.get("status") == "ok" or init_resp_json.get("message") == "waiting for audio"):
                         print(f"({datetime.now().strftime('%H:%M:%S')}) STT Răspuns inițial informativ/ACK de la Zevo: {initial_response_str}")

                except json.JSONDecodeError:
                     print(f"({datetime.now().strftime('%H:%M:%S')}) STT Răspuns inițial Zevo: string non-JSON: {initial_response_str}")
                     if initial_response_str.strip().upper() != "OK": 
                        return json.dumps({"error": "Răspuns inițial invalid (text non-JSON)", "details": initial_response_str})
            else:
                print(f"({datetime.now().strftime('%H:%M:%S')}) STT Răspuns inițial Zevo de tip neașteptat: {type(initial_response_str)}")
                return json.dumps({"error": "Tip de răspuns inițial neașteptat", "details": "Serverul nu a răspuns cu un string la configurare."})

            offset = 0
            while offset < len(audio_data):
                chunk = audio_data[offset:offset + chunk_size]
                await websocket.send(chunk)
                response = await websocket.recv() 
                if isinstance(response, str):
                    try:
                        resp_json = json.loads(response)
                        if "status" in resp_json and resp_json["status"] != "ok":
                             print(f"({datetime.now().strftime('%H:%M:%S')}) STT Mesaj Zevo în timpul transcrierii (status ne-ok): {response}")
                        elif "message" in resp_json and resp_json["message"] not in ["ok", "waiting for audio", "processing"]: 
                             print(f"({datetime.now().strftime('%H:%M:%S')}) STT Mesaj Zevo în timpul transcrierii: {response}")
                    except json.JSONDecodeError:
                         pass 
                offset += chunk_size

            await websocket.send('{"eof" : 1}') 
            final_response = await websocket.recv() 
            return final_response

    except websockets.exceptions.ConnectionClosed as e: 
        print(f"({datetime.now().strftime('%H:%M:%S')}) STT Eroare critică: Conexiunea WebSocket închisă: {e.reason} (cod: {e.code})")
        return json.dumps({"error": "Conexiune WebSocket închisă", "details": f"Cod: {e.code}, Motiv: {e.reason}"})
    except asyncio.TimeoutError:
        print(f"({datetime.now().strftime('%H:%M:%S')}) STT Eroare critică: Timeout la operațiunea WebSocket.")
        return json.dumps({"error": "Timeout WebSocket"})
    except Exception as e:
        print(f"({datetime.now().strftime('%H:%M:%S')}) STT Eroare critică în speech_to_text_ws: {type(e).__name__} - {e}")
        return json.dumps({"error": "Eroare generală WebSocket", "details": str(e)})


def record_and_transcribe_from_mic(api_key, domain=DEFAULT_DOMAIN_STT_GENERAL, 
                                   phrases_list=None, 
                                   sample_rate=DEFAULT_SAMPLE_RATE, 
                                   ambient_duration=AMBIENT_NOISE_DURATION, 
                                   listen_timeout=LISTEN_TIMEOUT, 
                                   phrase_timeout=LISTEN_PHRASE_TIMEOUT):
    """
    Înregistrează audio de la microfon, îl trimite la Zevo STT și returnează textul transcris.
    """
    r = sr.Recognizer()
    original_stderr_fd = suppress_alsa_warnings()
    audio_data_object = None 

    try:
        with sr.Microphone(sample_rate=sample_rate) as source:
            print(f"({datetime.now().strftime('%H:%M:%S')}) Ajustare zgomot de fond pentru {ambient_duration} secunde...")
            r.adjust_for_ambient_noise(source, duration=ambient_duration)
            print(f"({datetime.now().strftime('%H:%M:%S')}) Puteți vorbi acum (prag energie: {r.energy_threshold:.0f})...")
            
            try:
                audio_data_object = r.listen(source, timeout=listen_timeout, phrase_time_limit=phrase_timeout)
            except sr.WaitTimeoutError:
                print(f"({datetime.now().strftime('%H:%M:%S')}) STT Timeout: Nu s-a detectat vorbire în {listen_timeout} secunde.")
                return "" 
            
    except sr.RequestError as e: 
        print(f"({datetime.now().strftime('%H:%M:%S')}) STT Eroare API SpeechRecognition: {e}")
        return ""
    except Exception as e: 
        print(f"({datetime.now().strftime('%H:%M:%S')}) STT Eroare la înregistrarea audio: {e}")
        return ""
    finally: 
        if original_stderr_fd is not None:
            restore_stderr(original_stderr_fd)

    if not audio_data_object: 
        print(f"({datetime.now().strftime('%H:%M:%S')}) STT: Nu s-au capturat date audio.")
        return ""

    try:
        print(f"({datetime.now().strftime('%H:%M:%S')}) Se trimite audio către Zevo STT (domeniu: {domain})...")
        wav_audio_data = audio_data_object.get_wav_data(convert_rate=sample_rate, convert_width=2) 

        transcription_json_str = asyncio.run(speech_to_text_ws(wav_audio_data, api_key, domain, phrases_list=phrases_list, sample_rate=sample_rate))
        
        if not transcription_json_str: 
            print(f"({datetime.now().strftime('%H:%M:%S')}) STT: Nu s-a primit răspuns valid de la WebSocket.")
            return ""

        transcription_data = json.loads(transcription_json_str) 

        if "error" in transcription_data: 
            print(f"({datetime.now().strftime('%H:%M:%S')}) STT Eroare de la Zevo (gestionată intern): {transcription_data.get('details', transcription_data['error'])}")
            return ""
        if "message" in transcription_data and transcription_data.get("status") != "ok" and transcription_data.get("message") != "waiting for audio": 
            if isinstance(transcription_data["message"], dict) and "connection_id" in transcription_data["message"]:
                print(f"({datetime.now().strftime('%H:%M:%S')}) STT Răspuns informativ Zevo (nu eroare): {transcription_data['message']}")
            else:
                print(f"({datetime.now().strftime('%H:%M:%S')}) STT Eroare de la Zevo API: {transcription_data['message']}")
                return ""

        transcribed_text = transcription_data.get("text_pp", "") 
        if transcribed_text:
            print(f"({datetime.now().strftime('%H:%M:%S')}) STT Text transcris: '{transcribed_text}'")
        else:
            raw_text = transcription_data.get("text", "")
            if raw_text:
                 print(f"({datetime.now().strftime('%H:%M:%S')}) STT Text transcris (brut, text_pp gol): '{raw_text}'")
                 transcribed_text = raw_text 
            else:
                 print(f"({datetime.now().strftime('%H:%M:%S')}) STT: Transcriere goală primită de la Zevo. Răspuns complet: {transcription_json_str}")
        return transcribed_text

    except json.JSONDecodeError as e:
        print(f"({datetime.now().strftime('%H:%M:%S')}) STT Eroare la parsarea JSON de la Zevo: {e}. Răspuns primit: {transcription_json_str if 'transcription_json_str' in locals() else 'Nedefinit'}")
        return ""
    except Exception as e: 
        print(f"({datetime.now().strftime('%H:%M:%S')}) STT Eroare generală la transcriere Zevo: {type(e).__name__} - {e}")
        return ""


if __name__ == "__main__":
    API_KEY_ZEVO = 'icvsilab2025' 
    
    print(f"({datetime.now().strftime('%H:%M:%S')}) Inițiere test Zevo STT...")
    
    print(f"\n({datetime.now().strftime('%H:%M:%S')}) --- Test cu Domeniul General ---")
    domain_to_test = DEFAULT_DOMAIN_STT_GENERAL
    phrases_for_test = None # Nu folosim listă de fraze pentru domeniul general

    # Dacă vrei să testezi rapid doar domeniul de fraze, poți decomenta următoarele și comenta testul general:
    # print(f"\n({datetime.now().strftime('%H:%M:%S')}) --- Test cu Domeniul Fraze ---")
    # domain_to_test = DEFAULT_DOMAIN_STT_PHRASES
    # phrases_for_test = ["da", "nu", "confirm", "anulez", "programare", "stop"] 
    # print(f"({datetime.now().strftime('%H:%M:%S')}) Se folosește domeniul: {domain_to_test} cu frazele: {phrases_for_test}")

    print(f"({datetime.now().strftime('%H:%M:%S')}) Se folosește domeniul: {domain_to_test}")
    transcription = record_and_transcribe_from_mic(
        api_key=API_KEY_ZEVO, 
        domain=domain_to_test,
        phrases_list=phrases_for_test,
        listen_timeout=7,  
        phrase_timeout=10  
    )
    if transcription:
        print(f"\n({datetime.now().strftime('%H:%M:%S')}) Text final transcris: '{transcription}'")
    else:
        print(f"\n({datetime.now().strftime('%H:%M:%S')}) Nu s-a putut obține transcrierea sau nu s-a detectat vorbire inteligibilă.")

    print(f"\n({datetime.now().strftime('%H:%M:%S')}) Test STT finalizat.")

