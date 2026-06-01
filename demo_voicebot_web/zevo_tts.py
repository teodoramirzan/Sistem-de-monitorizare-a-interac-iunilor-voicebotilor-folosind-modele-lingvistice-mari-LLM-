#!/usr/bin/env python3

import asyncio
import websockets
import numpy as np
import sounddevice as sd
import json
import wave
from datetime import datetime # Pentru timestamp-uri
import os

# Constante
DEFAULT_VOICE_TTS = 'gia' # Vocea implicită Zevo
DEFAULT_SAMPLE_RATE_TTS = 22050
DEFAULT_AUDIO_FORMAT_TTS = "WAV_PCM" # Zevo suportă diverse formate
DEFAULT_BITS_PER_SAMPLE_TTS = 16
ZEVO_TTS_API_URI = 'wss://api-tts.zevo-tech.com:2083'

class TTSRequestParams:
    """Clasa pentru a stoca parametrii cererii TTS."""
    def __init__(self, key, text='Acesta este un test.',
                 voice=DEFAULT_VOICE_TTS, audio_format=DEFAULT_AUDIO_FORMAT_TTS, 
                 sample_rate=DEFAULT_SAMPLE_RATE_TTS,
                 pace=1.0, pitch=0, bits_per_sample=DEFAULT_BITS_PER_SAMPLE_TTS,
                 output_filename="tts_output.wav"): # output_filename e mai mult un default dacă se salvează
        self.key = key
        self.text = text # Poate fi text simplu sau SSML
        self.voice = voice
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.pace = pace
        self.pitch = pitch
        self.bits_per_sample = bits_per_sample
        self.output_filename = output_filename


def construct_tts_message(params: TTSRequestParams):
    """Construiește mesajul JSON pentru API-ul Zevo TTS."""
    # Documentația Zevo ar trebui să specifice dacă SSML-ul se pune direct în câmpul "text"
    # sau dacă necesită un alt flag/parametru. Presupunem că se pune direct în "text".
    return json.dumps({
        "task": [
            {"text": params.text}, # Aici se poate pune și SSML
            {"voice": params.voice},
            {"key": params.key},
            {"pace": str(params.pace)},
            {"pitch": str(params.pitch)},
            {"audio_format": params.audio_format},
            {"bits_per_sample": str(params.bits_per_sample)},
            {"sample_rate": str(params.sample_rate)}
        ]
    })


async def text_to_speech_ws(api_uri, params: TTSRequestParams):
    """Funcția async care comunică cu API-ul Zevo TTS prin WebSocket."""
    try:
        # Setează timeout-uri mai mari pentru conexiuni TTS, deoarece generarea audio poate dura
        async with websockets.connect(api_uri, max_size=10000000, ping_interval=30, ping_timeout=30, close_timeout=20) as websocket:
            message = construct_tts_message(params)
            await websocket.send(message)
            
            # API-ul Zevo TTS trimite datele audio direct ca un mesaj binar.
            # Pot exista mesaje de status JSON înainte, dar datele audio sunt binare.
            result = await websocket.recv() # Așteaptă răspunsul (care ar trebui să fie datele audio)

            if isinstance(result, str):
                # Dacă primim un string, este probabil un mesaj de eroare sau status de la Zevo
                print(f"({datetime.now().strftime('%H:%M:%S')}) TTS Mesaj Zevo (string): {result}")
                try:
                    # Încercăm să vedem dacă e un JSON de eroare
                    error_json = json.loads(result)
                    if "error" in error_json or "message" in error_json:
                         return None # Indică eroare
                except json.JSONDecodeError:
                    pass # Nu era JSON, dar tot e un string neașteptat în loc de audio
                return None # Indică faptul că nu s-a primit audio valid
            elif isinstance(result, bytes):
                # Acesta este cazul așteptat: date audio binare
                return result
            else:
                print(f"({datetime.now().strftime('%H:%M:%S')}) TTS Tip de răspuns neașteptat de la Zevo: {type(result)}")
                return None

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"({datetime.now().strftime('%H:%M:%S')}) TTS Eroare critică: Conexiunea WebSocket închisă prematur: {e}")
        return None
    except asyncio.TimeoutError:
        print(f"({datetime.now().strftime('%H:%M:%S')}) TTS Eroare critică: Timeout la operațiunea WebSocket.")
        return None
    except Exception as e:
        print(f"({datetime.now().strftime('%H:%M:%S')}) TTS Eroare critică în text_to_speech_ws: {e}")
        return None


def play_audio_data(audio_data, sample_rate):
    """Redă datele audio folosind sounddevice."""
    try:
        # Zevo TTS returnează de obicei PCM 16-bit.
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        sd.play(audio_array, samplerate=sample_rate)
        sd.wait() # Așteaptă până când redarea se termină
    except Exception as e:
        print(f"({datetime.now().strftime('%H:%M:%S')}) TTS Eroare la redarea audio: {e}")


def save_audio_to_file(audio_data, filename, sample_rate, bits_per_sample):
    """Salvează datele audio într-un fișier WAV."""
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Presupunem audio mono
            wf.setsampwidth(bits_per_sample // 8) # Ex: 16 bits -> 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        print(f"({datetime.now().strftime('%H:%M:%S')}) TTS: Audio salvat în {filename}")
    except Exception as e:
        print(f"({datetime.now().strftime('%H:%M:%S')}) TTS Eroare la salvarea fișierului audio: {e}")


def speak_text_zevo(api_key, text_to_speak, voice=DEFAULT_VOICE_TTS, 
                    sample_rate=DEFAULT_SAMPLE_RATE_TTS, 
                    save_to_file=False, filename="tts_cache/default_tts_output.wav", # Salvează într-un subdirector
                    use_ssml=False): # Flag pentru a indica dacă text_to_speak este SSML
    """
    Funcția principală pentru a converti text (sau SSML) în vorbire folosind Zevo TTS și a-l reda.
    Opcional, salvează audio-ul într-un fișier.
    """
    if not text_to_speak:
        print(f"({datetime.now().strftime('%H:%M:%S')}) TTS: Text gol furnizat, nu se sintetizează nimic.")
        return

    # Pentru SSML, textul este deja formatat. Pentru text simplu, e doar textul.
    # Documentația Zevo ar trebui să clarifice dacă e nevoie de un flag special pentru SSML
    # sau dacă API-ul detectează automat tag-urile <speak>.
    # Presupunem că API-ul gestionează SSML dacă este trimis în câmpul "text".
    
    # Creează directorul cache dacă nu există și se dorește salvarea
    if save_to_file:
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"({datetime.now().strftime('%H:%M:%S')}) TTS Eroare la crearea directorului cache '{output_dir}': {e}")
                save_to_file = False # Nu se poate salva, continuă fără salvare

    params = TTSRequestParams(key=api_key, text=text_to_speak, voice=voice, 
                              sample_rate=sample_rate, 
                              audio_format=DEFAULT_AUDIO_FORMAT_TTS, # Asigură-te că formatul e compatibil
                              bits_per_sample=DEFAULT_BITS_PER_SAMPLE_TTS)
    
    # Afișează doar începutul textului dacă e lung, pentru lizibilitate în loguri
    log_text = text_to_speak if len(text_to_speak) < 70 else text_to_speak[:67] + "..."
    print(f"({datetime.now().strftime('%H:%M:%S')}) TTS: Se trimite la Zevo (voce: {voice}): '{log_text}'")
    
    audio_data = asyncio.run(text_to_speech_ws(ZEVO_TTS_API_URI, params))

    if audio_data:
        print(f"({datetime.now().strftime('%H:%M:%S')}) TTS: Audio primit ({len(audio_data)} bytes), se redă...")
        play_audio_data(audio_data, params.sample_rate)
        if save_to_file:
            save_audio_to_file(audio_data, filename, params.sample_rate, params.bits_per_sample)
    else:
        print(f"({datetime.now().strftime('%H:%M:%S')}) TTS: Nu s-au primit date audio valide de la Zevo.")


if __name__ == '__main__':
    API_KEY_ZEVO = 'icvsilab2025' # Cheia API furnizată
    VOICE_TTS_TEST = 'gia' # Sau 'anca', 'radu' etc. conform documentației Zevo

    print(f"({datetime.now().strftime('%H:%M:%S')}) Inițiere test Zevo TTS...")
    
    text_simplu = "Bună ziua! Acesta este un test de vorbire cu vocea Gia."
    speak_text_zevo(API_KEY_ZEVO, text_simplu, voice=VOICE_TTS_TEST, save_to_file=True, filename="tts_cache/test_simplu_gia.wav")

    # Exemplu cu o altă voce, dacă este disponibilă și dorită
    # speak_text_zevo(API_KEY_ZEVO, "Testare cu vocea Radu.", voice='radu', save_to_file=True, filename="tts_cache/test_radu.wav")

    # Exemplu pentru SSML (Speech Synthesis Markup Language)
    # SSML permite control mai fin asupra pronunției, pauzelor, etc.
    # Asigură-te că formatul SSML este corect și compatibil cu Zevo.
    text_ssml_exemplu = """
    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="ro-RO">
      Programarea dumneavoastră este pe data de 
      <say-as interpret-as="date" format="dmy">26.05.2025</say-as>.
      <break time="500ms"/>
      Vă așteptăm!
    </speak>
    """
    print(f"\n({datetime.now().strftime('%H:%M:%S')}) Testare TTS cu SSML...")
    # speak_text_zevo(API_KEY_ZEVO, text_ssml_exemplu, voice=VOICE_TTS_TEST, use_ssml=True, save_to_file=True, filename="tts_cache/test_ssml.wav")
    # Momentan, flag-ul use_ssml nu schimbă logica de trimitere, presupunând că Zevo detectează SSML.
    # Dacă Zevo necesită un parametru special pentru SSML, funcția construct_tts_message ar trebui adaptată.
    # Pentru test, trimitem direct string-ul SSML.
    speak_text_zevo(API_KEY_ZEVO, text_ssml_exemplu.strip(), voice=VOICE_TTS_TEST, save_to_file=True, filename="tts_cache/test_ssml.wav")

    print(f"\n({datetime.now().strftime('%H:%M:%S')}) Teste TTS finalizate.")
