# Demo web Bănuțel

Acest folder conține interfața HTML pentru demo-ul live și pentru pagina de evaluare a unei conversații complete.

## Rulare din repo-ul de dizertație

Din rădăcina repo-ului:

```powershell
python demo_voicebot_web/web_demo_server.py
```

Apoi deschide:

```text
http://127.0.0.1:8787
```

## Ce include

- tab `Live` cu conversație text, voce și telefon;
- tab `Evaluator`, unde poți lipi o conversație completă `USER:` / `ASSISTANT:`;
- selector de model pentru fiecare task;
- recomandări pentru modelele cele mai bune pe task;
- knowledge base local din `data/master_dataset_refined_180.json`.

## Rulare cu modele reale

În selectorul `Mod rulare`, alege `Model real/API/Ollama`.

Pentru OpenAI:

```powershell
$env:OPENAI_API_KEY="cheia_ta"
```

Pentru Gemini:

```powershell
$env:GOOGLE_API_KEY="cheia_ta"
```

Pentru Ollama:

```powershell
ollama pull mistral:7b
ollama pull qwen2.5:3b
ollama serve
```

Instalează dependențele pipeline-ului din rădăcina repo-ului:

```powershell
pip install -r requirements-evaluation-pipeline.txt
```

În modul real, fiecare task folosește promptul configurat pentru model/task și încarcă exemplele few-shot pentru `v4`.
