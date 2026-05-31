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

În pagina web, evaluarea este calculată local pentru demo, iar modelul ales apare ca metadată de comparație. Pentru rulări reale cu OpenAI, Gemini sau Ollama, folosește pipeline-ul CLI din `src/evaluation_pipeline`.
