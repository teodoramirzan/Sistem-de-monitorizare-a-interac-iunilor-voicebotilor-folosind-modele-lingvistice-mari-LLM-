# Demo pipeline de evaluare

Acest demo rulează cele 3 taskuri din dizertație pe o conversație dată:

- `intent`
- `final_status`
- `incongruities`

Pipeline-ul poate folosi configurația recomandată pe baza rezultatelor existente sau poate rula cu alt model pentru comparație.

## Recomandări implicite

| Task | Model recomandat | Prompt | Sursă |
|---|---|---|---|
| intent | `openai_o3` | `en` + `v4` | `evaluation_report_intent.txt`, acuratețe/F1 98.3% |
| final_status | `openai_o3` | `ro` + `v4` | notebook-ul existent; nu există JSON de rezultate salvat în repo |
| incongruities | `gemini_2.5_flash` | `ro` + `v4` | `outputs_incongruities/exp_inc_gemini_2.5_flash__ro__v4.json`, binary F1 0.8219 |

Pentru toate variantele `v4`, codul încarcă exemple few-shot:

- `configs/few_shot_examples_intent.json`
- `configs/few_shot_examples_final_status.json`
- `configs/few_shot_examples_final_status_en.json`
- `configs/few_shot_examples_incongruities.json`

## Rulare rapidă fără API

Modul `local` folosește euristici locale și este util pentru verificarea pipeline-ului:

```powershell
python -m src.evaluation_pipeline.cli --conversation-id conv_simple_0001 --task all
```

Pentru randarea completă a prompturilor și rularea cu modele reale:

```powershell
pip install -r requirements-evaluation-pipeline.txt
```

## Rulare cu modelul recomandat

```powershell
python -m src.evaluation_pipeline.cli --conversation-id conv_simple_0001 --task all --provider auto
```

Pentru `openai_o3` este necesară variabila `OPENAI_API_KEY`. Pentru `gemini_2.5_flash` este necesară configurarea clientului Google GenAI. Pentru modelele locale este necesar Ollama.

## Modele disponibile

```powershell
python -m src.evaluation_pipeline.cli --list-models
```

Include modelele studiate și modelele adăugate pentru comparație:

- `openai_o3`
- `gemini_2.5_flash`
- `aya_expanse_8b`
- `rollama2_7b`
- `roberta_encoder`
- `robert_encoder`
- `mistral_7b`
- `qwen2.5_3b`

## Comparație între modele

```powershell
python -m src.evaluation_pipeline.cli --conversation-id conv_simple_0001 --task intent --compare-models openai_o3,gemini_2.5_flash,mistral_7b,qwen2.5_3b --provider auto
```

Pentru Mistral și Qwen prin Ollama:

```powershell
ollama pull mistral:7b
ollama pull qwen2.5:3b
python -m src.evaluation_pipeline.cli --conversation-id conv_simple_0001 --task all --model qwen2.5_3b --provider ollama
```

## Rulare pe o conversație scrisă în terminal

```powershell
$conv = @"
USER: Vreau să-mi blochez cardul.
ASSISTANT: Pentru siguranță, spuneți ultimele 4 cifre.
USER: Revin mai târziu.
"@
python -m src.evaluation_pipeline.cli --task all --text $conv
```

Pentru conversații mai lungi, creează un JSON cu schema:

```json
{
  "conversation_id": "demo_001",
  "turns": [
    {"role": "user", "text": "Vreau să-mi blochez cardul."},
    {"role": "assistant", "text": "Pentru siguranță, spuneți ultimele 4 cifre."}
  ]
}
```

și rulează:

```powershell
python -m src.evaluation_pipeline.cli --conversation-file demo_001.json --task all
```
