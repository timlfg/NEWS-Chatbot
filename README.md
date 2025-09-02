# TR-NEWS Chatbot

A production-ready, Turkish news assistant that can:

- Summarize Turkish news articles (title-style, concise summaries)
- Answer questions about a given news article
- Run multilingual tasks using mT5 (e.g., summarize/translate as a fallback)
- Serve both a FastAPI backend and a Streamlit UI
- Train its own multitask LoRA adapters (summarization + QA) on open datasets

This repo contains end-to-end components: data preparation, training scripts, an API server, a simple web UI, and quick inference tools.

---

## Key Features

- Multitask text-to-text training (summarization + QA) with QLoRA-style adapters
- Hugging Face Transformers + PEFT integration (LoRA)
- Models used:
  - QA: `savasy/mt0-large-Turkish-qa` (PEFT if adapter given, else HF pipeline)
  - Multilingual/fallback: `google/mt5-small`
  - Local fine-tuned summarization: `outputs/multitask-lora-fast` and/or `outputs/multitask-lora`
- FastAPI service with endpoints: `/summarize`, `/qa`, `/multilingual`, `/models/status`, `/ui`
- Streamlit app to drive summaries and QA interactively
- Hardware-aware training auto-tuning (batch size, grad accumulation, dtype)

---

## Repository Structure

```
.
├─ mehmet-updates/
│  ├─ api/
│  │  └─ enhanced_multi_model_api.py   # FastAPI app (serves summarization, QA, multilingual)
│  └─ streamlit_app.py                 # Streamlit front-end for summaries & QA
│
├─ scripts/
│  ├─ prepare_data_2.py                # Build multitask dataset from public sources
│  ├─ train_multitask_qlora_2.py       # Train LoRA adapters (summarization + QA)
│  ├─ quick_infer.py                   # Local CLI for quick summarization/QA with adapter
│  ├─ eval_metrics.py, analysis2.py    # Utilities and analysis
│  └─ ...
│
├─ notebook/
│  ├─ train_analysis.ipynb             # EDA/training analysis
│  └─ sanity_check_notebook.py         # Quick dataset checks
│
├─ data2/                              # Will contain processed datasets (created by prepare script)
├─ outputs/                            # Will contain trained adapters and artifacts
├─ requirements.txt
└─ README.md
```

---

## Requirements

- Python 3.9+
- Recommended: CUDA-capable GPU for training/inference (CPU works but is slow)
- Install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
. .venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Notes:
- `bitsandbytes` may be optional depending on your GPU and quantization needs. If installation fails on Windows, you can still train/run without it.
- Hugging Face will download models on first run. Set `HF_HOME` or `TRANSFORMERS_CACHE` to control cache location if needed.

---

## Data Preparation

Build a multitask (text-to-text) dataset combining:
- Summarization: TR-News (`batubayk/TR-News`) and `xtinge/turkish-extractive-summarization-dataset` (several configs)
- QA: `ucsahin/TR-Extractive-QA-82K`

Run:

```bash
python scripts/prepare_data_2.py
```

By default this creates a Hugging Face dataset at `data2/processed/multitask_text2text`, plus convenience JSONL files under the same directory.

Environment variables (optional):
- `OUT_DIR`: output folder (default `data2/processed/multitask_text2text`)
- `TR_NEWS_NAME`: TR-News dataset id (default `batubayk/TR-News`)
- `SUMM_CONFIGS`: comma-separated configs for `xtinge/turkish-extractive-summarization-dataset` (default `mlsum_tr_ext,tes,xtinge-sum_tr_ext`)
- `QA_DATASET`: QA dataset id (default `ucsahin/TR-Extractive-QA-82K`)
- Sample controls: `MAX_TRNEWS_SUMM_SAMPLES`, `MAX_SUMM_HF_SAMPLES`, `MAX_QA_SAMPLES`
- Split/seed: `TRAIN_RATIO`, `VAL_RATIO`, `SEED`
- Prefixes: `PREFIX_SUMM` (default `"summarize: "`), `PREFIX_QA` (default `"answer: "`)

---

## Training (LoRA Adapters)

Train a small, fast adapter on the prepared dataset:

```bash
python scripts/train_multitask_qlora_2.py
```

Default behavior:
- Base model: `google/mt5-small` (override with `BASE_MODEL`)
- Dataset: `data2/processed/multitask_text2text` (override with `DATA_DIR`)
- Output: `outputs/multitask-lora-fast` (override with `OUTPUT_DIR`)
- Auto-tunes batch size, gradient accumulation, dtype based on your GPU
- Evaluates periodically; writes `metrics.csv`, `eval_predictions.jsonl`, and `test_metrics.txt` (if test split exists)

Useful environment overrides:
- Core: `BASE_MODEL`, `DATA_DIR`, `OUTPUT_DIR`
- Lengths: `MAX_SOURCE_LEN`, `MAX_TARGET_LEN`
- Samples: `MAX_TRAIN_SAMPLES`, `MAX_EVAL_SAMPLES`, `MAX_TEST_SAMPLES`
- Scheduling: `EVAL_STEPS`, `SAVE_STEPS`, `LOG_STEPS`
- Tuning: `BATCH_SIZE`, `GRAD_ACCUM`, `LR`, `EPOCHS`, `SEED`
- Merging: `MERGE_AND_SAVE=1` to export a merged full model under `outputs/multitask-lora-fast/merged_full_model`

Artifacts:
- Trained adapter/tokenizer under `outputs/multitask-lora-fast/`
- Metrics under the same folder

---

## Quick Local Inference (CLI)

Use a trained adapter directly from the command line:

```bash
python scripts/quick_infer.py --model-name google/mt5-small --adapter outputs/multitask-lora-fast --max-len 48
```

- Choose mode: summarize or QA
- Paste multi-line inputs; type `END` on a new line to finish

---

## API Server (FastAPI)

All-in-one server that loads:
- QA model `mt0` (from `QA_MODEL` env or default `savasy/mt0-large-Turkish-qa`). If a local PEFT adapter is found, it uses it; otherwise falls back to a HF pipeline.
- Multilingual/fallback summarization `mt5` (`google/mt5-small`)
- Local summarization adapters if present: `outputs/multitask-lora-fast`, `outputs/multitask-lora` (also checks `models/`)

Start with Python:

```bash
python mehmet-updates/api/enhanced_multi_model_api.py
```

Or via Uvicorn (module path):

```bash
uvicorn mehmet-updates.api.enhanced_multi_model_api:app --host 0.0.0.0 --port 8000 --reload
```

Environment variables (server):
- `QA_MODEL`: HF id or local adapter path for QA (default `savasy/mt0-large-Turkish-qa`)

Local model discovery:
- Looks under `models/` and `outputs/` for: `multitask-lora-fast`, `multitask-lora`, `mt0` candidates
- If not found, uses the HF id

Health and docs:
- Swagger UI: http://localhost:8000/docs
- Model status: http://localhost:8000/models/status
- Demo page: http://localhost:8000/ui

### Endpoints

1) Summarization

- `POST /summarize`

Request:
```json
{
  "text": "...Turkish news article...",
  "max_length": 128,
  "model": "multitask-lora-fast"  
}
```
- `model` options: `multitask-lora-fast`, `multitask-lora`, `mt5`

Response (example):
```json
{
  "model": "outputs/multitask-lora-fast",
  "input_text": "...",
  "summary": "Kısa başlık tarzı özet.",
  "input_length": 180,
  "summary_length": 12,
  "compression_ratio": 0.066,
  "timestamp": "2025-08-27T23:59:59"
}
```

2) Question Answering

- `POST /qa`

Request:
```json
{
  "question": "Hangi sektörlerden bahsediliyor?",
  "context": "...makale metni...",
  "model": "mt0"
}
```

Response (example):
```json
{
  "model": "savasy/mt0-large-Turkish-qa",
  "question": "Hangi sektörlerden bahsediliyor?",
  "context": "...",
  "answer": "fintech, e-ticaret ve oyun",
  "confidence": 0.73,
  "start_position": 211,
  "end_position": 245,
  "timestamp": "2025-08-27T23:59:59"
}
```

3) Multilingual Tasks (mT5)

- `POST /multilingual`

Request:
```json
{
  "text": "...",
  "task": "summarize",
  "max_length": 64
}
```

Response:
```json
{
  "model": "mT5-small",
  "task": "summarize",
  "generated_text": "..."
}
```

4) Utility

- `GET /models/status` — which models are loaded, where they came from
- `GET /ui` — minimal HTML to try summarization with a model selector

---

## Streamlit App

A simple UI that calls the FastAPI server (assumes it runs at `http://localhost:8000`).

Start:

```bash
streamlit run mehmet-updates/streamlit_app.py
```

Features:
- Paste or pick an example news article and generate a summary
- Ask follow-up questions about the summarized text
- Choose the summarization backend (`multitask-lora-fast`, `multitask-lora`, or `mt5`)
- Displays word counts and compression ratio

If your API runs elsewhere, change `API_BASE_URL` in `mehmet-updates/streamlit_app.py`.

---

## Tips & Troubleshooting

- CPU-only environments work but are slow. Prefer a CUDA GPU.
- If `bitsandbytes` fails to install (not strictly required here), continue without it.
- First-time model downloads can be large; ensure enough disk space.
- For Windows + PowerShell, remember to activate venv with:
  - `. .venv\Scripts\Activate.ps1`
- If you trained a new adapter under a different folder, copy/rename it to `outputs/multitask-lora-fast` or update the Streamlit/API config accordingly.

---

## Acknowledgments

- Datasets: `batubayk/TR-News`, `xtinge/turkish-extractive-summarization-dataset`, `ucsahin/TR-Extractive-QA-82K`
- Models: `google/mt5-small`, `savasy/mt0-large-Turkish-qa`
- Libraries: Hugging Face Transformers, PEFT, Datasets, FastAPI, Streamlit

---

## License

No license provided in this repository. If you plan to distribute models trained with these datasets, review their respective licenses and terms.

