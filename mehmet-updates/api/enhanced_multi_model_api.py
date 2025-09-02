#!/usr/bin/env python3
"""
🚀 Enhanced Multi-Model API Server
==================================

API serving:
- mt0 (QA - PEFT or HF pipeline)
- google/mt5-small (multilingual & summarization fallback)
- local fine-tuned summarization models: multitask-lora-fast, multitask-lora
"""

import os
import json
import torch
import logging
import uvicorn
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ML imports
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline
)
# Optional PEFT support for demo QA model
try:
    from peft import PeftConfig, PeftModel
    _HAS_PEFT = True
except Exception:
    PeftConfig = None
    PeftModel = None
    _HAS_PEFT = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMultiModelAPI:
    """Enhanced multi-model API with all requested models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.model_dir = self.base_dir / "models"

        logger.info(f"🖥️ Using device: {self.device}")

        # Model configurations
        # Keep only the models we want to expose in the API.
        # - mt0: QA model (could be PEFT adapter or HF id)
        # - mt5: Google mT5 for multilingual / summarization fallback
        # - multitask-lora-fast and multitask-lora: local fine-tuned summarization models
        self.models = {
            "mt0": os.getenv("QA_MODEL", "savasy/mt0-large-Turkish-qa"),
            "mt5": "google/mt5-small",
            "multitask-lora-fast": "multitask-lora-fast",
            "multitask-lora": "multitask-lora"
        }

        # Map keys to candidate local folder names (checked in order)
        # We prefer local outputs (e.g., outputs/multitask-lora-fast) if present.
        self.local_candidates = {
            "mt0": ["mt0_qa", "mt0", "qa"],
            "mt5": ["mt5_multilingual", "mt5_model"],
            "multitask-lora-fast": ["multitask-lora-fast", "multitask-lora-fast-model"],
            "multitask-lora": ["multitask-lora", "multitask-lora-model"]
        }

        # Loaded models cache
        self.loaded_models = {}

        # Load models on startup
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all models on startup"""
        logger.info("🚀 LOADING ALL MODELS FOR API")
        logger.info("=" * 50)
        
        # Load local LoRA summarization models (if available)
        for sname in ["multitask-lora-fast", "multitask-lora"]:
            try:
                logger.info(f"📝 Loading summarization model: {sname}...")
                src = self._select_model_source(sname)
                # attempt seq2seq load
                tok = AutoTokenizer.from_pretrained(src)
                model = AutoModelForSeq2SeqLM.from_pretrained(src)
                model.to(self.device)
                model.eval()

                self.loaded_models[sname] = {
                    "tokenizer": tok,
                    "model": model,
                    "type": "summarization",
                    "source": str(src)
                }
                logger.info(f"✅ Summarization model {sname} loaded from {src}")
            except Exception as e:
                logger.warning(f"Could not load summarization model {sname}: {e}")
        
        # Load QA model (mt0). Prefer PEFT seq2seq models; fall back to pipeline.
        try:
            logger.info("🎯 Loading QA model (mt0)...")
            qa_source = self._select_model_source("mt0")

            loaded = False
            if _HAS_PEFT:
                try:
                    cfg = PeftConfig.from_pretrained(qa_source)
                    base_model_id = cfg.base_model_name_or_path

                    qa_tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
                    qa_base = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
                    qa_model = PeftModel.from_pretrained(qa_base, qa_source)
                    qa_model.to(self.device)
                    qa_model.eval()

                    self.loaded_models["mt0"] = {
                        "tokenizer": qa_tokenizer,
                        "model": qa_model,
                        "type": "qa_peft",
                        "source": str(qa_source)
                    }
                    logger.info(f"✅ PEFT QA model loaded successfully from {qa_source}")
                    loaded = True
                except Exception as e:
                    logger.warning(f"PEFT QA load failed, will try pipeline fallback: {e}")

            if not loaded:
                try:
                    qa_pipeline = pipeline(
                        "question-answering",
                        model=qa_source,
                        tokenizer=qa_source,
                        device=0 if torch.cuda.is_available() else -1
                    )
                    self.loaded_models["mt0"] = {
                        "pipeline": qa_pipeline,
                        "type": "qa_pipeline",
                        "source": str(qa_source)
                    }
                    logger.info(f"✅ Pipeline QA model loaded successfully from {qa_source}")
                except Exception as e:
                    logger.error(f"❌ Failed to load QA model (both PEFT and pipeline failed): {e}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize QA model: {e}")
        
    # Load mT5 for multilingual and as a summarization fallback
        try:
            logger.info("🌍 Loading mT5 multilingual model...")
            mt5_source = self._select_model_source("mt5")
            mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_source)
            mt5_model = AutoModelForSeq2SeqLM.from_pretrained(mt5_source)
            mt5_model.to(self.device)
            mt5_model.eval()

            self.loaded_models["mt5"] = {
                "tokenizer": mt5_tokenizer,
                "model": mt5_model,
                "type": "multilingual",
                "source": str(mt5_source)
            }
            logger.info(f"✅ mT5 model loaded successfully from {mt5_source}")
        except Exception as e:
            logger.error(f"❌ Failed to load mT5: {e}")
            
        logger.info(f"🎉 Loaded {len(self.loaded_models)} models successfully!")

    def _select_model_source(self, key: str):
        """Return a local path if a candidate exists in models/, otherwise return HF id."""
        # check for local candidates first
        candidates = self.local_candidates.get(key, [])
        for name in candidates:
            # check models/ and outputs/ directories for local copies
            candidate_path = self.model_dir / name
            if candidate_path.exists():
                logger.info(f"Using local model for {key}: {candidate_path}")
                return str(candidate_path)

            outputs_path = self.base_dir / "outputs" / name
            if outputs_path.exists():
                logger.info(f"Using local outputs model for {key}: {outputs_path}")
                return str(outputs_path)

        # fallback to HF identifier
        return self.models.get(key)


# Helper to load summarization models on-demand (handles PEFT adapters)
def _load_summarization_on_demand(model_key: str) -> bool:
    """Attempt to load a summarization model on demand (supports PEFT adapters).

    Returns True if loaded successfully into multi_model_system.loaded_models.
    """
    try:
        if model_key not in ["multitask-lora-fast", "multitask-lora", "mt5"]:
            return False

        src = multi_model_system._select_model_source(model_key)

        # Try direct load first
        try:
            tok = AutoTokenizer.from_pretrained(src)
            model = AutoModelForSeq2SeqLM.from_pretrained(src)
            model.to(multi_model_system.device)
            model.eval()

            multi_model_system.loaded_models[model_key] = {
                "tokenizer": tok,
                "model": model,
                "type": "summarization" if model_key != "mt5" else "multilingual",
                "source": str(src)
            }
            logger.info(f"[on-demand] Loaded summarization model '{model_key}' from {src}")
            return True
        except Exception as direct_err:
            logger.info(f"[on-demand] Direct load failed for '{model_key}' from '{src}': {direct_err}")

        # Try PEFT adapter load
        if _HAS_PEFT:
            try:
                adapter_cfg_path = Path(str(src)) / "adapter_config.json"
                if adapter_cfg_path.exists():
                    cfg = PeftConfig.from_pretrained(str(src))
                    base_model_id = cfg.base_model_name_or_path

                    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
                    base = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
                    model = PeftModel.from_pretrained(base, str(src))
                    model.to(multi_model_system.device)
                    model.eval()

                    multi_model_system.loaded_models[model_key] = {
                        "tokenizer": tok,
                        "model": model,
                        "type": "summarization_peft",
                        "source": str(src),
                        "base_model": base_model_id
                    }
                    logger.info(f"[on-demand] Loaded PEFT adapter '{model_key}' from {src} on base {base_model_id}")
                    return True
            except Exception as peft_err:
                logger.warning(f"[on-demand] PEFT load failed for '{model_key}' from '{src}': {peft_err}")

        return False
    except Exception as e:
        logger.warning(f"[on-demand] Unexpected error while loading '{model_key}': {e}")
        return False

# Initialize the multi-model system
multi_model_system = EnhancedMultiModelAPI()

# FastAPI app
app = FastAPI(
    title="🚀 Multi-Model Turkish NLP API",
    description="API with mt0 (QA), mT5 and local multitask-lora summarization models",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class SummarizationRequest(BaseModel):
    text: str
    max_length: int = 128
    model: str = "multitask-lora-fast"  # options: multitask-lora-fast, multitask-lora, mt5

class QuestionAnsweringRequest(BaseModel):
    question: str
    context: str
    model: str = "mt0"

class MultilingualRequest(BaseModel):
    text: str
    task: str = "summarize"  # summarize, translate, etc.
    max_length: int = 64

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🚀 Multi-Model Turkish NLP API",
        "version": "2.0.0",
        "models": {
            "mt0": multi_model_system.models.get("mt0"),
            "multitask-lora-fast": "local fine-tuned summarization (if present)",
            "multitask-lora": "local fine-tuned summarization (if present)",
            "mT5": multi_model_system.models.get("mt5")
        },
        "endpoints": {
            "/summarize": "Text summarization (choose model via model parameter)",
            "/multilingual": "Multilingual tasks with mT5",
            "/qa": "Question answering with mt0",
            "/models/status": "Check model loading status",
            "/ui": "Simple web UI with model select box"
        },
        "docs": "/docs"
    }

@app.get("/models/status")
async def get_models_status():
    """Get the status of all loaded models"""
    status = {}
    for model_name in ["mt0", "multitask-lora-fast", "multitask-lora", "mt5"]:
        if model_name in multi_model_system.loaded_models:
            lm = multi_model_system.loaded_models[model_name]
            status[model_name] = {
                "loaded": True,
                "type": lm.get("type"),
                "model_path": lm.get("source") or multi_model_system.models.get(model_name),
                "base_model": lm.get("base_model")
            }
        else:
            status[model_name] = {
                "loaded": False,
                "error": "Failed to load or not present locally"
            }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "device": str(multi_model_system.device),
        "models": status,
        "total_loaded": len(multi_model_system.loaded_models)
    }

@app.post("/summarize")
async def summarize_text(request: SummarizationRequest):
    """Summarize text choosing between local multitask models and mT5 fallback."""
    try:
        model_choice = request.model

        if model_choice not in ["multitask-lora-fast", "multitask-lora", "mt5"]:
            raise HTTPException(status_code=400, detail="Invalid summarization model")

        if model_choice not in multi_model_system.loaded_models:
            # Try to load the model on demand (handles local PEFT adapters)
            if not _load_summarization_on_demand(model_choice):
                raise HTTPException(status_code=503, detail=f"Model {model_choice} not loaded")

        model_info = multi_model_system.loaded_models[model_choice]
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]

        # Prepare input (for mt5 prefix the task)
        input_text = request.text
        if model_choice == "mt5":
            input_text = f"summarize: {request.text}"

        # Tokenize input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        inputs = {k: v.to(multi_model_system.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                num_beams=4 if model_choice != "mt5" else 3,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "model": model_info.get("source") or model_choice,
            "input_text": request.text,
            "summary": summary,
            "input_length": len(request.text.split()),
            "summary_length": len(summary.split()),
            "compression_ratio": (len(summary.split()) / len(request.text.split())) if len(request.text.split())>0 else None,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/summarize/mt5")
async def summarize_text_mt5(request: SummarizationRequest):
    """Summarize text using mT5 model"""
    try:
        if "mt5" not in multi_model_system.loaded_models:
            raise HTTPException(status_code=503, detail="mT5 model not loaded")
        
        model_info = multi_model_system.loaded_models["mt5"]
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        
        # Add task prefix for mT5
        input_text = f"summarize: {request.text}"
        
        # Tokenize input
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        inputs = {k: v.to(multi_model_system.device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "model": "mT5-small",
            "input_text": request.text,
            "summary": summary,
            "input_length": len(request.text.split()),
            "summary_length": len(summary.split()),
            "compression_ratio": len(summary.split()) / len(request.text.split()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"mT5 summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"mT5 summarization failed: {str(e)}")

@app.post("/qa")
async def answer_question(request: QuestionAnsweringRequest):
        """Answer questions using mt0 (PEFT or pipeline)."""
        try:
                model_choice = request.model
                if model_choice != "mt0":
                        raise HTTPException(status_code=400, detail="Only mt0 QA model is supported")

                if "mt0" not in multi_model_system.loaded_models:
                        raise HTTPException(status_code=503, detail="QA model not loaded")

                model_info = multi_model_system.loaded_models["mt0"]
                mtype = model_info.get("type")

                if mtype == "qa_peft":
                        tokenizer = model_info["tokenizer"]
                        model = model_info["model"]

                        prompt = f"Soru: {request.question}\nContext: {request.context}\nCevap:"
                        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                        inputs = {k: v.to(multi_model_system.device) for k, v in inputs.items()}

                        with torch.no_grad():
                                out_ids = model.generate(**inputs, max_new_tokens=128, num_beams=4)
                        answer = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

                        return {
                                "model": model_info.get("source"),
                                "question": request.question,
                                "context": request.context,
                                "answer": answer,
                                "confidence": None,
                                "start_position": None,
                                "end_position": None,
                                "context_length": len(request.context.split()),
                                "question_length": len(request.question.split()),
                                "timestamp": datetime.now().isoformat()
                        }

                elif mtype == "qa_pipeline":
                        qa_pipeline = model_info["pipeline"]
                        result = qa_pipeline(question=request.question, context=request.context)

                        return {
                                "model": model_info.get("source"),
                                "question": request.question,
                                "context": request.context,
                                "answer": result.get("answer"),
                                "confidence": result.get("score"),
                                "start_position": result.get("start"),
                                "end_position": result.get("end"),
                                "context_length": len(request.context.split()),
                                "question_length": len(request.question.split()),
                                "timestamp": datetime.now().isoformat()
                        }
                else:
                        raise HTTPException(status_code=500, detail=f"Unsupported QA model type: {mtype}")

        except HTTPException:
                raise
        except Exception as e:
                logger.error(f"QA error: {e}")
                raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


@app.get('/ui', response_class=HTMLResponse)
async def simple_ui():
        """Return a tiny HTML page with a select box for summarization model choice."""
        html = '''
        <!doctype html>
        <html>
            <head>
                <meta charset="utf-8" />
                <title>Summarization UI</title>
            </head>
            <body>
                <h3>Summarization demo</h3>
                <label for="model">Choose model:</label>
                <select id="model">
                    <option value="multitask-lora-fast">multitask-lora-fast</option>
                    <option value="multitask-lora">multitask-lora</option>
                    <option value="mt5">google/mt5-small</option>
                </select>
                <br/><br/>
                <textarea id="input" rows="8" cols="80">Türkçe özetlenecek metni buraya yapıştırın.</textarea>
                <br/>
                <button onclick="summarize()">Summarize</button>
                <h4>Output</h4>
                <pre id="output"></pre>

                <script>
                async function summarize(){
                    const model = document.getElementById('model').value;
                    const text = document.getElementById('input').value;
                    const resp = await fetch('/summarize', {
                        method: 'POST',
                        headers: {'Content-Type':'application/json'},
                        body: JSON.stringify({text, model})
                    });
                    const j = await resp.json();
                    document.getElementById('output').textContent = JSON.stringify(j, null, 2);
                }
                </script>
            </body>
        </html>
        '''
        return HTMLResponse(content=html)

@app.post("/multilingual")
async def multilingual_task(request: MultilingualRequest):
    """Perform multilingual tasks using mT5 model"""
    try:
        if "mt5" not in multi_model_system.loaded_models:
            raise HTTPException(status_code=503, detail="mT5 model not loaded")
        
        model_info = multi_model_system.loaded_models["mt5"]
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        
        # Add task prefix
        input_text = f"{request.task}: {request.text}"
        
        # Tokenize input
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        inputs = {k: v.to(multi_model_system.device) for k, v in inputs.items()}
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                num_beams=3,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "model": "mT5-small",
            "task": request.task,
            "input_text": request.text,
            "generated_text": generated_text,
            "input_length": len(request.text.split()),
            "output_length": len(generated_text.split()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Multilingual task error: {e}")
        raise HTTPException(status_code=500, detail=f"Multilingual task failed: {str(e)}")

@app.get("/demo/examples")
async def get_demo_examples():
    """Get demo examples for testing"""
    return {
        "summarization_examples": [
            {
                "text": "Türkiye'de teknoloji sektörü son yıllarda hızla büyümekte ve önemli gelişmeler yaşanmaktadır. Yapay zeka, fintech, e-ticaret ve oyun geliştirme alanlarında birçok yeni girişim kurulmaktadır. İstanbul, Ankara ve İzmir'deki teknoloji merkezleri bu girişimlere ev sahipliği yapmakta, yatırım fonları da bu sektöre büyük ilgi göstermektedir.",
                "expected_summary": "Türkiye teknoloji sektöründe hızla büyüyor ve yatırım ilgisi artıyor."
            }
        ],
        "qa_examples": [
            {
                "context": "Türkiye Cumhuriyeti, 29 Ekim 1923'te Mustafa Kemal Atatürk önderliğinde kurulmuştur. Ankara'nın başkent ilan edilmesi, yeni cumhuriyetin modernleşme yolundaki ilk adımlarından biridir.",
                "question": "Türkiye Cumhuriyeti hangi tarihte kurulmuştur?",
                "expected_answer": "29 Ekim 1923"
            }
        ]
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Enhanced Multi-Model API Server")
    logger.info("=" * 60)
    logger.info("🌐 API Documentation: http://localhost:8000/docs")
    logger.info("🔍 Model Status: http://localhost:8000/models/status")
    logger.info("🎯 Demo Examples: http://localhost:8000/demo/examples")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,
        log_level="info"
    )
