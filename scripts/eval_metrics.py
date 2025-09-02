# scripts/train_multitask_qlora.py
# Türkçe Haber Özet + QA (çok görevli) — LoRA ile hızlı eğitim
# Güvenli decode, metin tabanlı metrikler, ayrıntılı logging

import os, inspect, json, math
import numpy as np
import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None
    _HAS_PEFT = False

# ============= Konfig =============
MODEL_NAME = os.environ.get("BASE_MODEL", "google/mt5-small")
DATA_DIR   = os.environ.get("DATA_DIR", "data2/processed/multitask_text2text")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/multitask-lora-fast")

MAX_SOURCE_LEN = int(os.environ.get("MAX_SOURCE_LEN", 384))
MAX_TARGET_LEN = int(os.environ.get("MAX_TARGET_LEN", 64))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 4))
LR         = float(os.environ.get("LR", 5e-4))
EPOCHS     = float(os.environ.get("EPOCHS", 1.5))
SEED       = int(os.environ.get("SEED", 42))

# Küçük subset — hızlı deneme
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", 3000))
MAX_EVAL_SAMPLES  = int(os.environ.get("MAX_EVAL_SAMPLES", 600))
MAX_TEST_SAMPLES  = int(os.environ.get("MAX_TEST_SAMPLES", 600))

# Sık değerlendirme/log
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", 100))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", 200))
LOG_STEPS  = int(os.environ.get("LOG_STEPS", 25))

set_seed(SEED)

# ============= Cihaz & dtype =============
has_cuda = torch.cuda.is_available()
cap_major = torch.cuda.get_device_capability(0)[0] if has_cuda else 0
use_bf16 = has_cuda and cap_major >= 8
use_fp16 = has_cuda and not use_bf16
dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

# ============= LoRA =============
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q","k","v","o","wi","wo"],
)

# ============= Model & Tokenizer =============
print(f"Loading model: {MODEL_NAME}")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

try:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
except Exception as e:
    print("[WARN] fallback to float32 due to:", e)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============= Dataset & Subset =============
assert os.path.exists(DATA_DIR), f"Dataset not found: {DATA_DIR} (run prepare_data.py)"
ds = load_from_disk(DATA_DIR)

def subset(d, n):
    return d.select(range(min(len(d), n))) if n and len(d) > n else d

train_raw = subset(ds["train"], MAX_TRAIN_SAMPLES)
val_raw   = subset(ds["validation"], MAX_EVAL_SAMPLES)
test_raw  = subset(ds["test"], MAX_TEST_SAMPLES)

print(f"Loaded dataset: train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}")

# ============= Tokenization =============
def preprocess(batch):
    X = tok(batch["source"], max_length=MAX_SOURCE_LEN, truncation=True)
    Y = tok(text_target=batch["target"], max_length=MAX_TARGET_LEN, truncation=True)
    X["labels"] = Y["input_ids"]
    return X

remove_cols = list(train_raw.features)
train_tok = train_raw.map(preprocess, batched=True, remove_columns=remove_cols, desc="Tokenize train")
val_tok   = val_raw.map(preprocess, batched=True, remove_columns=remove_cols, desc="Tokenize val")
test_tok  = test_raw.map(preprocess, batched=True, remove_columns=remove_cols, desc="Tokenize test")

collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model, label_pad_token_id=-100, padding=True)

# ============= Güvenli decode yardımcıları =============
def _to_numpy(a):
    if isinstance(a, dict) and "sequences" in a:
        a = a["sequences"]
    if isinstance(a, (list, tuple)):
        a = a[0]
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return a

def _safe_ids(arr, pad_id):
    arr = _to_numpy(arr)
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.signedinteger):
        arr = arr.astype(np.int64, copy=False)
    arr[arr < 0] = pad_id  # -100 vb -> pad
    return arr

def _safe_decode(arr):
    pad_id = tok.pad_token_id or tok.eos_token_id or 0
    ids = _safe_ids(arr, pad_id)
    return tok.batch_decode(ids, skip_special_tokens=True)

def _norm(s):
    return " ".join(str(s).strip().split()).lower()

# ============= Basit metin tabanlı metrikler =============
def token_f1_text_list(preds, refs):
    """Her örnek için token-set F1; sonra ortalama."""
    scores = []
    for p, r in zip(preds, refs):
        p = _norm(p); r = _norm(r)
        ps, rs = set(p.split()), set(r.split())
        if len(ps) == 0 or len(rs) == 0:
            scores.append(0.0); continue
        inter = len(ps & rs)
        if inter == 0:
            scores.append(0.0); continue
        prec = inter / len(ps)
        rec  = inter / len(rs)
        scores.append(2*prec*rec/(prec+rec) if (prec+rec) else 0.0)
    return float(np.mean(scores)) if scores else 0.0

def _lcs_len(a_tokens, b_tokens):
    # O(n*m) LCS; kısa hedefler için yeterli
    n, m = len(a_tokens), len(b_tokens)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        ai = a_tokens[i]
        for j in range(m):
            dp[i+1][j+1] = dp[i][j] + 1 if ai == b_tokens[j] else max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def rouge_l_f1_text_list(preds, refs):
    scores = []
    for p, r in zip(preds, refs):
        p = _norm(p); r = _norm(r)
        pt, rt = p.split(), r.split()
        if not pt or not rt:
            scores.append(0.0); continue
        lcs = _lcs_len(pt, rt)
        prec = lcs / len(pt)
        rec  = lcs / len(rt)
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0

# ============= compute_metrics (metin üstünden) =============
def compute_metrics(eval_pred):
    pred_ids, label_ids = eval_pred
    lab_ids = _safe_ids(label_ids, tok.pad_token_id or tok.eos_token_id or 0)

    preds_text  = _safe_decode(pred_ids)
    labels_text = tok.batch_decode(lab_ids, skip_special_tokens=True)

    em   = float(np.mean([1.0 if _norm(p)==_norm(t) else 0.0 for p,t in zip(preds_text, labels_text)]))
    t_f1 = token_f1_text_list(preds_text, labels_text)
    rl_f1= rouge_l_f1_text_list(preds_text, labels_text)

    return {"em": em, "token_f1": t_f1, "rougeL_f1": rl_f1}

# ============= Training Args =============
def make_args(**kw):
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    return Seq2SeqTrainingArguments(**{k:v for k,v in kw.items() if k in sig.parameters})

args = make_args(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    bf16=use_bf16, fp16=use_fp16,
    logging_steps=LOG_STEPS,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_total_limit=2,
    report_to="none",
    # Eski-yeni isimler birlikte (uyumluluk)
    evaluation_strategy="steps",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    generation_num_beams=4,
    remove_unused_columns=False,
    dataloader_pin_memory=has_cuda,
    eval_accumulation_steps=1,
)

# ============= Callback: train subset eval + CSV log =============
class TrainEvalLogger(TrainerCallback):
    def __init__(self, trainer, train_subset, out_dir):
        self.t = trainer
        self.sub = train_subset
        self.path = os.path.join(out_dir, "metrics.csv")
        os.makedirs(out_dir, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("step,epoch,train_loss,eval_loss,train_em,eval_em,train_token_f1,eval_token_f1\n")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 1) Validation metrikleri
        ev = metrics or {}
        eval_loss = ev.get("eval_loss")
        eval_em   = ev.get("eval_em")
        eval_f1   = ev.get("eval_token_f1")

        # 2) Train subset üzerinde predict -> metin tabanlı metrikler
        pred = self.t.predict(self.sub)
        train_loss = pred.metrics.get("test_loss")
        pred_txt = _safe_decode(pred.predictions)
        gold_txt = tok.batch_decode(_safe_ids(pred.label_ids, tok.pad_token_id or tok.eos_token_id or 0), skip_special_tokens=True)
        train_em  = float(np.mean([1.0 if _norm(p)==_norm(t) else 0.0 for p,t in zip(pred_txt, gold_txt)]))
        train_f1  = token_f1_text_list(pred_txt, gold_txt)

        # CSV
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{state.global_step},{state.epoch or 0},{train_loss},{eval_loss},{train_em},{eval_em},{train_f1},{eval_f1}\n")

        # Konsol
        if (train_loss is not None) and (eval_loss is not None):
            print(f"[STEP {state.global_step}] "
                  f"train_loss={train_loss:.4f} | val_loss={eval_loss:.4f} | "
                  f"train_acc(EM)={train_em:.4f} | val_acc(EM)={eval_em:.4f}")
        else:
            print(f"[STEP {state.global_step}] (metrics pending write)")

# ============= Trainer =============
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=collator,
    tokenizer=tok,
    compute_metrics=compute_metrics,
)

# Küçük train subset (hızlı ölçüm)
train_subset_for_eval = train_tok.select(range(min(600, len(train_tok))))
trainer.add_callback(TrainEvalLogger(trainer, train_subset_for_eval, OUTPUT_DIR))

# ============= Train =============
print("Starting training...")
trainer.train()
print("Training done!")

# ============= Val tahminlerini kaydet =============
val_preds = trainer.predict(val_tok)
pred_txt = _safe_decode(val_preds.predictions)
gold_txt = tok.batch_decode(_safe_ids(val_preds.label_ids, tok.pad_token_id or tok.eos_token_id or 0), skip_special_tokens=True)

def _token_f1_ex(p, t):
    p, t = _norm(p), _norm(t)
    ps, ts = set(p.split()), set(t.split())
    if not ps or not ts: return 0.0
    inter = len(ps & ts)
    if inter == 0: return 0.0
    prec = inter / len(ps); rec = inter / len(ts)
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

eval_pred_path = os.path.join(OUTPUT_DIR, "eval_predictions.jsonl")
with open(eval_pred_path, "w", encoding="utf-8") as f:
    for s, y, p in zip(ds["validation"]["source"][:len(pred_txt)], ds["validation"]["target"][:len(pred_txt)], pred_txt):
        em = int(_norm(p) == _norm(y))
        f1 = float(_token_f1_ex(p, y))
        f.write(json.dumps({"source": s, "target": y, "prediction": p, "em": em, "token_f1": f1}, ensure_ascii=False) + "\n")
print(f"[INFO] Saved validation predictions to: {eval_pred_path}")

# ============= Kaydet =============
trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print("Saved to:", OUTPUT_DIR)

# Opsiyonel: LoRA merge
if os.environ.get("MERGE_AND_SAVE", "0").lower() in ("1","true","yes"):
    if not _HAS_PEFT:
        print("PEFT not installed: cannot merge and save full model.")
    else:
        try:
            base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            merged = PeftModel.from_pretrained(base, OUTPUT_DIR)
            out_merge = os.path.join(OUTPUT_DIR, "merged_full_model")
            merged.save_pretrained(out_merge)
            tok.save_pretrained(out_merge)
            print(f"Merged full model saved to: {out_merge}")
        except Exception as e:
            print("Failed to export merged model:", e)

# Test (varsa)
if len(test_tok) > 0:
    test_metrics = trainer.evaluate(eval_dataset=test_tok, metric_key_prefix="test")
    print("[TEST] metrics:", test_metrics)
    with open(os.path.join(OUTPUT_DIR, "test_metrics.txt"), "w", encoding="utf-8") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")
    print("Test metrics written.")
