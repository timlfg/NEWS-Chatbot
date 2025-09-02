# scripts/train_multitask_qlora.py
# Türkçe Haber Özet + QA (çok görevli) — LoRA ile eğitim
import os
import inspect
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    PeftModel = None
    _HAS_PEFT = False
from eval_metrics import rouge_l_f1, token_f1

# =======================
# Konfigürasyon
# =======================
MODEL_NAME      = os.environ.get("BASE_MODEL", "google/mt5-small")
DATA_DIR        = os.environ.get("DATA_DIR", "data1/processed/multitask_text2text")  # sen data1 kullandın
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", "outputs/multitask-lora")
MAX_SOURCE_LEN  = int(os.environ.get("MAX_SOURCE_LEN", 512))
MAX_TARGET_LEN  = int(os.environ.get("MAX_TARGET_LEN", 128))
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", 4))
GRAD_ACCUM      = int(os.environ.get("GRAD_ACCUM", 4))
LR              = float(os.environ.get("LR", 5e-4))
EPOCHS          = float(os.environ.get("EPOCHS", 3))
SEED            = int(os.environ.get("SEED", 42))

set_seed(SEED)

# =======================
# Cihaz / dtype mantığı
# =======================
has_cuda = torch.cuda.is_available()
cap_major = torch.cuda.get_device_capability(0)[0] if has_cuda else 0
use_bf16 = has_cuda and cap_major >= 8   # Ampere+ ise bf16 güvenli
use_fp16 = has_cuda and not use_bf16     # Eski GPU'larda fp16; CPU'da ikisi de False

dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

# =======================
# LoRA yapılandırması
# =======================
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    # T5/mT5: q,k,v,o ve FFN (wi,wo) hedeflemek genel olarak daha iyi sonuç verir
    target_modules=["q", "k", "v", "o", "wi", "wo"],
)

# =======================
# Tokenizer / Model
# =======================
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# T5/mT5'te pad token genelde yok; eos'u pad olarak ayarla (collator bunu düzgün yönetir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Modeli yükle (Windows'ta QLoRA yok; klasik LoRA ile)
# use_safetensors=True varsa güvenli ve hızlı yüklenir; yoksa otomatik .bin'e düşer
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, use_safetensors=True)
except Exception as e:
    print("[WARN] safetensors bulunamadı, .bin ile yükleniyor ->", e)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)

# LoRA adaptörünü uygula
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =======================
# Dataset
# =======================
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset klasörü bulunamadı: {DATA_DIR}. Lütfen önce prepare_data.py çalıştır.")

ds = load_from_disk(DATA_DIR)

if len(ds["train"]) == 0:
    raise ValueError("Train split boş! prepare_data.py içindeki veri kaynaklarını kontrol et.")

has_test = "test" in ds and len(ds["test"]) > 0
print(f"Loaded dataset: train={len(ds['train'])}, val={len(ds['validation'])}" + (f", test={len(ds['test'])}" if has_test else ""))

# =======================
# Tokenization
# =======================
def preprocess(batch):
    # Girdi
    model_inputs = tokenizer(
        batch["source"],
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        # padding yapılmıyor; dynamic padding'i collator halleder
    )
    # Hedef
    labels = tokenizer(
        text_target=batch["target"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Orijinal kolonlar
remove_cols = list(ds["train"].features)
print("Tokenizing dataset...")
tokenized = ds.map(preprocess, batched=True, remove_columns=remove_cols, desc="Tokenizing")

sizes_msg = f"Tokenization complete. Train size: {len(tokenized['train'])}, Validation size: {len(tokenized['validation'])}"
if has_test:
    sizes_msg += f", Test size: {len(tokenized['test'])}"
print(sizes_msg)

# =======================
# Data collator
# =======================
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,   # loss'tan maskelensin
    padding=True               # dynamic padding
)

# =======================
# Training Arguments (Seq2Seq)
# Eski/uyumsuz sürümlerde bilinmeyen argümanları filtrelemek için ufak yardımcı
# =======================
def make_seq2seq_args(**kwargs) -> Seq2SeqTrainingArguments:
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    missing = [k for k in kwargs if k not in sig.parameters]
    if missing:
        print("[WARN] Desteklenmeyen Seq2SeqTrainingArguments parametreleri atlandı:", missing)
    return Seq2SeqTrainingArguments(**filtered)

args = make_seq2seq_args(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    bf16=use_bf16,
    fp16=use_fp16,
    logging_steps=50,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
    save_total_limit=2,

    # Değerlendirme/saklama stratejisi (desteklenmezse helper atar)
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,

    # Seq2Seq spesifik ayarlar
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    generation_num_beams=4,

    # Stabilite
    gradient_checkpointing=False,   # VRAM tasarrufu
    dataloader_pin_memory=has_cuda,
    remove_unused_columns=False,
)

# =======================
# Trainer
# =======================
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=lambda eval_pred: {
        "rougeL_f1": rouge_l_f1(eval_pred.predictions, eval_pred.label_ids),
        "token_f1": token_f1(eval_pred.predictions, eval_pred.label_ids)
    },
)

# =======================
# Train
# =======================
print("Starting training...")
trainer.train()
print("Training completed successfully!")

# =======================
# Save
# =======================
print("Saving model and tokenizer...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to: {OUTPUT_DIR}")

# Optionally export a merged full model (base + LoRA) for easy inference.
# Set env MERGE_AND_SAVE=1 to enable. This will create OUTPUT_DIR/merged_full_model
if os.environ.get("MERGE_AND_SAVE", "0").lower() in ("1", "true", "yes"):
    if not _HAS_PEFT:
        print("PEFT not installed: cannot merge and save full model.")
    else:
        try:
            print("Exporting merged full model (base + LoRA) to OUTPUT_DIR/merged_full_model ...")
            from transformers import AutoModelForSeq2SeqLM
            base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            merged = PeftModel.from_pretrained(base, OUTPUT_DIR)
            out_merge = os.path.join(OUTPUT_DIR, "merged_full_model")
            merged.save_pretrained(out_merge)
            tokenizer.save_pretrained(out_merge)
            print(f"Merged full model saved to: {out_merge}")
        except Exception as e:
            print("Failed to export merged model:", e)

# =======================
# Test değerlendirmesi (varsa)
# =======================
if has_test:
    print("Evaluating on TEST split...")
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized["test"],
        metric_key_prefix="test"
    )
    print("[TEST] metrics:", test_metrics)

    # Write metrics to file
    metrics_path = os.path.join(OUTPUT_DIR, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in test_metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Test metrics written to {metrics_path}")

print("Training pipeline completed!")
