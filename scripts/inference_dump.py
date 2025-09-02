# scripts/inference_dump.py

import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_MODEL = os.environ.get("BASE_MODEL", "google/mt5-small")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs/multitask-lora")
DATA_DIR   = os.environ.get("DATA_DIR", "data1/processed/multitask_text2text")
SPLIT      = os.environ.get("SPLIT", "validation")
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Dataset
ds = load_from_disk(DATA_DIR)
if SPLIT not in ds:
    raise ValueError(f"Split bulunamadı: {SPLIT}")
dataset = ds[SPLIT]

# 2) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = None

# 3) Önce merged full model varsa onu yükle
merged_dir = os.path.join(OUTPUT_DIR, "merged_full_model")
if os.path.isdir(merged_dir):
    print(f"[INFO] Merged full model bulundu: {merged_dir}")
    model = AutoModelForSeq2SeqLM.from_pretrained(merged_dir).to(device)
else:
    # 4) PEFT adapter yolunu dene
    try:
        from peft import PeftModel  # sürüm uyumsuzsa burada patlar
        print(f"[INFO] Base model yükleniyor: {BASE_MODEL}")
        base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
        model = PeftModel.from_pretrained(base, OUTPUT_DIR).to(device)
        print("[INFO] LoRA adapter takıldı.")
        # İstersen tek parça istiyorsan:
        # model = model.merge_and_unload().to(device)
    except Exception as e:
        print("[WARN] PEFT ile yükleme olmadı:", e)
        # 5) OUTPUT_DIR tam model mi (önceden merge etmiş olabilirsin)
        try:
            print(f"[INFO] OUTPUT_DIR tam model olarak denenecek: {OUTPUT_DIR}")
            model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR).to(device)
        except Exception as e2:
            raise RuntimeError(
                f"Model yüklenemedi. Çözümler: "
                f"(a) sürümleri hizala, (b) eğitimde MERGE_AND_SAVE=1 ile merged_full_model çıkar. Detay: {e2}"
            )

model.eval()

# 6) Tahmin döngüsü
preds, refs = [], []
try:
    from tqdm.auto import tqdm
    iterator = tqdm(dataset, total=len(dataset))
except Exception:
    iterator = dataset

with torch.no_grad():
    for ex in iterator:
        source, target = ex["source"], ex["target"]
        inputs = tokenizer(
            source, return_tensors="pt", truncation=True, max_length=MAX_SOURCE_LEN
        ).to(device)
        output = model.generate(
            **inputs,
            max_length=MAX_TARGET_LEN,
            num_beams=4,
            no_repeat_ngram_size=3
        )
        pred = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        preds.append(pred)
        refs.append(target.strip())

with open("preds.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(preds))
with open("refs.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(refs))

print(f"[OK] {len(preds)} örnek kaydedildi → preds.txt & refs.txt")
