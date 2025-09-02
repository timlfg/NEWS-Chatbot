# scripts/quick_infer.py
# -*- coding: utf-8 -*-

import os
import sys
import re
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ------------ Argümanlar -------------
def get_args():
    p = argparse.ArgumentParser(description="TR-NEWS quick inference (Özet + QA)")
    p.add_argument("--model-name", type=str, default=os.environ.get("MODEL_NAME", "google/mt5-small"),
                   help="Base model adı (örn: google/mt5-small)")
    p.add_argument("--adapter", type=str, default=os.environ.get("ADAPTER_DIR", "outputs/multitask-lora"),
                   help="Eğitilmiş LoRA klasörü")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                   help="cuda/cpu/auto")
    p.add_argument("--max-len", type=int, default=48, help="Varsayılan çıkış token uzunluğu")
    return p.parse_args()

# ------------ Yardımcılar -------------
def pick_device(choice: str) -> torch.device:
    if choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_multiline(prompt: str) -> str:
    print(prompt)
    print("Metni bitirmek için ayrı satıra sadece END yazın ve Enter’a basın.")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()

def cleanup(text: str) -> str:
    # Modelin istemdeki belirteçleri tekrar etmesini temizle
    t = text.strip()
    t = re.sub(r"\b(answer|cevap|summary|özet)\s*:\s*", "", t, flags=re.I)
    # çoklu tekrarları kırp
    t = re.sub(r"(.)\1{3,}", r"\1\1", t)
    # başa/sona tırnaklar
    t = t.strip("„“”\"'` ")
    return t

# ------------ Modeli Yükle -------------
def load_model_and_tokenizer(model_name: str, adapter_dir: str, device: torch.device):
    # dtype: GPU'da fp16, CPU'da fp32
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Model yükleniyor: {model_name}")
    base = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
    print(f"LoRA yükleniyor: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)
    # LoRA'yı birleştir (deployment için ideale yakın)
    model = model.merge_and_unload()
    model.to(device).eval()

    tok = AutoTokenizer.from_pretrained(model_name)
    return model, tok

# ------------ Üretim -------------
def generate(model, tok, device, inp: str, max_len: int):
    enc = tok(inp, return_tensors="pt", truncation=True, max_length=768)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_len,
            min_new_tokens=8,
            do_sample=True,          # kopyalamayı azaltmak için örnekleme
            top_p=0.9,
            temperature=0.8,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
        )
    return cleanup(tok.decode(out[0], skip_special_tokens=True))

# ------------ Modlar -------------
def summarize(model, tok, device, default_max_len=32):
    text = read_multiline("\n[Özet Modu] Haber metnini yapıştır:")
    if not text:
        print("Boş metin girdin. Çıkılıyor.")
        return
    # Eğitim formatıyla uyumlu ve stile yönlendiren istem
    prompt = (
        f"summarize: {text}\n"
        f"Talimat: Haberi tarafsızca, tek cümlede, başlık tarzında özetle.\n"
        f"summary:"
    )
    print("\n### Özet ###")
    print(generate(model, tok, device, prompt, max_len=default_max_len))

def qa(model, tok, device, default_max_len=32):
    question = input("\n[Soru-Cevap Modu] Soruyu yazın: ").strip()
    if not question:
        print("Boş soru girdin. Çıkılıyor.")
        return
    ctx = read_multiline("\nBağlam/paragrafı yapıştırın:")
    if not ctx:
        print("Boş bağlam girdin. Çıkılıyor.")
        return
    # Eğitim formatındaki anahtarlar: answer:/context:
    prompt = (
        f"answer: {question} Tek cümlede genel başlık gibi yanıt ver. "
        f"context: {ctx}\nanswer:"
    )
    print("\n### Cevap ###")
    print(generate(model, tok, device, prompt, max_len=default_max_len))

# ------------ Ana -------------
def main():
    args = get_args()
    device = pick_device(args.device)

    # Klasör kontrolü
    if not os.path.isdir(args.adapter):
        print(f"HATA: LoRA klasörü bulunamadı: {args.adapter}")
        sys.exit(1)

    model, tok = load_model_and_tokenizer(args.model_name, args.adapter, device)

    while True:
        print("\nMod seçin: 1) Özet  2) Soru-Cevap  3) Çıkış")
        choice = input("Seçiminiz (1/2/3): ").strip()
        if choice == "1":
            summarize(model, tok, device, default_max_len=args.max_len)
        elif choice == "2":
            qa(model, tok, device, default_max_len=args.max_len)
        elif choice == "3" or choice.lower() in {"q", "quit", "exit"}:
            print("Görüşürüz!")
            break
        else:
            print("Geçersiz seçim. 1, 2 ya da 3 girin.")

if __name__ == "__main__":
    main()
