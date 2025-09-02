# scripts/test_infer.py
# İnteraktif test: Özet + QA (mT5 + LoRA)
# Kullanım örnekleri:
#   - Sadece özet (interaktif yapıştırma):
#       python scripts/test_infer.py --model_dir outputs/multitask-lora-fast --task summarize --interactive
#   - Özet + QA (interaktif):
#       python scripts/test_infer.py --model_dir outputs/multitask-lora-fast --task both --interactive
#   - Dosyadan haber + komut satırından soru:
#       python scripts/test_infer.py --model_dir outputs/multitask-lora-fast --task qa --article_file haber.txt --question "Hangi şehirde oldu?"
#   çalıştırma komutu: python scripts\test_infer.py --model_dir outputs\multitask-lora-fast --task both --interactive --max_source_len 512 --max_new_tokens_summ 96 --max_new_tokens_qa 96


import os
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# ======================
# Yardımcılar
# ======================
def read_multiline(prompt: str, end_token: str = "###") -> str:
    """
    Konsola çok satırlı yapıştırma. Bitirmek için tek satıra end_token yaz.
    """
    print(prompt, end="")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == end_token:
            break
        lines.append(line)
    return "\n".join(lines)

def read_text_file(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as f:
        return f.read()

def pick_device_and_dtype():
    if torch.cuda.is_available():
        cap_major = torch.cuda.get_device_capability(0)[0]
        use_bf16 = cap_major >= 8
        device = torch.device("cuda")
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype

def load_tokenizer(base_model: str):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def folder_has_lora_adapter(model_dir: str) -> bool:
    # HF PEFT adaptörlerinin tipik imzaları
    return any(os.path.exists(os.path.join(model_dir, fname)) for fname in [
        "adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"
    ])

def load_model(model_dir: str, base_model: str, device: torch.device, dtype: torch.dtype):
    """
    model_dir: LoRA adaptörü klasörü (trainer.save_model ile kaydedilmiş)
               veya bir "birleştirilmiş tam model" klasörü olabilir.
    """
    if folder_has_lora_adapter(model_dir):
        # LoRA adaptörünü taban modele tak
        base = AutoModelForSeq2SeqLM.from_pretrained(base_model, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, model_dir, torch_dtype=dtype)
        print(f"[INFO] Loaded PEFT adapter on base model -> base='{base_model}', adapter='{model_dir}'")
    else:
        # Birleştirilmiş tam modeli doğrudan yükle
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=dtype)
        print(f"[INFO] Loaded full model from '{model_dir}'")
    model.to(device)
    model.eval()
    return model

def generate_one(model, tok, source_text: str, device, max_source_len: int, max_new_tokens: int, num_beams: int):
    inputs = tok(
        source_text,
        max_length=max_source_len,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            early_stopping=True
        )
    text = tok.batch_decode(out, skip_special_tokens=True)[0]
    return text.strip()

# ======================
# Argümanlar
# ======================
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Eğitim çıktısı klasörü (LoRA adaptörü ya da birleştirilmiş tam model).")
    p.add_argument("--base_model", default="google/mt5-small",
                   help="LoRA adaptörü kullanılıyorsa taban model adı (örn: google/mt5-small).")
    p.add_argument("--task", choices=["summarize", "qa", "both"], default="both",
                   help="Çalıştırılacak görev.")
    p.add_argument("--article", type=str, default=None, help="Haber metni (komut satırından).")
    p.add_argument("--article_file", type=str, default=None, help="Haber metni için dosya yolu.")
    p.add_argument("--question", type=str, default=None, help="QA için soru.")
    p.add_argument("--interactive", action="store_true",
                   help="Çok satırlı yapıştırma ve soru girişi (### ile bitir).")

    p.add_argument("--max_source_len", type=int, default=384, help="Kaynak max uzunluk.")
    p.add_argument("--max_new_tokens_summ", type=int, default=64, help="Özet üretim uzunluğu.")
    p.add_argument("--max_new_tokens_qa", type=int, default=64, help="QA üretim uzunluğu.")
    p.add_argument("--num_beams", type=int, default=4, help="Beam search beam sayısı.")
    return p

# ======================
# Ana
# ======================
def main():
    args = build_parser().parse_args()

    device, dtype = pick_device_and_dtype()
    print(f"[INFO] Device={device.type} | dtype={dtype}")

    # Tokenizer
    print(f"[INFO] Loading tokenizer from: {args.base_model}")
    tok = load_tokenizer(args.base_model)

    # Model
    model = load_model(args.model_dir, args.base_model, device, dtype)

    # === Girdi toplama ===
    article = None
    question = args.question

    if args.article is not None:
        article = args.article
    elif args.article_file is not None:
        article = read_text_file(args.article_file)
    elif args.interactive:
        article = read_multiline(
            "Haber metnini yapıştır. Bitirmek için tek satıra ### yaz ve Enter'a bas:\n"
        )
    else:
        raise SystemExit("Haber metni vermediniz. --article, --article_file veya --interactive kullanın.")

    if args.task in ("qa", "both"):
        if not question:
            # Soru yoksa interaktif iste
            question = input("Soru (QA için): ").strip()
        if not question:
            raise SystemExit("QA görevi için soru gerekli (boş olamaz).")

    # === Çalıştır ===
    if args.task in ("summarize", "both"):
        src = f"summarize: {article}"
        t0 = time.time()
        summary = generate_one(
            model, tok, src, device,
            max_source_len=args.max_source_len,
            max_new_tokens=args.max_new_tokens_summ,
            num_beams=args.num_beams
        )
        dt = time.time() - t0
        print("\n" + "="*12 + " ÖZET " + "="*12)
        print(summary)
        print(f"[info] süre: {dt:.2f}s")

    if args.task in ("qa", "both"):
        src = f"answer: {question} context: {article}"
        t0 = time.time()
        answer = generate_one(
            model, tok, src, device,
            max_source_len=args.max_source_len,
            max_new_tokens=args.max_new_tokens_qa,
            num_beams=args.num_beams
        )
        dt = time.time() - t0
        print("\n" + "="*12 + " CEVAP " + "="*12)
        print(answer)
        print(f"[info] süre: {dt:.2f}s")


if __name__ == "__main__":
    main()

