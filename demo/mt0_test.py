"""""
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PEFT_MODEL = "savasy/mt0-large-Turkish-qa"
MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 200

def load_model_and_tokenizer(peft_model_path: str):
    config = PeftConfig.from_pretrained(peft_model_path)
    base_model_id = config.base_model_name_or_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Base modeli yükle (bellek kullanımını azaltmaya çalışır)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    # PEFT ağırlıklarını sar ve modele uygula
    peft_model = PeftModel.from_pretrained(model, peft_model_path)
    peft_model.eval()

    # Cihaza taşı ve fp16'e çevir (CUDA varsa)
    peft_model.to(device)
    if device.type == "cuda":
        try:
            peft_model.half()
        except Exception:
            pass

    return peft_model, tokenizer, device

def generate_answer(model, tokenizer, prompt: str,
                    device: torch.device,
                    max_input_length=MAX_INPUT_LENGTH,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=5,
                    temperature=0.2,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=False):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        padding="longest"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.inference_mode():
        generated = model.generate(**gen_kwargs)

    outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [o.strip() for o in outputs]

def main():
    model, tokenizer, device = load_model_and_tokenizer(PEFT_MODEL)

    test_input = (
        "Mustafa adını babası Ali Rıza Efendi kendi dedesinin adı olduğundan dolayı vermiştir. "
        "Çünkü Ali Rıza Efendi'nin babasının adı olan Ahmed adı ağabeylerinden birisine verilmişti. "
        "Mustafa'ya neden Kemal isminin verildiğine yönelik ise çeşitli iddialar vardır. "
        "Afet İnan, bu ismi ona matematik öğretmeni Üsküplü Mustafa Efendi'nin Kemal adının anlamında "
        "olduğu gibi onun \"mükemmel ve olgun\" olduğunu göstermek için verdiğini söylemiştir. (source: wikipedia). "
        "Ali Rıza Efendi kimdir?"
    )

    outputs = generate_answer(
        model,
        tokenizer,
        test_input,
        device,
        num_beams=5,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=False,
        max_new_tokens=200
    )

    for i, out in enumerate(outputs, start=1):
        print(f"\n--- Output #{i} ---\n{out}\n")

if __name__ == "__main__":
    main()
"""

import os
import time
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Hız için daha agresif ayarlar (daha kısa, tek geçiş)
PEFT_MODEL = os.getenv("PEFT_MODEL", "savasy/mt0-large-Turkish-qa")
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "512"))   # kısaltılmış input -> daha hızlı
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "64"))        # küçük çıktı -> hızlı
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))                   # 1 en hızlı (beamsearch yok)
DO_SAMPLE = os.getenv("DO_SAMPLE", "false").lower() == "true"
USE_TORCH_COMPILE = os.getenv("TORCH_COMPILE", "false").lower() == "true"

def prepare_prompt(text: str) -> str:
    base = (
        "Aşağıdaki metne dayanarak kısa ve kesin bir cevap verin. "
        "Sadece metinde açıkça yer alan bilgiyi kullanın.\n\nMetin:\n"
    )
    return base + text.strip() + "\n\nCevap:"

def load_model_and_tokenizer(peft_model_path: str):
    cfg = PeftConfig.from_pretrained(peft_model_path)
    base_model_id = cfg.base_model_name_or_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Base modeli düşük bellek modu ile yükle
    base = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    # PEFT ağırlıklarını uygula
    model = PeftModel.from_pretrained(base, peft_model_path)
    model.eval()
    model.to(device)

    # GPU varsa half (fp16) ve cache kullanımını zorla
    if device.type == "cuda":
        try:
            model.half()
        except Exception:
            pass

    # Torch.compile ile hız denemesi (opsiyonel)
    if USE_TORCH_COMPILE:
        try:
            model = torch.compile(model)
        except Exception:
            pass

    return model, tokenizer, device

def generate_answer(model, tokenizer, prompt: str, device: torch.device):
    # tokenize (tek örnek, padding yapmadan daha hızlı)
    max_len = min(MAX_INPUT_LENGTH, getattr(tokenizer, "model_max_length", MAX_INPUT_LENGTH))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len, padding=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS if not DO_SAMPLE else 1,
        early_stopping=True,
        do_sample=DO_SAMPLE,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    t0 = time.time()
    # FP16 autocast sadece CUDA'da ek hız ve kararlılık sağlayabilir
    if device.type == "cuda":
        autocast = torch.cuda.amp.autocast
    else:
        # CPU için noop context manager
        class _noop:
            def __enter__(self): pass
            def __exit__(self, exc_type, exc, tb): pass
        autocast = _noop

    with torch.inference_mode():
        with autocast():
            out_ids = model.generate(**gen_kwargs)
    elapsed = time.time() - t0

    decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    if "Cevap:" in decoded:
        decoded = decoded.split("Cevap:", 1)[-1].strip()
    return decoded, elapsed

def main():
    model, tokenizer, device = load_model_and_tokenizer(PEFT_MODEL)

    # Test örneği (kendi metninizi buraya koyun)
    test_input = (
        "Anadolu Yakası'nda 29 Haziran Pazar günü elektrik kesintileri yaşanacak. Kadıköy'de Suadiye Mahallesi, Aydın Sokak, Emin Al Paşa Caddesi, Hazan, Dere ve Yeni Gelin sokaklara, Beymen Mağazası, Büyük Hanlı İnşaat, Öncü ve Kitapçı sokaklara, Kozyatağı Mahallesi, Çarmıklı Konutları, Kaya Sultan Sokak, Tıbbiye, Dr. Eyüp Aksoy caddelerine, Siyami Ersek Hastanesi, Askeri Yurt, Harem E-5 Avrasya Şantiyesi, Harem Liman içi, Burhan Felek Caddesi, GATA Hastanesi'ne. Kartal'da Esentepe Mahallesi, Taş Ocakları mevkii, Milangaz Caddesi, Güney yanyol Lukoil Petrol ve yanındaki iş yerlerine, ABB, Aysa Tekstil, Alçin Fabrikası ve civarına. Ataşehir'de Yeni Çamlıca Mahallesi, Yedpa Sanayi Sitesi ve civarına. Tuzla'da Aydıntepe Mahallesi, Tuzla Gemi İş Merkezi, Çukurova Vinç, Akyıldız, Bucak, Selin sokaklara, Sahilyolu Bulvarı, Tersaneler ve Güzin sokaklara, Özek İş Merkezi, Çeksan Tersanesi'ne. Sancaktepe'de Yenidoğan Mahallesi, Seyhan Cadde girişi, Necip Fazıl Caddesi, Şık Örme, Ufuk Caddesi, Ufuk ve İlba sokaklara, Barbaros Caddesi, Mezarlık arkasına, aynı gün 14.00-17.00 saatleri arasında Osmangazi Mahallesi, Mertler Caddesi, Şenlik, Asmalı, Şölen, Mesut, Açangül sokakları ve civarına. Pendik'te Ramazanoğlu Mahallesi, Fatih, Yıldırım, Emek sokaklara, Sanayi Caddesi, Aks Anahtar, Makkalıp, Bahar Polümer, CMS Makine, Sim Oto, Pikasan, Şişli Sanayi Sitesi, Ensar Caddesi, Yıldız Metal, Kavi Kablo, Şanlı Vana, Mutaş Cıvata, May Torna, Eral Elektronik, Ekstel, Nur Plastik, Şahin Etiket ve Yılmaz, Sultan Selim, Mutlu, Rahmetli sokaklara, aynı gün 06.00-09.00 saatleri arasında Batı Mahalle Burhan Toprak, Ankara, Selim Berzek caddelerine, Pendik Devlet Hastanesi, Somtaş Sitesi, Yargıcılar Sokak ve My Otel Asya, 08.00.11.00 saatleri arasında Batı Mahallesi, 23 Nisan, Ankara, Gazi Paşa, Namık Kemal, Ortanca ve Lokman Hekim caddelerine, Lale, Pazar ve Yasemin sokaklara. Maltepe'de 08.00-16.00 saatleri arasında Altayçeşme Mahallesi, Erel Otomotiv, Maltepe Devlet Hastanesi, Seri, Samanyolu, Hikmet,Yasemin, Toygun sokaklara ve civarına 29 Haziran günü elektrik verilemeyecek."
        "Hangi gün elektrik kesintisi yaşanacak?"
    )

    prompt = prepare_prompt(test_input)
    answer, elapsed = generate_answer(model, tokenizer, prompt, device)

    print(f"Süre: {elapsed:.2f}s")
    print(f"Cevap: {answer}")

if __name__ == "__main__":
    main()