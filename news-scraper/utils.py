import re
import unicodedata
from langdetect import detect

def clean_text(text: str) -> str:
    """Haber metnini normalize edip gereksiz kısımları siler."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"http\S+", "", text)  # linkleri sil
    text = re.sub(r"\s+", " ", text)     # fazla boşluk sil
    return text.strip()

def is_turkish(text: str) -> bool:
    """Metnin Türkçe olup olmadığını kontrol eder."""
    try:
        return detect(text) == "tr"
    except:
        return False
