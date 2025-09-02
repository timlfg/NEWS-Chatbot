# prepare_data.py
import os, json, random
from typing import List, Dict, Optional
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

# =======================
# Konfig (ENV ile override edilebilir)
# =======================
OUT_DIR   = os.environ.get("OUT_DIR", "data2/processed/multitask_text2text")

TR_NEWS_NAME = os.environ.get("TR_NEWS_NAME", "batubayk/TR-News")

SUMM_DATASET = "xtinge/turkish-extractive-summarization-dataset"
SUMM_CONFIGS = os.environ.get("SUMM_CONFIGS", "mlsum_tr_ext,tes,xtinge-sum_tr_ext").split(",")

QA_DATASET   = os.environ.get("QA_DATASET", "ucsahin/TR-Extractive-QA-82K")

# Hızlı deneme için küçük default limitler (artırılabilir)
MAX_TRNEWS_SUMM_SAMPLES = int(os.environ.get("MAX_TRNEWS_SUMM_SAMPLES", 2000))
MAX_SUMM_HF_SAMPLES     = int(os.environ.get("MAX_SUMM_HF_SAMPLES", 2000))     # her xtinge config için
MAX_QA_SAMPLES          = int(os.environ.get("MAX_QA_SAMPLES", 4000))

TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", 0.8))
VAL_RATIO   = float(os.environ.get("VAL_RATIO", 0.1))
SEED        = int(os.environ.get("SEED", 42))

PREFIX_SUMM = os.environ.get("PREFIX_SUMM", "summarize: ")
PREFIX_QA   = os.environ.get("PREFIX_QA",   "answer: ")

random.seed(SEED)

# =======================
# Yardımcılar
# =======================
def three_way_split(rows: List[Dict], train_ratio=0.8, val_ratio=0.1, seed=42) -> DatasetDict:
    random.seed(seed)
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train = Dataset.from_list(rows[:n_train]) if n_train > 0 else Dataset.from_list([])
    val   = Dataset.from_list(rows[n_train:n_train+n_val]) if n_val > 0 else Dataset.from_list([])
    test  = Dataset.from_list(rows[n_train+n_val:]) if (n - n_train - n_val) > 0 else Dataset.from_list([])
    return DatasetDict({"train": train, "validation": val, "test": test})

def save_xy_files(dsd: DatasetDict, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    for split in ["train","validation","test"]:
        Xp = os.path.join(out_dir, f"{prefix}_{split}_X.jsonl")
        Yp = os.path.join(out_dir, f"{prefix}_{split}_y.jsonl")
        with open(Xp, "w", encoding="utf-8") as fx, open(Yp, "w", encoding="utf-8") as fy:
            for ex in dsd[split]:
                fx.write(json.dumps({"source": ex["source"]}, ensure_ascii=False) + "\n")
                fy.write(json.dumps({"target": ex["target"]}, ensure_ascii=False) + "\n")

def pick_text(ex: Dict, keys: List[str]) -> str:
    for k in keys:
        v = ex.get(k)
        if v is None:
            continue
        v = str(v).strip()
        if v:
            return v
    return ""

def safe_load_hf_dataset(path: str, config: Optional[str] = None):
    """
    Bazı ortamlarda config yüklerken dataclass hatası çıkabiliyor.
    Farklı çağrı biçimlerini sırayla dener; olmazsa None döner.
    """
    last_err = None
    try:
        if config is None:
            return load_dataset(path)
        return load_dataset(path=path, name=config)
    except Exception as e:
        last_err = e
    try:
        if config is None:
            return load_dataset(path, trust_remote_code=True)
        return load_dataset(path=path, name=config, trust_remote_code=True)
    except Exception as e:
        last_err = e
    try:
        if config is None:
            return load_dataset(path)
        return load_dataset(path, config)
    except Exception as e:
        last_err = e
    try:
        if config is None:
            return load_dataset(path, trust_remote_code=True)
        return load_dataset(path, config, trust_remote_code=True)
    except Exception as e:
        last_err = e
    print(f"[WARN] Could not load dataset '{path}' (config='{config}'). Skipping. Last error: {last_err}")
    return None

# =======================
# Summarization: TR-News (title≈summary)
# =======================
def build_trnews_summ() -> DatasetDict:
    ds = safe_load_hf_dataset(TR_NEWS_NAME)
    if ds is None:
        print("[TR-NEWS] not loaded; returning empty splits.")
        return DatasetDict({"train":Dataset.from_list([]),"validation":Dataset.from_list([]),"test":Dataset.from_list([])})
    rows = []
    splits = [s for s in ["train","validation","test"] if s in ds] or ["train"]
    for split in splits:
        for ex in ds[split]:
            article = pick_text(ex, ["content","text","body"])
            title   = pick_text(ex, ["title","summary","headline"])
            if len(article) >= 200 and title:
                rows.append({"source": PREFIX_SUMM + article, "target": title})
    if MAX_TRNEWS_SUMM_SAMPLES and len(rows) > MAX_TRNEWS_SUMM_SAMPLES:
        rows = rows[:MAX_TRNEWS_SUMM_SAMPLES]
    print(f"[TR-NEWS→SUMM] total={len(rows)}")
    return three_way_split(rows, TRAIN_RATIO, VAL_RATIO, SEED)

# =======================
# Summarization: xtinge configs
# =======================
def build_xtinge_summ() -> DatasetDict:
    all_rows: List[Dict] = []
    for cfg in [c.strip() for c in SUMM_CONFIGS if c.strip()]:
        ds = safe_load_hf_dataset(SUMM_DATASET, cfg)
        if ds is None:
            continue
        rows = []
        splits = [s for s in ["train","validation","dev","test"] if s in ds] or ["train"]
        for split in splits:
            for ex in ds[split]:
                article = pick_text(ex, ["document","text","content","body"])
                summ    = pick_text(ex, ["summary","title","highlight"])
                if len(article) >= 150 and summ:
                    rows.append({"source": PREFIX_SUMM + article, "target": summ})
        if MAX_SUMM_HF_SAMPLES and len(rows) > MAX_SUMM_HF_SAMPLES:
            rows = rows[:MAX_SUMM_HF_SAMPLES]
        all_rows.extend(rows)
        print(f"[XTINGE:{cfg}] added={len(rows)}")
    if not all_rows:
        print("[XTINGE] No data loaded; returning empty splits.")
        return DatasetDict({"train":Dataset.from_list([]),"validation":Dataset.from_list([]),"test":Dataset.from_list([])})
    return three_way_split(all_rows, TRAIN_RATIO, VAL_RATIO, SEED)

# =======================
# QA: TR-Extractive-QA-82K
# =======================
def build_qa() -> DatasetDict:
    ds = safe_load_hf_dataset(QA_DATASET)
    if ds is None:
        print("[QA] not loaded; returning empty splits.")
        return DatasetDict({"train":Dataset.from_list([]),"validation":Dataset.from_list([]),"test":Dataset.from_list([])})
    rows = []
    splits = [s for s in ["train","validation","dev","test"] if s in ds] or ["train","validation"]
    for split in splits:
        for ex in ds[split]:
            ctx = pick_text(ex, ["context","passage","paragraph","article","text"])
            q   = pick_text(ex, ["question","query"])
            ans = ex.get("answer")
            if not ans and isinstance(ex.get("answers"), dict):
                texts = ex["answers"].get("text") or []
                ans = texts[0] if texts else ""
            ans = str(ans or "").strip()
            if ctx and q and ans:
                rows.append({"source": f"{PREFIX_QA}{q} context: {ctx}", "target": ans})
    if MAX_QA_SAMPLES and len(rows) > MAX_QA_SAMPLES:
        rows = rows[:MAX_QA_SAMPLES]
    print(f"[QA:{QA_DATASET}] total={len(rows)}")
    return three_way_split(rows, TRAIN_RATIO, VAL_RATIO, SEED)

# =======================
# Builder
# =======================
def build_and_save():
    trnews = build_trnews_summ()
    xtinge = build_xtinge_summ()
    qa     = build_qa()

    # Birleşik multitask (aynı kolon: source/target)
    train = concatenate_datasets([trnews["train"], xtinge["train"], qa["train"]])
    val   = concatenate_datasets([trnews["validation"], xtinge["validation"], qa["validation"]])
    test  = concatenate_datasets([trnews["test"], xtinge["test"], qa["test"]])
    multi = DatasetDict({"train": train, "validation": val, "test": test})

    # Kaydet
    os.makedirs(OUT_DIR, exist_ok=True)
    multi.save_to_disk(OUT_DIR)
    print("[INFO] Saved HF dataset to:", OUT_DIR)

    # Analiz için X/y jsonl
    save_xy_files(multi, OUT_DIR, prefix="multi")
    save_xy_files(trnews, os.path.join(OUT_DIR, "summ_trnews"), prefix="summ_trnews")
    save_xy_files(xtinge, os.path.join(OUT_DIR, "summ_xtinge"), prefix="summ_xtinge")
    save_xy_files(qa,     os.path.join(OUT_DIR, "qa_tr82k"),   prefix="qa")
    print("[INFO] Saved X/y jsonl files.")

if __name__ == "__main__":
    build_and_save()
