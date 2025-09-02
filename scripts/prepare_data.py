import json
import os
import random
from datasets import load_dataset, Dataset, DatasetDict

# --- Sabitler (ihtiyacına göre düzenle) ---
TR_NEWS_NAME = "batubayk/TR-News"
SUMM_PATH = "data1/turkish_summ/train.jsonl"# data1 kullanıyorsan: "data1/..."
QA_PATH   = "data1/turkish_qa/train.jsonl"    # data1 kullanıyorsan: "data1/..."

PREFIX_SUMM = "summarize: "
PREFIX_QA   = "answer: "

random.seed(42)


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def three_way_split(rows, train_ratio=0.8, val_ratio=0.1, seed=42):
    """rows -> DatasetDict(train/validation/test)  (80/10/10 varsayılan)"""
    if not rows:
        raise ValueError("Split oluşturmak için satır bulunamadı (rows boş).")
    random.seed(seed)
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train = Dataset.from_list(rows[:n_train])
    val   = Dataset.from_list(rows[n_train:n_train + n_val])
    test  = Dataset.from_list(rows[n_train + n_val:])
    return DatasetDict({"train": train, "validation": val, "test": test})


def build_summarization_split():
    """TR-News (title≈özet) + varsa kendi özet çiftlerin → text2text (summarize:)"""
    tr = load_dataset(TR_NEWS_NAME)
    rows = []
    for split in ["train", "test"]:
        for ex in tr[split]:
            article = (ex.get("content") or ex.get("text") or "").strip()
            title   = (ex.get("title") or "").strip()
            if len(article) < 200 or not title:
                continue
            rows.append({"source": PREFIX_SUMM + article, "target": title})

    # Ek özet çiftlerini merge et (opsiyonel)
    for r in read_jsonl(SUMM_PATH):
        article = r.get("article", "").strip()
        summary = r.get("summary", "").strip()
        if article and summary:
            rows.append({"source": PREFIX_SUMM + article, "target": summary})

    dsd = three_way_split(rows, train_ratio=0.8, val_ratio=0.1, seed=42)
    print(f"[INFO] SUMM sizes → train={len(dsd['train'])} | val={len(dsd['validation'])} | test={len(dsd['test'])}")
    return dsd


def build_qa_split():
    """Kendi QA çiftlerin → text2text (answer: <q> context: <ctx>)"""
    rows = []
    for r in read_jsonl(QA_PATH):
        ctx = r.get("context", "").strip()
        q   = r.get("question", "").strip()
        a   = r.get("answer", "").strip()
        if ctx and q and a:
            rows.append({"source": f"{PREFIX_QA}{q} context: {ctx}", "target": a})

    if rows:
        dsd = three_way_split(rows, train_ratio=0.8, val_ratio=0.1, seed=42)
    else:
        # QA verin yoksa boş ama şemalı DatasetDict dönüyoruz
        dsd = DatasetDict({
            "train": Dataset.from_list([]),
            "validation": Dataset.from_list([]),
            "test": Dataset.from_list([]),
        })
    print(f"[INFO] QA   sizes → train={len(dsd['train'])} | val={len(dsd['validation'])} | test={len(dsd['test'])}")
    return dsd


def build_multitask_dataset():
    """Özet + QA birleşik multitask seti (train/val/test)"""
    summ = build_summarization_split()
    qa   = build_qa_split()

    train = Dataset.from_list(summ["train"].to_list()        + qa["train"].to_list())
    val   = Dataset.from_list(summ["validation"].to_list()   + qa["validation"].to_list())
    test  = Dataset.from_list(summ["test"].to_list()         + qa["test"].to_list())

    dsd = DatasetDict({"train": train, "validation": val, "test": test})
    print(f"[INFO] MULTI sizes → train={len(train)} | val={len(val)} | test={len(test)}")
    return dsd


if __name__ == "__main__":
    dsd = build_multitask_dataset()
    out_dir = "data1/processed/multitask_text2text"  # data1 kullanıyorsan burayı: "data1/processed/multitask_text2text"
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    dsd.save_to_disk(out_dir)
    print("[INFO] Saved to:", out_dir)
