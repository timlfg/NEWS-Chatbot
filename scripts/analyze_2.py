# analysis.py — görev bazlı (SUMM/QA) analiz + epoch grafikleri

import os, sys, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_kv_args(argv):
    args = {}
    for a in argv[1:]:
        if "=" in a:
            k,v=a.split("=",1); args[k.strip()]=v.strip()
    return args

ARGS = parse_kv_args(sys.argv)
OUTPUT_DIR = ARGS.get("OUTPUT_DIR", "outputs/multitask-lora-fast")
METRICS_CSV = os.path.join(OUTPUT_DIR, "metrics.csv")
EVAL_PRED_JSONL = os.path.join(OUTPUT_DIR, "eval_predictions.jsonl")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INFO] OUTPUT_DIR={OUTPUT_DIR}")

# ---------- yardımcılar ----------
def make_lineplot(x, y_list, labels, xlabel, ylabel, title, outpath):
    plt.figure(figsize=(8,5))
    for y,lab in zip(y_list, labels): plt.plot(x,y,label=lab)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close(); print(f"[INFO] saved {outpath}")

def confusion_matrix_binary(y_true, y_pred):
    tp = int(np.sum((y_true==1)&(y_pred==1)))
    tn = int(np.sum((y_true==0)&(y_pred==0)))
    fp = int(np.sum((y_true==0)&(y_pred==1)))
    fn = int(np.sum((y_true==1)&(y_pred==0)))
    return np.array([[tn, fp],[fn, tp]])

def plot_confusion_matrix(cm, labels, title, outpath):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation="nearest")
    plt.xticks([0,1], labels); plt.yticks([0,1], labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,str(cm[i,j]), ha="center", va="center")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close(); print(f"[INFO] saved {outpath}")

def roc_points(y_true, y_score):
    uniq = np.unique(y_score); thresh = np.sort(uniq)[::-1]
    P = np.sum(y_true==1); N = np.sum(y_true==0)
    if P==0 or N==0: return np.array([[0.0,0.0],[1.0,1.0]])
    pts=[]
    for t in thresh:
        y_pred=(y_score>=t).astype(int)
        tp=np.sum((y_true==1)&(y_pred==1)); fp=np.sum((y_true==0)&(y_pred==1))
        tpr=tp/P; fpr=fp/N; pts.append([fpr,tpr])
    pts=np.array(pts+[[0.0,0.0],[1.0,1.0]])
    return pts[np.argsort(pts[:,0], kind="mergesort")]

def auc_trapz(roc_pts): return np.trapz(roc_pts[:,1], roc_pts[:,0])

# Normalizasyon (TR-dostu, noktalama/numara sadeleştirme)
_punc_re = re.compile(r"[^\w\sçğıöşüâîû]", flags=re.IGNORECASE)
_ws_re = re.compile(r"\s+")
def norm(s: str):
    s = str(s).lower()
    s = _punc_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip()
    return s

def token_f1_pair(p,t):
    ps, ts = set(norm(p).split()), set(norm(t).split())
    if not ps or not ts: return 0.0
    inter = len(ps & ts)
    if inter==0: return 0.0
    prec = inter/len(ps); rec = inter/len(ts)
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

def _lcs_len(a,b):
    n,m=len(a),len(b); dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            dp[i+1][j+1] = dp[i][j]+1 if a[i]==b[j] else max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def rouge_l_f1_pair(p,t):
    pt, tt = norm(p).split(), norm(t).split()
    if not pt or not tt: return 0.0
    lcs=_lcs_len(pt,tt); prec=lcs/len(pt); rec=lcs/len(tt)
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

# ---------- 1) metrics.csv → step ve epoch grafikleri ----------
if os.path.exists(METRICS_CSV):
    df = pd.read_csv(METRICS_CSV)
    for col in ["step","train_loss","eval_loss","train_em","eval_em","train_token_f1","eval_token_f1"]:
        if col not in df.columns: df[col]=np.nan

    x = df["step"].values if df["step"].notna().any() else np.arange(len(df))
    make_lineplot(x, [df["train_loss"].values, df["eval_loss"].values],
                  ["train_loss","val_loss"], "step", "loss",
                  "Loss (train vs val)", os.path.join(OUTPUT_DIR,"loss_curves.png"))
    make_lineplot(x, [df["train_token_f1"].values, df["eval_token_f1"].values],
                  ["train_token_f1","val_token_f1"], "step", "token_f1",
                  "Token F1 (train vs val)", os.path.join(OUTPUT_DIR,"tokenf1_curves.png"))
    make_lineplot(x, [df["train_em"].values, df["eval_em"].values],
                  ["train_EM","val_EM"], "step", "EM",
                  "Accuracy (Exact Match)", os.path.join(OUTPUT_DIR,"accuracy_curves_em.png"))

    if "epoch" in df.columns and df["epoch"].notna().any():
        df_ep = df.copy()
        df_ep["epoch_i"] = np.floor(df_ep["epoch"].astype(float)).astype(int)+1
        g = df_ep.groupby("epoch_i").agg({"train_loss":"last","eval_loss":"last"}).reset_index()
        make_lineplot(g["epoch_i"].values, [g["train_loss"].values, g["eval_loss"].values],
                      ["Training Loss","Validation Loss"], "Epochs", "Loss",
                      "Training and Validation Loss", os.path.join(OUTPUT_DIR,"loss_curves_by_epoch.png"))
else:
    print(f"[WARN] metrics.csv yok: {METRICS_CSV}")

# ---------- 2) eval_predictions.jsonl → görev bazlı analiz ----------
summ_rouges=[]; qa_em=[]; qa_f1=[]

if os.path.exists(EVAL_PRED_JSONL):
    with open(EVAL_PRED_JSONL,"r",encoding="utf-8") as f:
        for line in f:
            try:
                r=json.loads(line)
            except: 
                continue
            src = (r.get("source") or "").lower().strip()
            y = r.get("target",""); p = r.get("prediction","")
            if src.startswith("summarize:"):
                summ_rouges.append(rouge_l_f1_pair(p,y))
            elif src.startswith("answer:"):
                # EM’i yeniden, daha toleranslı normalize ederek de hesapla:
                em = 1 if norm(p)==norm(y) else 0
                qa_em.append(em)
                qa_f1.append(token_f1_pair(p,y))
    # SUMM raporu
    if summ_rouges:
        sr=np.array(summ_rouges,dtype=float)
        plt.figure(figsize=(7,4)); plt.hist(sr, bins=30)
        plt.title(f"Summarization — ROUGE-L F1 (mean={sr.mean():.4f})")
        plt.xlabel("ROUGE-L F1"); plt.ylabel("count"); plt.tight_layout()
        out=os.path.join(OUTPUT_DIR,"summ_rouge_hist.png"); plt.savefig(out,dpi=150); plt.close(); print(f"[INFO] saved {out}")
    else:
        print("[WARN] SUMM örneği bulunamadı (veya source prefix 'summarize:' değil).")

    # QA raporu
    if qa_em:
        y_true=np.array(qa_em,dtype=int); y_score=np.array(qa_f1,dtype=float)

        thr=0.5
        y_pred=(y_score>=thr).astype(int)
        cm = confusion_matrix_binary(y_true,y_pred)
        plot_confusion_matrix(cm, ["EM=0","EM=1"],
                              f"Confusion Matrix (QA, thr={thr})",
                              os.path.join(OUTPUT_DIR,"qa_confusion_matrix.png"))

        roc=roc_points(y_true,y_score); auc=auc_trapz(roc)
        plt.figure(figsize=(6,5))
        plt.plot(roc[:,0], roc[:,1], label=f"AUC={auc:.4f}")
        plt.plot([0,1],[0,1],"--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC (QA) — score=token_f1, label=EM"); plt.legend()
        out=os.path.join(OUTPUT_DIR,"qa_roc_curve.png"); plt.tight_layout(); plt.savefig(out,dpi=150); plt.close(); print(f"[INFO] saved {out}")

        # QA F1 histogramı
        plt.figure(figsize=(7,4)); plt.hist(y_score, bins=30)
        plt.title(f"QA — token F1 histogram (mean={y_score.mean():.4f})")
        plt.xlabel("token F1"); plt.ylabel("count"); plt.tight_layout()
        out=os.path.join(OUTPUT_DIR,"qa_tokenf1_hist.png"); plt.savefig(out,dpi=150); plt.close(); print(f"[INFO] saved {out}")
    else:
        print("[WARN] QA örneği bulunamadı (veya source prefix 'answer:' değil).")
else:
    print(f"[WARN] eval_predictions.jsonl yok: {EVAL_PRED_JSONL}")

print("[DONE] Analiz bitti.")
