# scripts/eval_cli.py

# python scripts/eval_cli.py -h
# source .venv/bin/activate
# cd scripts

import argparse, os, sys
from eval_metrics import rouge_l_f1, exact_match, token_f1
def normalize(s: str) -> str:
    return " ".join(s.strip().lower().split())

def read_txt_pair(pred_path: str, ref_path: str):
    with open(pred_path, "r", encoding="utf-8") as f:
        preds = [l.rstrip("\n") for l in f]
    with open(ref_path, "r", encoding="utf-8") as f:
        refs = [l.rstrip("\n") for l in f]
    n = min(len(preds), len(refs))
    if len(preds) != len(refs):
        print(f"[WARN] Satır sayısı farklı (pred={len(preds)} ref={len(refs)}). İlk {n} satır kıyaslanacak.", file=sys.stderr)
    return preds[:n], refs[:n]

def main():
    ap = argparse.ArgumentParser(description="TR-NEWS değerlendirme CLI")
    ap.add_argument("--text", nargs=2, metavar=("PRED_TXT", "REF_TXT"),
                    help="İki TXT dosyası (satır-satır). Eğer verilmezse ./preds.txt ve ./refs.txt aranır.")
    ap.add_argument("--plot", nargs='?', const="eval_plot.png", metavar="PNG_PATH",
                    help="Skor çubuğu grafiği kaydet (opsiyonel). Eğer yol verilmezse ./eval_plot.png kullanılır.")
    ap.add_argument("--show-samples", type=int, default=0, help="EM=0 örneklerinden N adet göster.")
    args = ap.parse_args()

    # Resolve input text files: use provided ones or fall back to ./preds.txt and ./refs.txt
    if args.text:
        pred_path, ref_path = args.text[0], args.text[1]
    else:
        pred_path, ref_path = os.path.join('.', 'preds.txt'), os.path.join('.', 'refs.txt')

    if not os.path.exists(pred_path) or not os.path.exists(ref_path):
        print(f"HATA: Girdi dosyaları bulunamadı: {pred_path} veya {ref_path}. Lütfen --text ile yolları verin.", file=sys.stderr)
        sys.exit(2)

    preds, refs = read_txt_pair(pred_path, ref_path)
    if not preds:
        print("HATA: Kıyaslanacak örnek yok.", file=sys.stderr)
        sys.exit(2)

    r = rouge_l_f1(preds, refs)
    em = exact_match(preds, refs)
    f1 = token_f1(preds, refs)

    print("\n=== Değerlendirme Sonuçları ===")
    print(f"Örnek sayısı : {len(preds)}")
    print(f"ROUGE-L  F1  : {r:.4f}")
    print(f"Token    F1  : {f1:.4f}")
    print(f"ExactMatch   : {em:.4f}")

    if args.show_samples > 0:
        print("\n--- Örnek farklar (EM=0) ---")
        shown = 0
        for i, (p, r_) in enumerate(zip(preds, refs)):
            if normalize(p) != normalize(r_):
                print(f"[{i}] PRED: {p}")
                print(f"    REF : {r_}\n")
                shown += 1
                if shown >= args.show_samples:
                    break
        if shown == 0:
            print("(Tüm tahminler tam eşleşiyor.)")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print("[WARN] matplotlib bulunamadı. Lütfen sanal ortam aktifleştirip 'pip install matplotlib' çalıştırın.", file=sys.stderr)
            print(f"Detay: {e}", file=sys.stderr)
        else:
            xs, ys = ["ROUGE-L F1", "Token F1", "ExactMatch"], [r, f1, em]
            plt.figure()
            plt.bar(xs, ys)
            plt.ylim(0, 1.0)
            plt.title("Değerlendirme Metrikleri")
            out_path = args.plot or "eval_plot.png"
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            try:
                plt.savefig(out_path, bbox_inches="tight", dpi=150)
                print(f"\n[OK] Grafik kaydedildi: {out_path}")
            except Exception as e:
                print(f"[WARN] Grafik kaydedilemedi: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
