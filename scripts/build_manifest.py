# scripts/build_manifest.py
import re, json
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/WiFi_and_MQTT")
OUT = Path("data/manifest_processed.csv")
LABELS = Path("configs/labels.json")

# dosya adindan temiz bir etiket cikar (TCP_IP-DDoS-ICMP5_train.pcap.csv → TCP_IP_DDoS_ICMP)
def normalize_label_from_filename(fname: str) -> str:
    """
    'TCP_IP-DDoS-ICMP5_train.pcap.csv' -> 'TCP_IP_DDoS_ICMP'
    'Recon-Port_Scan_test.pcap.csv'    -> 'Recon_Port_Scan'
    'Benign_train.pcap.csv'            -> 'Benign'
    """
    base = fname
    # sondaki _train/_test ve uzantıyı sil
    base = re.sub(r'_(train|test)\.pcap\.csv$', '', base)
    base = base.replace('.pcap.csv','')  # güvenlik
    # tireleri altçizgi yap
    base = base.replace('-', '_')
    # numaralı varyantları sadeleştir (ICMP1..8 -> ICMP; UDP1..8 -> UDP; TCP1..4 -> TCP; SYN1..4 -> SYN)
    base = re.sub(r'(ICMP|UDP|TCP|SYN)\d+$', r'\1', base)
    # çift altçizgileri düzelt
    base = re.sub(r'__+', '_', base)
    return base

def main():
    # RAW klasoru gercekten var mi diye kontrol et
    assert RAW.exists(), f"{RAW} yok!"
    label_map = json.loads(LABELS.read_text(encoding="utf-8"))
    rows = [] # her dosya icin satir buraya eklenecek

    # dosyalari dolas ve etiketleri cikar
    for split in ["train", "test"]:
        split_dir = RAW / split
        for p in split_dir.glob("*.pcap.csv"):
            fname = p.name
            label = normalize_label_from_filename(fname)
            # kontrol: labels.json’da var mı?
            if label not in label_map:
                print(f"[UYARI] '{label}' labels.json içinde yok → {fname}")
            rows.append({
                "filepath": str(p.as_posix()),
                "split": split,
                "label": label,
                "label_id": label_map.get(label, -1)
            })

    # rows listesini DataFrame'e cevirir, siralar
    df = pd.DataFrame(rows).sort_values(["split","label"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    
    print(f"Yazıldı → {OUT}  (satır: {len(df)})")
    print("split dağılımı:\n", df["split"].value_counts())
    print("label örnekleri:\n", df["label"].value_counts().head(10))

if __name__ == "__main__":
    main()
