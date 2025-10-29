# scripts/merge_to_parquet.py
from pathlib import Path
import pandas as pd
import numpy as np

MANIFEST = Path("data/manifest_processed.csv")
OUTDIR = Path("data/interim")
OUTDIR.mkdir(parents=True, exist_ok=True)

KEEP_COLS = None  # okunan dosyadaki sutunlarin yapisini sabitler, diger dosyalara hizalama yapilir

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # inf -> NaN (sadece sayisal sutunlarda)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df

# dosyalari etiketle ve hizala
def load_files_with_labels(file_rows, expected_cols=None):
    """
    file_rows: DataFrame satırları (manifest'ten) -> her satırda filepath, label, label_id
    expected_cols: ilk dosyadan belirlediğimiz kolon listesi (özellik seti)
    """
    global KEEP_COLS
    dfs = []
    for _, r in file_rows.iterrows():
        fp = r["filepath"]
        lbl = r["label"]
        lid = r["label_id"]

        df = pd.read_csv(fp, low_memory=False)
        df = clean_df(df)

        # Özellik kolonlarını sabitle
        if KEEP_COLS is None:
            KEEP_COLS = list(df.columns)
        # train/test birbirinden farklı kolon seti çıkarsa hizala
        missing = [c for c in KEEP_COLS if c not in df.columns]
        for c in missing:
            df[c] = np.nan
        df = df[KEEP_COLS]  # fazlalıkları at

        # Etiketleri bu dosyanın TÜM satırlarına ekle
        df["label"] = lbl
        df["label_id"] = lid
        df["source_filepath"] = fp  # izlenebilirlik için faydalı
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

def main():
    assert MANIFEST.exists(), "Manifest yok: data/manifest_processed.csv"
    m = pd.read_csv(MANIFEST)
    req = {"filepath", "split", "label", "label_id"}
    assert req.issubset(m.columns), f"Manifest sütunları eksik: {req - set(m.columns)}"

    # TRAIN
    m_train = m[m["split"] == "train"].copy()
    if len(m_train) == 0:
        print("[UYARI] Train dosyası bulunamadı; sadece test birleştirilecek.")
    else:
        train_df = load_files_with_labels(m_train)
        train_out = OUTDIR / "merged_train.parquet"
        train_df.to_parquet(train_out, index=False)
        print(f"TRAIN yazıldı → {train_out}  satır: {len(train_df)}, kolon: {len(train_df.columns)}")
        print("TRAIN sınıf dağılımı (ilk 10):")
        print(train_df["label"].value_counts().head(10))

    # TEST (train kolonlarına hizala)
    global KEEP_COLS
    m_test = m[m["split"] == "test"].copy()
    if len(m_test) == 0:
        print("[UYARI] Test dosyası bulunamadı.")
    else:
        # Eğer TRAIN hiç yoksa, testin ilk dosyasından KEEP_COLS belirleyelim
        if KEEP_COLS is None and len(m_test):
            sample_fp = m_test.iloc[0]["filepath"]
            sample_df = pd.read_csv(sample_fp, low_memory=False)
            KEEP_COLS = list(sample_df.columns)

        test_df = load_files_with_labels(m_test, expected_cols=KEEP_COLS)
        test_out = OUTDIR / "merged_test.parquet"
        test_df.to_parquet(test_out, index=False)
        print(f"TEST yazıldı → {test_out}  satır: {len(test_df)}, kolon: {len(test_df.columns)}")
        print("TEST sınıf dağılımı (ilk 10):")
        print(test_df["label"].value_counts().head(10))

if __name__ == "__main__":
    main()
