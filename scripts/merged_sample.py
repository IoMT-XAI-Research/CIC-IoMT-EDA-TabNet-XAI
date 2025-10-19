# scripts/create_merged_sample.py
import pandas as pd
import glob
import os

DATA_ROOT = "data"          # senin yolun
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

per_file_sample = 5000      # her dosyadan çekilecek örnek sayısı (dilediğini ayarla)

csv_files = glob.glob(os.path.join(DATA_ROOT, "**", "*.csv"), recursive=True)
samples = []
meta = []
for f in csv_files:
    try:
        df = pd.read_csv(f)
        n = min(len(df), per_file_sample)
        s = df.sample(n=n, random_state=42)
        # attack_type: dosya yolundan mantıklı bir etiket çıkar
        rel = os.path.relpath(f, DATA_ROOT)
        attack = rel.replace(os.sep, "_").replace(".csv", "")
        s["attack_type"] = attack
        samples.append(s)
        meta.append({"file": rel, "rows": len(df), "sampled": n})
        print(f"Sampled {n} rows from {rel}")
    except Exception as e:
        print("FAILED", f, e)

if samples:
    merged = pd.concat(samples, ignore_index=True)
    out_path = os.path.join(OUT_DIR, "merged_sample.csv")
    merged.to_csv(out_path, index=False)
    meta_df = pd.DataFrame(meta)
    meta_df.to_csv(os.path.join(OUT_DIR, "manifest_samples.csv"), index=False)
    print("Saved merged sample to:", out_path)
else:
    print("No samples created.")
