import pandas as pd
import numpy as np
from pathlib import Path

def clean_dataset(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    print(f"✅ Yüklendi: {df.shape} satır, {df.columns.size} sütun")

    # inf değerlerini NaN yap
    df = df.replace([np.inf, -np.inf], np.nan)

    # NaN değerleri median ile doldur
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # uç değerleri clip'le (99.5 persentil)
    for col in df.select_dtypes(include=[np.number]).columns:
        upper = df[col].quantile(0.995)
        df[col] = np.clip(df[col], None, upper)

    # kaydet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"💾 Kaydedildi: {output_path}")

if __name__ == "__main__":
    clean_dataset("data/processed/merged_sample.csv", "data/processed/merged_clean.parquet")
