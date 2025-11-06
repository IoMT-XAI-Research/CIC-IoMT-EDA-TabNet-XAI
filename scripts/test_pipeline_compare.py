#!/usr/bin/env python3
"""
Comparative test: Before vs After Feature Engineering
- Loads data
- Cleans data
- Computes baseline stats
- Applies feature engineering (without selection first), compares stats
- Computes mutual information to justify engineered features
- Saves a compact JSON summary under artifacts/results/
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# Local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.data_loader import DataLoader
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.utils.logging import setup_logging


def describe_dataframe(df: pd.DataFrame, target_col: str) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_count = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
    na_count = int(df.isna().sum().sum())
    desc = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "num_numeric_cols": int(len(numeric_cols)),
        "num_object_cols": int(len(df.select_dtypes(include=["object"]).columns)),
        "missing_values_total": na_count,
        "infinite_values_total": inf_count,
    }
    if target_col in df.columns:
        desc["classes"] = int(df[target_col].nunique())
        desc["class_top5"] = df[target_col].value_counts().head(5).to_dict()
    return desc


def compute_mutual_info(X: pd.DataFrame, y: pd.Series, top_k: int = 15) -> pd.DataFrame:
    # MI expects numeric X and encoded y
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    # Fill any residual NaNs
    X_num = X.select_dtypes(include=[np.number]).copy()
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
    mi = mutual_info_classif(X_num.values, y_enc, random_state=42)
    mi_df = pd.DataFrame({"feature": X_num.columns, "mi": mi}).sort_values("mi", ascending=False)
    return mi_df.head(top_k)


def main():
    setup_logging("INFO")

    data_path = "data/processed/merged_sample.csv"
    target_col = "attack_type"

    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        sys.exit(1)

    loader = DataLoader()
    engineer = FeatureEngineer()

    # 1) Load raw (CSV)
    raw_df = loader.load_data(data_path, file_format="csv")

    # Optionally subsample for speed (comment out to use full data)
    sample_size = min(20000, len(raw_df))
    raw_df = raw_df.head(sample_size)

    # 2) Describe raw
    raw_desc = describe_dataframe(raw_df, target_col)

    # 3) Clean
    clean_df = loader.clean_data(raw_df)
    clean_desc = describe_dataframe(clean_df, target_col)

    # 4) Baseline X, y (before FE)
    X_base, y_base = loader.prepare_features(clean_df, target_col)

    # 5) Engineer features (disable selection/PCA to compare fairly)
    fe_df = clean_df.copy()
    X_fe, y_fe = engineer.engineer_all_features(
        fe_df, target_col=target_col, feature_selection=False, pca=False, scaling=False
    )

    # 6) Align targets
    assert y_base.equals(y_fe), "Targets differ after FE alignment"

    # 7) Comparative metrics
    base_desc = {
        "features": int(X_base.shape[1]),
        "missing_values": int(X_base.isna().sum().sum()),
        "infinite_values": int(np.isinf(X_base.select_dtypes(include=[np.number])).sum().sum()),
    }
    fe_desc = {
        "features": int(X_fe.shape[1]),
        "missing_values": int(X_fe.isna().sum().sum()),
        "infinite_values": int(np.isinf(X_fe.select_dtypes(include=[np.number])).sum().sum()),
        "new_features": [c for c in X_fe.columns if c not in X_base.columns][:50],  # preview
    }

    # 8) Mutual information: top features overall and contribution of new features
    mi_overall = compute_mutual_info(X_fe, y_fe, top_k=20)
    new_feats_set = set([c for c in X_fe.columns if c not in X_base.columns])
    mi_new = mi_overall[mi_overall["feature"].isin(new_feats_set)].head(10)

    # 9) Save JSON summary
    out_dir = Path("artifacts/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "raw": raw_desc,
        "clean": clean_desc,
        "before_fe": base_desc,
        "after_fe": fe_desc,
        "mi_overall": mi_overall.to_dict(orient="records"),
        "mi_new_top": mi_new.to_dict(orient="records"),
    }
    with open(out_dir / "feature_engineering_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 10) Print concise report
    print("\n=== Comparative Report (Before vs After Feature Engineering) ===")
    print(f"Rows (sampled): {raw_desc['shape'][0]}")
    print(f"Target classes: {clean_desc.get('classes', 'N/A')}")

    print("\nBefore FE:")
    print(f"  Features: {base_desc['features']}")
    print(f"  Missing values: {base_desc['missing_values']}")
    print(f"  Infinite values: {base_desc['infinite_values']}")

    print("\nAfter FE:")
    print(f"  Features: {fe_desc['features']} (Δ {fe_desc['features'] - base_desc['features']})")
    print(f"  Missing values: {fe_desc['missing_values']}")
    print(f"  Infinite values: {fe_desc['infinite_values']}")
    print(f"  New features (preview, up to 50): {fe_desc['new_features']}")

    print("\nTop-10 new engineered features by Mutual Information (if present among top-20):")
    if len(mi_new) == 0:
        print("  (No engineered features in top-20 MI ranking for this sample)")
    else:
        for i, row in mi_new.iterrows():
            print(f"  - {row['feature']}: MI={row['mi']:.5f}")

    print("\nFull MI ranking saved to artifacts/results/feature_engineering_comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
