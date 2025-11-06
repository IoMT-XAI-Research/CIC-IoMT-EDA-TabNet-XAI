#!/usr/bin/env python3
"""
Spark Comparative Test:
- Load CSV via Spark
- Clean with Spark (median fill, clip, inf->null)
- Add basic network features on Spark
- Collect a sampled Pandas DF
- Apply extended FeatureEngineering (Pandas) without selection/PCA to get rich features
- Compute Mutual Information, and add rationale per feature family
- Save JSON summary under artifacts/results/feature_engineering_spark_comparison.json
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
from src.data.spark_loader import create_spark, load_csv, clean_spark_dataframe, add_basic_network_features
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.utils.logging import setup_logging


def rationale_for_feature(name: str) -> str:
    # Map feature patterns to human-readable rationale
    rules = [
        ("tcp_ratio", "Protocol balance anomalies; flood patterns alter TCP/UDP ratios"),
        ("http_ratio", "Clear-text vs TLS mix; suspicious surges in HTTP can indicate misuse"),
        ("protocol_diversity", "Broader protocol usage in scans/reconnaissance increases diversity"),
        ("flag_diversity", "Aggregate TCP flags; abnormal SYN/ACK/ RST dynamics signal attacks"),
        ("syn_ack_ratio", "SYN flood raises SYN relative to ACK (half-open connections)"),
        ("rst_ratio", "RST anomalies appear in scanning/connection-reset storms"),
        ("packet_rate", "Floods elevate rate level/variance; CV/STD catch volatility"),
        ("iat_", "Inter-Arrival-Time instability is typical for DoS/DDoS"),
        ("size_efficiency", "Anomalous payload per packet/connection indicates crafted traffic"),
        ("_mean_", "Windowed mean captures trend shifts"),
        ("_std_", "Windowed std captures variability spikes"),
        ("_cv_", "Coefficient of variation normalizes volatility vs mean"),
        ("_min_", "Min values detect low bound behavior under constraints"),
        ("_max_", "Max values expose burst peaks in attacks"),
        ("_q25_", "Lower quartile summarises lower-tail distribution"),
        ("_q75_", "Upper quartile summarises upper-tail distribution"),
        ("_iqr_", "IQR captures spread and outliers within a window"),
        ("_lag_", "Lagged values encode temporal dependencies"),
        ("_diff_", "Differences measure abrupt changes between steps"),
        ("_ma_", "Moving averages smooth short-term fluctuations"),
        ("_x_", "Interactions combine signals multiplicatively"),
        ("_div_", "Ratios capture relative magnitude across features"),
        ("_pow_", "Non-linear scaling to emphasize heavy tails/large values"),
    ]
    for key, why in rules:
        if key in name:
            return why
    return "General engineered feature capturing statistical or protocol behavior"


def compute_mutual_info(X: pd.DataFrame, y: pd.Series, top_k: int = 20) -> pd.DataFrame:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
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
        return 1

    spark = create_spark()

    # 1) Load via Spark
    sdf = load_csv(spark, data_path, header=True, infer_schema=True)

    # 2) Clean via Spark
    sdf_clean = clean_spark_dataframe(sdf, exclude=[target_col])

    # 3) Add basic network features via Spark
    sdf_fe_basic = add_basic_network_features(sdf_clean)

    # 4) Collect sampled Pandas for extended FE and MI
    sample_n = 20000
    pdf = sdf_fe_basic.limit(sample_n).toPandas()

    # Describe before extended FE
    base_X = pdf.drop(columns=[target_col]) if target_col in pdf.columns else pdf.copy()
    base_desc = {
        "rows": int(len(pdf)),
        "features": int(base_X.shape[1]),
        "missing_values": int(base_X.isna().sum().sum()),
        "infinite_values": int(np.isinf(base_X.select_dtypes(include=[np.number])).sum().sum()),
    }

    # 5) Extended FE in Pandas
    engineer = FeatureEngineer()
    X_ext, y_ext = engineer.engineer_all_features(
        pdf, target_col=target_col, feature_selection=False, pca=False, scaling=False
    )

    fe_desc = {
        "features": int(X_ext.shape[1]),
        "missing_values": int(X_ext.isna().sum().sum()),
        "infinite_values": int(np.isinf(X_ext.select_dtypes(include=[np.number])).sum().sum()),
    }

    # 6) MI
    mi_df = compute_mutual_info(X_ext, y_ext, top_k=20)
    mi_df["rationale"] = mi_df["feature"].apply(rationale_for_feature)

    # 7) Save JSON
    out_dir = Path("artifacts/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "spark_before_fe": base_desc,
        "after_extended_fe": fe_desc,
        "mi_top": mi_df.to_dict(orient="records"),
    }
    with open(out_dir / "feature_engineering_spark_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 8) Print concise
    print("\n=== Spark Comparative Report ===")
    print(f"Rows (sampled): {base_desc['rows']}")
    print(f"Before FE (Spark-cleaned): features={base_desc['features']}, missing={base_desc['missing_values']}, inf={base_desc['infinite_values']}")
    print(f"After Extended FE (Pandas): features={fe_desc['features']}, missing={fe_desc['missing_values']}, inf={fe_desc['infinite_values']}")

    print("\nTop-10 features by MI with rationale:")
    for i, row in mi_df.head(10).iterrows():
        print(f"  - {row['feature']}: MI={row['mi']:.5f} | why: {row['rationale']}")

    print("\nSaved: artifacts/results/feature_engineering_spark_comparison.json")
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
