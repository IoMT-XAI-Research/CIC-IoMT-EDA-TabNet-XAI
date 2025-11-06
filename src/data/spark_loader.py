"""
Spark-based data loading and cleaning utilities for IoMT IDS.
"""

from typing import List, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T


def create_spark(app_name: str = "IoMT_IDS_Spark") -> SparkSession:
    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    return spark


def load_csv(spark: SparkSession, path: str, header: bool = True, infer_schema: bool = True) -> DataFrame:
    df = (
        spark.read.option("header", str(header).lower())
        .option("inferSchema", str(infer_schema).lower())
        .csv(path)
    )
    return df


def _numeric_columns(df: DataFrame, exclude: List[str] = None) -> List[str]:
    exclude = set(exclude or [])
    num_types = {T.IntegerType, T.LongType, T.FloatType, T.DoubleType, T.ShortType}
    cols: List[str] = []
    for f in df.schema.fields:
        if type(f.dataType) in num_types and f.name not in exclude:
            cols.append(f.name)
    return cols


def replace_inf_with_null(df: DataFrame, cols: List[str]) -> DataFrame:
    # If source CSV had string 'inf'/'-inf', they might be read as strings; cast first
    for c in cols:
        df = df.withColumn(c, F.col(c).cast("double"))
        df = df.withColumn(
            c,
            F.when(F.col(c) == float("inf"), F.lit(None))
             .when(F.col(c) == float("-inf"), F.lit(None))
             .otherwise(F.col(c))
        )
    return df


def fillna_with_median(df: DataFrame, cols: List[str]) -> DataFrame:
    for c in cols:
        # approxQuantile for median
        median = df.approxQuantile(c, [0.5], 0.001)[0] if c in df.columns else None
        if median is not None:
            df = df.withColumn(c, F.when(F.col(c).isNull(), F.lit(median)).otherwise(F.col(c)))
    return df


def clip_upper_quantile(df: DataFrame, cols: List[str], quantile: float = 0.995) -> DataFrame:
    for c in cols:
        upper = df.approxQuantile(c, [quantile], 0.001)[0] if c in df.columns else None
        if upper is not None:
            df = df.withColumn(c, F.when(F.col(c) > F.lit(upper), F.lit(upper)).otherwise(F.col(c)))
    return df


def clean_spark_dataframe(df: DataFrame, exclude: List[str] = None) -> DataFrame:
    exclude = exclude or []
    num_cols = _numeric_columns(df, exclude)
    df = replace_inf_with_null(df, num_cols)
    df = fillna_with_median(df, num_cols)
    df = clip_upper_quantile(df, num_cols, 0.995)
    return df


def add_basic_network_features(df: DataFrame) -> DataFrame:
    # Safe division helper: a/(b+1e-8)
    def safe_div(a, b):
        return (a / (b + F.lit(1e-8)))

    # Protocol diversity
    protocol_cols = [
        "HTTP","HTTPS","DNS","Telnet","SMTP","SSH","IRC","TCP","UDP","DHCP","ARP","ICMP","IGMP"
    ]
    available_protocols = [c for c in protocol_cols if c in df.columns]
    if available_protocols:
        df = df.withColumn("protocol_diversity", sum(F.col(c) for c in available_protocols))
        df = df.withColumn("tcp_ratio", safe_div(F.col("TCP"), F.col("UDP")) if "TCP" in df.columns and "UDP" in df.columns else F.lit(None))
        df = df.withColumn("http_ratio", safe_div(F.col("HTTP"), F.col("HTTPS")) if "HTTP" in df.columns and "HTTPS" in df.columns else F.lit(None))

    # Flags
    flag_cols = ["fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number"]
    available_flags = [c for c in flag_cols if c in df.columns]
    if available_flags:
        df = df.withColumn("flag_diversity", sum(F.col(c) for c in available_flags))
        if "syn_flag_number" in df.columns and "ack_flag_number" in df.columns:
            df = df.withColumn("syn_ack_ratio", safe_div(F.col("syn_flag_number"), F.col("ack_flag_number")))
        if "rst_flag_number" in df.columns and "syn_flag_number" in df.columns:
            df = df.withColumn("rst_ratio", safe_div(F.col("rst_flag_number"), F.col("syn_flag_number")))

    return df
