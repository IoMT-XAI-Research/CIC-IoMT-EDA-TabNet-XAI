from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, regexp_extract, col, when, lit, count, mean, stddev, min, max
from pyspark.sql.types import DoubleType
import os
import glob

class DataLoader:
    def __init__(self, base_path):
        """
        Initializes the DataLoader with a SparkSession.
        
        Args:
            base_path (str): Path to the root directory containing the CSV files.
        """
        self.base_path = base_path
        print(f"[INFO] Initializing SparkSession...")
        self.spark = SparkSession.builder \
            .appName("IDS_Project_Loader") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        print(f"[INFO] SparkSession initialized.")
        print(f"[INFO] Spark Version: {self.spark.version}")

    def parse_filename(self, df):
        """
        Extracts Label, SubType, and Protocol from the filename using regex.
        Assumes filename format like: Protocol-Label-SubType_train.pcap.csv
        
        Args:
            df (DataFrame): Input DataFrame with 'filename' column (from input_file_name()).
            
        Returns:
            DataFrame: DataFrame with new columns 'Label', 'SubType', 'Protocol'.
        """
        # Regex patterns based on examples:
        # MQTT-DDoS-Connect_Flood_train.pcap.csv
        # ARP_Spoofing_test.pcap.csv
        # Recon-OS_Scan_test.pcap.csv
        # TCP_IP-DDoS-UDP1_train.pcap.csv
        
        # We will use a general regex to capture parts separated by hyphens or underscores before _train/_test
        # This is a bit complex due to variations. Let's try a more specific approach based on the known structure.
        
        # Strategy:
        # 1. Extract the basename.
        # 2. Use regexp_extract to parse specific known patterns.
        
        # Add a column with just the filename
        df = df.withColumn("basename", regexp_extract(input_file_name(), r"([^/]+)$", 1))
        
        # Pattern 1: Protocol-Label-SubType (e.g., MQTT-DDoS-Connect_Flood)
        # Pattern 2: Protocol-Label (e.g., ARP_Spoofing - here Protocol=ARP, Label=Spoofing? Or Label=Spoofing, SubType=ARP?)
        # Let's look at the user request examples:
        # ARP_Spoofing -> Label=Spoofing (User didn't specify Protocol, but implied ARP)
        # Recon-OS_Scan -> Label=Recon, SubType=OS_Scan (User said Label=Recon)
        
        # Let's try to capture the first part as Protocol (if hyphen exists) or Label?
        # Actually, the user said: "MQTT-DDoS-Connect_Flood_train.pcap.csv -> Label=DDoS, SubType=Connect_Flood, Protocol=MQTT"
        
        # Let's define a flexible regex.
        # We can extract the part before _train or _test.
        # e.g. "MQTT-DDoS-Connect_Flood"
        
        # Then split that string.
        # But Spark SQL functions are better than UDFs for performance.
        
        # Let's use a sequence of when/otherwise or specific regexes for known types if possible, 
        # or a generic one.
        
        # Generic Regex for "Part1-Part2-Part3_..."
        # Group 1: Protocol
        # Group 2: Label
        # Group 3: SubType
        
        # Case 1: 3 parts (MQTT-DDoS-Connect_Flood)
        p3 = r"^([^-]+)-([^-]+)-([^_]+)_.*"
        
        # Case 2: 2 parts (Recon-OS_Scan, ARP_Spoofing)
        # Wait, ARP_Spoofing uses underscore? "ARP_Spoofing_test.pcap.csv"
        # Recon uses hyphen? "Recon-OS_Scan_test.pcap.csv"
        
        # Let's handle specific prefixes.
        
        df = df.withColumn("Protocol", 
            when(col("basename").startswith("MQTT-"), "MQTT")
            .when(col("basename").startswith("TCP_IP-"), "TCP_IP")
            .when(col("basename").startswith("Recon-"), "Recon") # Recon might be a Label, but let's call it Protocol for now or handle separately
            .when(col("basename").startswith("ARP_"), "ARP")
            .when(col("basename").startswith("Benign"), "Benign")
            .otherwise("Unknown")
        )
        
        # Now extract Label and SubType based on Protocol
        
        # MQTT
        # Format: MQTT-Label-SubType_...
        df = df.withColumn("Label",
            when(col("Protocol") == "MQTT", regexp_extract(col("basename"), r"MQTT-([^-]+)-", 1))
            .when(col("Protocol") == "TCP_IP", regexp_extract(col("basename"), r"TCP_IP-([^-]+)-", 1))
            .when(col("Protocol") == "Recon", "Recon") # User example: Recon-OS_Scan -> Label=Recon? Or Label=OS_Scan? User said "Recon-OS_Scan_test.pcap.csv" -> Label=Recon usually in these datasets.
            .when(col("Protocol") == "ARP", "Spoofing") # ARP_Spoofing
            .when(col("Protocol") == "Benign", "Benign")
            .otherwise("Unknown")
        )
        
        df = df.withColumn("SubType",
            when(col("Protocol") == "MQTT", regexp_extract(col("basename"), r"MQTT-[^-]+-([^_]+)_", 1))
            .when(col("Protocol") == "TCP_IP", regexp_extract(col("basename"), r"TCP_IP-[^-]+-([^_]+)_", 1))
            .when(col("Protocol") == "Recon", regexp_extract(col("basename"), r"Recon-([^_]+)_", 1)) # Recon-OS_Scan -> OS_Scan
            .when(col("Protocol") == "ARP", "ARP_Spoofing")
            .when(col("Protocol") == "Benign", "Benign")
            .otherwise("Unknown")
        )
        
        return df

    def load_data(self):
        """
        Loads all CSV files from the base_path into a single Spark DataFrame.
        """
        print(f"[INFO] Scanning directory: {self.base_path}...")
        
        # Find all CSV files recursively
        # We can just pass the path with wildcard to spark.read.csv
        # But to be safe and verbose as requested, let's count them first.
        
        # glob pattern for recursive search
        search_pattern = os.path.join(self.base_path, "**", "*.csv")
        files = glob.glob(search_pattern, recursive=True)
        
        if not files:
            print(f"[WARNING] No CSV files found in {self.base_path}")
            return None
            
        print(f"[INFO] Found {len(files)} CSV files.")
        
        # Read all files
        # spark.read.csv can take a list of files.
        # We use the list of files found by glob to ensure we only read CSVs and avoid things like index.html
        print(f"[INFO] Reading {len(files)} files into Spark DataFrame...")
        
        # Spark 3.x can take a list of paths
        # We need to make sure paths are absolute or relative correctly. glob returns what we gave it.
        # If the list is very long, we might hit command line limits if we pass them as args, 
        # but spark.read.csv([list]) handles it in python.
        
        df = self.spark.read.option("header", "true") \
                            .option("inferSchema", "true") \
                            .csv(files)
                            
        # Add filename column
        df = df.withColumn("source_file", input_file_name())
        
        # Parse filenames
        print(f"[INFO] Parsing filenames to extract labels...")
        df = self.parse_filename(df)
        
        return df

    def preprocess(self, df):
        """
        Preprocesses the DataFrame:
        1. Converts Infinity/-Infinity to NaN.
        2. Fills NaN with Column Median.
        3. Casts columns to numeric types where appropriate.
        """
        print(f"[INFO] Starting preprocessing...")
        
        # 1. Convert Infinity to NaN
        # In Spark, we can replace string "Infinity" or float('inf')
        # The dataset might have string "Infinity" or "Inf".
        
        # Identify numeric columns (excluding the ones we just added)
        numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, DoubleType) or f.name not in ["source_file", "basename", "Label", "SubType", "Protocol", "Flow ID", "Src IP", "Dst IP", "Timestamp"]]
        
        # Actually, inferSchema might have made some columns strings if they contained "Infinity".
        # Let's check schema later. For now, let's try to cast to double and handle errors?
        # Or replace "Infinity" strings first.
        
        print(f"[INFO] Handling Infinity and NaN values...")
        
        # Replace "Infinity", "Inf" with NaN in all columns
        # This is expensive if done blindly. Let's target potential numeric columns.
        # For simplicity in this initial loader, we will iterate over columns that look numeric or are strings but should be numeric.
        
        # A better approach for Spark:
        # Select columns that are not our metadata columns.
        feature_cols = [c for c in df.columns if c not in ["source_file", "basename", "Label", "SubType", "Protocol", "Flow ID", "Src IP", "Dst IP", "Timestamp"]]
        
        for col_name in feature_cols:
            # Cast to DoubleType first. 
            # This converts numeric types to Double.
            # It also converts "Infinity" strings to Double.Infinity.
            # It converts other non-numeric strings to null.
            df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
            
            # Now replace Infinity with NaN
            df = df.withColumn(col_name, 
                               when(col(col_name) == float("inf"), float("nan"))
                               .when(col(col_name) == float("-inf"), float("nan"))
                               .otherwise(col(col_name)))
        
        # 2. Fill NaN with Median
        print(f"[INFO] Calculating medians for imputation...")
        
        # Calculating median for all columns can be expensive.
        # We use approxQuantile for efficiency in Spark.
        
        # We need to do this for each column.
        # To avoid multiple passes, we can try to do it in groups or just accept the cost for now (dataset size is manageable?).
        # User asked for "Column Median".
        
        medians = {}
        for col_name in feature_cols:
            # 0.5 quantile is median
            # relativeError 0.01
            median_val = df.stat.approxQuantile(col_name, [0.5], 0.01)[0]
            medians[col_name] = median_val
            
        print(f"[INFO] Imputing NaNs with calculated medians...")
        df = df.fillna(medians)
        
        return df

    def get_summary(self, df):
        """
        Returns a summary of the DataFrame.
        """
        count = df.count()
        cols = len(df.columns)
        return f"Rows: {count}, Columns: {cols}"

if __name__ == "__main__":
    # Test run
    loader = DataLoader("data/raw/WiFi_and_MQTT")
    df = loader.load_data()
    if df:
        df.show(5)
        df = loader.preprocess(df)
        df.show(5)
