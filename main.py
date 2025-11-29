import os
import sys
from src.processing.loader import DataLoader

def main():
    print("===============================================================")
    print("   Intrusion Detection System (IDS) - Project Initialization   ")
    print("===============================================================")
    
    # Define data path
    # Assuming the user moved the data as instructed or we moved it
    data_path = os.path.abspath("data/raw/WiFi_and_MQTT")
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Data directory not found: {data_path}")
        print("Please ensure you have run 'setup_project.py' and placed the dataset in 'data/raw/'.")
        sys.exit(1)
        
    print(f"[INFO] Target Data Directory: {data_path}")
    
    # Initialize DataLoader
    loader = DataLoader(data_path)
    
    # Load Data
    print("\n[STEP 1] Loading Data...")
    df = loader.load_data()
    
    if df is None:
        print("[ERROR] Failed to load data.")
        sys.exit(1)
        
    print("\n[INFO] Initial Data Schema:")
    df.printSchema()
    
    print("\n[INFO] Sample Raw Data (Top 5 rows):")
    df.show(5, truncate=False)
    
    # Preprocess Data
    print("\n[STEP 2] Preprocessing Data...")
    df_clean = loader.preprocess(df)
    
    print("\n[INFO] Preprocessing Complete.")
    print(loader.get_summary(df_clean))
    
    print("\n[INFO] Sample Cleaned Data (Top 5 rows):")
    df_clean.show(5, truncate=False)
    
    print("\n[INFO] Verifying Label Extraction:")
    df_clean.select("basename", "Protocol", "Label", "SubType").distinct().show(20, truncate=False)
    
    print("\n===============================================================")
    print("   Initialization & Loading Test Complete   ")
    print("===============================================================")

if __name__ == "__main__":
    main()
