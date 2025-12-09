import time
import sys
import os
import glob
import pandas as pd
import requests
import joblib

# Configuration
BACKEND_URL = "https://iomt-ids-backend.onrender.com/predict" # Render URL
# BACKEND_URL = "http://localhost:8000/predict/traffic" # Local URL for testing

DATA_DIR = "data/raw/WiFi_and_MQTT/test"
HOSPITAL_ID = 1
DEVICE_ID = 1

def simulate():
    # Load Test Data
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("[ERROR] No CSV files found.")
        return

    print(f"[INFO] Found {len(csv_files)} test files. Starting simulation...")
    
    # Pick a file
    for file_path in csv_files:
        print(f"[INFO] Replaying file: {file_path}")
        df = pd.read_csv(file_path)
        
        # Iterate rows
        for index, row in df.iterrows():
            try:
                # Convert row to dict
                features = row.to_dict()
                
                # Payload
                payload = {
                    "hospital_id": HOSPITAL_ID,
                    "device_id": DEVICE_ID,
                    "features": features
                }
                
                # Send to Backend
                start_time = time.time()
                response = requests.post(BACKEND_URL, json=payload)
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("device_status", "UNKNOWN")
                    risk = data.get("risk_score", 0.0)
                    summary = data.get("summary_text", "")
                    
                    # Color output
                    color = "\033[92m" if status == "SAFE" else "\033[91m"
                    reset = "\033[0m"
                    
                    print(f"Packet {index}: {color}{status} ({risk:.2f}){reset} - {summary} ({latency:.0f}ms)")
                else:
                    print(f"[ERROR] Failed to send packet {index}: {response.status_code} - {response.text}")

                # Sleep to simulate real-time
                time.sleep(1) 
                    
            except Exception as e:
                print(f"[ERROR] Processing row {index}: {e}")
                time.sleep(1)

if __name__ == "__main__":
    simulate()
