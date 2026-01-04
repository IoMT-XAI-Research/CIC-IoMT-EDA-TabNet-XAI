"""
simulate_traffic_benign.py
========================
"Warm-up & Burst" Traffic Simulation (Fixed)
- NO pd.get_dummies (training had no categorical columns)
- Uses reindex for exact column alignment
- Suppresses sklearn version warnings
"""

import time
import sys
import os
import glob
import random
import requests
import pandas as pd
import numpy as np
import joblib
import argparse
import warnings

# Suppress sklearn warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)

from pytorch_tabnet.tab_model import TabNetClassifier

# Configuration
BACKEND_URL = "https://iomtbackend.space/ws/internal/report-attack"
ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data/raw/WiFi_and_MQTT/test"


def get_ground_truth(filename):
    """Extract ground truth label from filename."""
    filename_lower = filename.lower()
    
    if "benign" in filename_lower:
        return "Benign"
    elif "ddos" in filename_lower:
        return "DDoS"
    elif "dos" in filename_lower and "ddos" not in filename_lower:
        return "DoS"
    elif "mqtt" in filename_lower:
        return "MQTT"
    elif "recon" in filename_lower:
        return "Recon"
    elif "spoof" in filename_lower:
        return "Spoofing"
    return "Unknown"


def load_artifacts():
    """Load all required artifacts."""
    print("[INFO] Loading artifacts...")
    
    feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, "final_feature_names.pkl"))
    print(f"  ✓ {len(feature_names)} feature names")
    
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "final_preprocessor.pkl"))
    print("  ✓ Scaler")
    
    label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder_multiclass.pkl"))
    print(f"  ✓ Label encoder: {list(label_encoder.classes_)}")
    
    model_path = os.path.join(ARTIFACTS_DIR, "multiclass_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(ARTIFACTS_DIR, "tabnet_model.zip")
    
    clf = TabNetClassifier()
    clf.load_model(model_path)
    print("  ✓ Model loaded")
    
    return clf, scaler, label_encoder, feature_names


def categorize_files(data_dir):
    """Separate files into benign and attack categories."""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    benign_files = []
    attack_files = {}
    
    for f in csv_files:
        filename = os.path.basename(f)
        label = get_ground_truth(filename)
        
        if label == "Benign":
            benign_files.append(f)
        elif label != "Unknown":
            if label not in attack_files:
                attack_files[label] = []
            attack_files[label].append(f)
    
    print(f"\n[DATA] Categorized files:")
    print(f"  - Benign: {len(benign_files)} files")
    for attack_type, files in attack_files.items():
        print(f"  - {attack_type}: {len(files)} files")
    
    return benign_files, attack_files


def load_packets(file_list, max_packets=500):
    """Load packets from a list of files."""
    all_packets = []
    
    for f in file_list:
        try:
            df = pd.read_csv(f, nrows=max_packets // len(file_list) if file_list else max_packets)
            filename = os.path.basename(f)
            label = get_ground_truth(filename)
            
            for _, row in df.iterrows():
                all_packets.append({
                    "data": row,
                    "ground_truth": label,
                    "source": filename
                })
        except Exception as e:
            print(f"[WARNING] Failed to load {f}: {e}")
    
    return all_packets


def preprocess_packet(row, scaler, feature_names):
    """
    Preprocess packet with exact column alignment.
    FIX: No get_dummies (training had no categorical columns).
    """
    if isinstance(row, pd.Series):
        df = pd.DataFrame([row])
    else:
        df = pd.DataFrame([row])
    
    cols_to_drop = ['source_file', 'basename', 'SubType', 'Flow ID', 
                    'Src IP', 'Dst IP', 'Source IP', 'Destination IP',
                    'Timestamp', 'Label', 'Protocol', 'Flow Duration',
                    'Source Port', 'Destination Port', 'Time', 'MAC']
    
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # =====================================================
    # FIX: Do NOT apply get_dummies
    # Training had categorical_columns = [] (empty)
    # Just convert remaining object columns to numeric
    # =====================================================
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df.astype(float)
    
    # CRITICAL: Reindex to exact training columns
    df = df.reindex(columns=feature_names, fill_value=0.0)
    
    X_scaled = scaler.transform(df.values)
    return X_scaled


def format_result(ground_truth, pred_label, confidence):
    """Format prediction result."""
    if ground_truth == pred_label:
        return f"✅ [ACTUAL: {ground_truth}] -> [PRED: {pred_label}] ({confidence:.0%})"
    elif ground_truth == "Benign" and pred_label != "Benign":
        return f"❌ [ACTUAL: {ground_truth}] -> [PRED: {pred_label}] ({confidence:.0%}) FALSE POSITIVE"
    elif ground_truth != "Benign" and pred_label == "Benign":
        return f"⚠️ [ACTUAL: {ground_truth}] -> [PRED: {pred_label}] ({confidence:.0%}) MISSED ATTACK"
    else:
        return f"🔄 [ACTUAL: {ground_truth}] -> [PRED: {pred_label}] ({confidence:.0%}) WRONG TYPE"


def run_scenario(target_ip, send_alerts=True):
    """Run the Warm-up & Burst scenario."""
    
    clf, scaler, label_encoder, feature_names = load_artifacts()
    
    benign_files, attack_files = categorize_files(DATA_DIR)
    
    if not benign_files:
        print("[ERROR] No Benign files found!")
        return
    
    print("\n[LOADING] Loading Benign packets...")
    benign_packets = load_packets(benign_files, max_packets=300)
    print(f"  - Loaded {len(benign_packets)} benign packets")
    
    attack_packets = {}
    for attack_type, files in attack_files.items():
        print(f"\n[LOADING] Loading {attack_type} packets...")
        attack_packets[attack_type] = load_packets(files, max_packets=100)
        print(f"  - Loaded {len(attack_packets[attack_type])} {attack_type} packets")
    
    # Build the Playlist
    traffic_playlist = []
    
    # Phase 1: Warm-up (100 Benign)
    phase1_packets = benign_packets[:100]
    traffic_playlist.append({
        "phase": "PHASE 1: SYSTEM WARM-UP",
        "description": "100 Benign Packets",
        "packets": phase1_packets
    })
    
    # Phase 2: Attack Waves
    benign_idx = 100
    for attack_type, packets in attack_packets.items():
        if len(packets) < 20:
            continue
        
        attack_burst = packets[:20]
        recovery_packets = benign_packets[benign_idx:benign_idx + 20]
        benign_idx += 20
        
        traffic_playlist.append({
            "phase": f"PHASE 2: ATTACK WAVE - {attack_type}",
            "description": f"20 {attack_type} + 20 Benign Recovery",
            "packets": attack_burst + recovery_packets
        })
    
    stats = {"total": 0, "correct": 0, "fp": 0, "fn": 0, "wrong": 0}
    
    print("\n" + "="*70)
    print("[SIMULATION] Starting Warm-up & Burst Scenario")
    print("="*70)
    
    for phase in traffic_playlist:
        phase_name = phase["phase"]
        description = phase["description"]
        packets = phase["packets"]
        
        print(f"\n{'='*70}")
        print(f"=== {phase_name} ===")
        print(f"    {description}")
        print("="*70)
        
        for idx, pkt in enumerate(packets):
            try:
                row = pkt["data"]
                ground_truth = pkt["ground_truth"]
                
                X = preprocess_packet(row, scaler, feature_names)
                
                probs = clf.predict_proba(X)[0]
                pred_idx = np.argmax(probs)
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(probs[pred_idx])
                
                result = format_result(ground_truth, pred_label, confidence)
                print(f"  [{idx+1:3d}] {result}")
                
                stats["total"] += 1
                if ground_truth == pred_label:
                    stats["correct"] += 1
                elif ground_truth == "Benign" and pred_label != "Benign":
                    stats["fp"] += 1
                elif ground_truth != "Benign" and pred_label == "Benign":
                    stats["fn"] += 1
                else:
                    stats["wrong"] += 1
                
                if send_alerts and pred_label != "Benign" and confidence > 0.5:
                    try:
                        requests.post(BACKEND_URL, json={
                            "device_ip": target_ip,
                            "prediction": {"is_attack": True, "probability": confidence},
                            "attack_type": pred_label
                        }, timeout=3)
                    except:
                        pass
                    time.sleep(0.3)
                else:
                    time.sleep(0.05)
                    
            except Exception as e:
                print(f"  [ERROR] Packet {idx}: {e}")
    
    print("\n" + "="*70)
    print("[SUMMARY]")
    print("="*70)
    total = max(1, stats["total"])
    print(f"  Total:         {stats['total']}")
    print(f"  ✅ Correct:     {stats['correct']} ({stats['correct']/total*100:.1f}%)")
    print(f"  ❌ False Pos:   {stats['fp']} ({stats['fp']/total*100:.1f}%)")
    print(f"  ⚠️ Missed:      {stats['fn']} ({stats['fn']/total*100:.1f}%)")
    print(f"  🔄 Wrong Type:  {stats['wrong']} ({stats['wrong']/total*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warm-up & Burst Traffic Simulator")
    parser.add_argument("--ip", type=str, required=True, help="Target Device IP")
    parser.add_argument("--no-alerts", action="store_true", help="Disable backend alerts")
    args = parser.parse_args()
    
    run_scenario(args.ip, send_alerts=not args.no_alerts)

