"""
simulate_traffic.py
========================
Shuffled Block Traffic Simulation (Fixed)
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

# Block Configuration
MAX_ROWS_PER_FILE = 2000
BLOCK_SIZE = 75


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
    """Load all required artifacts for inference."""
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
    print("  ✓ Model")
    
    return clf, scaler, label_encoder, feature_names


def build_blocks(data_dir, max_rows=2000, block_size=75):
    """Build shuffled blocks from all CSV files."""
    print("\n[BLOCK BUILDER] Loading and chunking data...")
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print("[ERROR] No CSV files found.")
        return []
    
    all_blocks = []
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        ground_truth = get_ground_truth(filename)
        
        print(f"  - Loading: {filename} (Label: {ground_truth})")
        
        df = pd.read_csv(file_path, nrows=max_rows)
        
        n_blocks = len(df) // block_size
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            block_data = df.iloc[start:end].copy()
            
            all_blocks.append({
                "ground_truth": ground_truth,
                "filename": filename,
                "block_id": i + 1,
                "data": block_data
            })
        
        remaining = len(df) % block_size
        if remaining >= 10:
            block_data = df.iloc[n_blocks * block_size:].copy()
            all_blocks.append({
                "ground_truth": ground_truth,
                "filename": filename,
                "block_id": n_blocks + 1,
                "data": block_data
            })
    
    print(f"\n[BLOCK BUILDER] Created {len(all_blocks)} blocks")
    
    random.shuffle(all_blocks)
    print("[BLOCK BUILDER] Blocks shuffled!")
    
    return all_blocks


def preprocess_packet(row, scaler, feature_names):
    """
    Preprocess packet with exact column alignment.
    FIX: No get_dummies (training had no categorical columns).
    """
    if isinstance(row, pd.Series):
        df = pd.DataFrame([row])
    else:
        df = pd.DataFrame([row])
    
    # Drop leakage/ID columns
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
            # Try to convert to numeric, fill failures with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df.astype(float)
    
    # CRITICAL: Reindex to exact training columns
    df = df.reindex(columns=feature_names, fill_value=0.0)
    
    X_scaled = scaler.transform(df.values)
    return X_scaled


def format_comparison(ground_truth, pred_label, confidence):
    """Format comparison output with visual indicators."""
    if ground_truth == pred_label:
        return f"✅ [ACTUAL: {ground_truth}] -> [PRED: {pred_label}] ({confidence:.0%})"
    elif ground_truth == "Benign" and pred_label != "Benign":
        return f"❌ [ACTUAL: {ground_truth}] -> [PRED: {pred_label}] ({confidence:.0%}) - FALSE POSITIVE"
    elif ground_truth != "Benign" and pred_label == "Benign":
        return f"⚠️ [ACTUAL: {ground_truth}] -> [PRED: {pred_label}] ({confidence:.0%}) - MISSED ATTACK"
    else:
        return f"🔄 [ACTUAL: {ground_truth}] -> [PRED: {pred_label}] ({confidence:.0%}) - WRONG TYPE"


def simulate(target_ip, max_blocks=None, send_alerts=True):
    """Main simulation with shuffled blocks."""
    clf, scaler, label_encoder, feature_names = load_artifacts()
    
    all_blocks = build_blocks(DATA_DIR, MAX_ROWS_PER_FILE, BLOCK_SIZE)
    
    if not all_blocks:
        return
    
    if max_blocks:
        all_blocks = all_blocks[:max_blocks]
    
    print(f"\n[SIMULATION] Starting with {len(all_blocks)} blocks")
    print(f"[SIMULATION] Device IP: {target_ip}")
    print("="*70)
    
    stats = {"total": 0, "correct": 0, "false_positive": 0, "false_negative": 0, "wrong_type": 0}
    
    for block_idx, block in enumerate(all_blocks):
        ground_truth = block["ground_truth"]
        filename = block["filename"]
        block_id = block["block_id"]
        data = block["data"]
        
        print(f"\n{'='*70}")
        print(f"=== BLOCK {block_idx + 1}/{len(all_blocks)}: {ground_truth} ({len(data)} packets) ===")
        print(f"    Source: {filename} (Block #{block_id})")
        print("="*70)
        
        for pkt_idx, (_, row) in enumerate(data.iterrows()):
            try:
                X = preprocess_packet(row, scaler, feature_names)
                
                probs = clf.predict_proba(X)[0]
                pred_idx = np.argmax(probs)
                pred_label = label_encoder.inverse_transform([pred_idx])[0]
                confidence = float(probs[pred_idx])
                
                comparison = format_comparison(ground_truth, pred_label, confidence)
                print(f"  [{pkt_idx + 1:3d}] {comparison}")
                
                stats["total"] += 1
                if ground_truth == pred_label:
                    stats["correct"] += 1
                elif ground_truth == "Benign" and pred_label != "Benign":
                    stats["false_positive"] += 1
                elif ground_truth != "Benign" and pred_label == "Benign":
                    stats["false_negative"] += 1
                else:
                    stats["wrong_type"] += 1
                
                if send_alerts and pred_label != "Benign" and confidence > 0.5:
                    try:
                        requests.post(BACKEND_URL, json={
                            "device_ip": target_ip,
                            "prediction": {"is_attack": True, "probability": confidence},
                            "attack_type": pred_label
                        }, timeout=3)
                    except:
                        pass
                    time.sleep(0.5)
                else:
                    time.sleep(0.02)
                    
            except Exception as e:
                print(f"  [ERROR] Packet {pkt_idx}: {e}")
    
    print("\n" + "="*70)
    print("[SUMMARY]")
    print("="*70)
    total = max(1, stats["total"])
    print(f"  Total Packets:   {stats['total']}")
    print(f"  ✅ Correct:       {stats['correct']} ({stats['correct']/total*100:.1f}%)")
    print(f"  ❌ False Pos:     {stats['false_positive']} ({stats['false_positive']/total*100:.1f}%)")
    print(f"  ⚠️ False Neg:     {stats['false_negative']} ({stats['false_negative']/total*100:.1f}%)")
    print(f"  🔄 Wrong Type:    {stats['wrong_type']} ({stats['wrong_type']/total*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffled Block Traffic Simulator")
    parser.add_argument("--ip", type=str, required=True, help="Target Device IP")
    parser.add_argument("--blocks", type=int, default=None, help="Limit number of blocks")
    parser.add_argument("--no-alerts", action="store_true", help="Disable backend alerts")
    args = parser.parse_args()
    
    simulate(args.ip, max_blocks=args.blocks, send_alerts=not args.no_alerts)
