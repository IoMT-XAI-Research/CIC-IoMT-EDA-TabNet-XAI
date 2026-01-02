#!/usr/bin/env python3
"""
IoMT Traffic Simulator - Multiclass Edition
=============================================
This script uses the MULTICLASS TabNet model to predict specific attack types
(DDoS, DoS, MQTT, Recon, Spoofing) instead of just binary Safe/Attack.

Model Classes: Benign, DDoS, DoS, MQTT, Recon, Spoofing

Requirements:
- multiclass_model.zip
- label_encoder.pkl  
- scaler.pkl
- feature_names.pkl
"""

import time
import sys
import os
import glob
import json
import random
import requests
import pandas as pd
import numpy as np
import joblib
import argparse
from datetime import datetime, timezone
from pytorch_tabnet.tab_model import TabNetClassifier

# ============================================================================
# CONFIGURATION
# ============================================================================
BACKEND_URL = "https://iomtbackend.space/ws/internal/report-attack"
ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data/raw/WiFi_and_MQTT/test"

# Model Files (MULTICLASS ONLY - NO BINARY)
MODEL_FILE = "multiclass_model.zip"
LABEL_ENCODER_FILE = "label_encoder.pkl"
SCALER_FILE = "scaler.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"

# Sampling Configuration
MAX_ROWS_PER_FILE = 1000
ATTACK_DELAY = 1.5
BENIGN_DELAY = 0.05

# Confidence Threshold
MIN_CONFIDENCE_THRESHOLD = 0.60

# ============================================================================
# ARTIFACT LOADING
# ============================================================================

def load_artifacts():
    """Load the MULTICLASS TabNet model and preprocessing artifacts."""
    print("📦 [INFO] Loading MULTICLASS model artifacts...")
    
    # Load Multiclass Model
    model_path = os.path.join(ARTIFACTS_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"❌ [ERROR] Multiclass model not found at {model_path}")
        print("   Make sure 'multiclass_model.zip' exists in artifacts/")
        sys.exit(1)
    
    clf = TabNetClassifier()
    clf.load_model(model_path)
    print(f"   ✅ Loaded model: {MODEL_FILE}")
    
    # Load Label Encoder
    encoder_path = os.path.join(ARTIFACTS_DIR, LABEL_ENCODER_FILE)
    if not os.path.exists(encoder_path):
        print(f"❌ [ERROR] Label encoder not found at {encoder_path}")
        sys.exit(1)
    
    label_encoder = joblib.load(encoder_path)
    print(f"   ✅ Loaded encoder: {LABEL_ENCODER_FILE}")
    print(f"   📋 Classes: {list(label_encoder.classes_)}")
    
    # Load Scaler
    scaler_path = os.path.join(ARTIFACTS_DIR, SCALER_FILE)
    if not os.path.exists(scaler_path):
        print(f"❌ [ERROR] Scaler not found at {scaler_path}")
        sys.exit(1)
    
    scaler = joblib.load(scaler_path)
    print(f"   ✅ Loaded scaler: {SCALER_FILE}")
    
    # Load Feature Names
    features_path = os.path.join(ARTIFACTS_DIR, FEATURE_NAMES_FILE)
    if not os.path.exists(features_path):
        print(f"❌ [ERROR] Feature names not found at {features_path}")
        sys.exit(1)
    
    feature_names = joblib.load(features_path)
    print(f"   ✅ Loaded features: {len(feature_names)} features")
    
    print("✅ [INFO] All artifacts loaded successfully!\n")
    return clf, label_encoder, scaler, feature_names


def preprocess_row(row, scaler, feature_names):
    """Preprocess a single row using the loaded scaler."""
    cols_to_drop = ['_source_file', 'source_file', 'basename', 'SubType', 'Flow ID', 
                    'Src IP', 'Dst IP', 'Timestamp', 'Label', 'Protocol']
    row = row.copy()
    
    for col in cols_to_drop:
        if col in row.index:
            row = row.drop(col)
    
    # Ensure all features exist
    for col in feature_names:
        if col not in row.index:
            row[col] = 0 
    row = row[feature_names]
    
    # Convert to numeric
    row = pd.to_numeric(row, errors='coerce').fillna(0)
    
    # Apply scaler
    X = row.values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    return X_scaled


def load_and_mix_traffic():
    """Load samples from ALL CSV files and shuffle them randomly."""
    print("🌀 [INFO] Loading and mixing traffic data (CHAOS MODE)...")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        print("❌ [ERROR] No CSV files found in", DATA_DIR)
        sys.exit(1)
    
    all_samples = []
    
    for file_path in csv_files:
        basename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path, nrows=MAX_ROWS_PER_FILE)
            df['_source_file'] = basename
            all_samples.append(df)
            print(f"   📄 Loaded {len(df)} rows from {basename}")
        except Exception as e:
            print(f"   ⚠️ Failed to load {basename}: {e}")
            continue
    
    if not all_samples:
        print("❌ [ERROR] Could not load any data!")
        sys.exit(1)
    
    combined_df = pd.concat(all_samples, ignore_index=True)
    print(f"\n🎲 [INFO] Total packets loaded: {len(combined_df)}")
    
    # Shuffle for chaos
    combined_df = combined_df.sample(frac=1, random_state=None).reset_index(drop=True)
    print("🔀 [INFO] Traffic shuffled randomly!")
    print(f"🎯 [INFO] Confidence Threshold: {MIN_CONFIDENCE_THRESHOLD:.0%}\n")
    
    return combined_df


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def simulate(target_ip):
    """Main simulation loop using MULTICLASS model."""
    clf, label_encoder, scaler, feature_names = load_artifacts()
    mixed_traffic = load_and_mix_traffic()
    
    # Available classes from the model
    classes = list(label_encoder.classes_)
    print(f"🏷️ [INFO] Attack Classes: {classes}")
    print(f"🚀 [INFO] Starting MULTICLASS simulation for Device IP: {target_ip}")
    print("=" * 70)
    
    # Counters
    class_counts = {c: 0 for c in classes}
    attack_count = 0
    benign_count = 0
    suppressed_count = 0
    
    for index, row in mixed_traffic.iterrows():
        source_file = row.get('_source_file', 'Unknown')
        
        try:
            X = preprocess_row(row, scaler, feature_names)
            probs = clf.predict_proba(X)[0]
            pred_idx = np.argmax(probs)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            
            class_counts[pred_label] = class_counts.get(pred_label, 0) + 1
            
            # Broad Threat Detection: Anything NOT "Benign" or "Normal" is an attack
            is_attack = pred_label.upper() not in ['BENIGN', 'NORMAL', 'SAFE']
            
            if is_attack:
                if confidence < MIN_CONFIDENCE_THRESHOLD:
                    suppressed_count += 1
                    print(f"🟡 [{source_file}] Packet {index}: {pred_label} ({confidence:.2%}) -> SUPPRESSED")
                    time.sleep(BENIGN_DELAY)
                else:
                    attack_count += 1
                    print(f"🔴 [{source_file}] Packet {index}: {pred_label} ({confidence:.2%}) -> ATTACK!")
                    
                    # Build Multiclass Payload
                    payload = {
                        "device_ip": target_ip,
                        "attack_type": pred_label,  # Specific attack type!
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "prediction": {
                            "is_attack": True,
                            "probability": round(confidence * 100, 2),
                            "label": pred_label,
                            "all_probabilities": {
                                classes[i]: round(float(probs[i]) * 100, 2) 
                                for i in range(len(classes))
                            }
                        },
                        "source_file": source_file,
                        "message": f"⚠️ {pred_label} saldırısı tespit edildi!"
                    }
                    
                    # Send to backend
                    try:
                        response = requests.post(BACKEND_URL, json=payload, timeout=5)
                        print(f"   ➡️ Alert Sent: {response.status_code}")
                    except requests.exceptions.RequestException as e:
                        print(f"   ❌ Failed to send alert: {e}")
                    
                    time.sleep(ATTACK_DELAY)
            else:
                benign_count += 1
                if index % 100 == 0:
                    print(f"🟢 [{source_file}] Packet {index}: {pred_label} (Safe)")
                time.sleep(BENIGN_DELAY)
                    
        except Exception as e:
            print(f"⚠️ [ERROR] Processing packet {index}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 MULTICLASS SIMULATION COMPLETE")
    print("=" * 70)
    print("Class Distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(mixed_traffic) * 100 if len(mixed_traffic) > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   {cls:12} : {count:6} ({pct:5.1f}%) {bar}")
    print("-" * 70)
    print(f"   🔴 Attack Alerts Sent:  {attack_count}")
    print(f"   🟢 Benign Packets:      {benign_count}")
    print(f"   🟡 Suppressed (Low %):  {suppressed_count}")
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IoMT Traffic Simulator - Multiclass Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Uses the MULTICLASS model to predict specific attack types:
  - Benign (Safe)
  - DDoS
  - DoS
  - MQTT
  - Recon
  - Spoofing

Examples:
  python simulate_traffic_2.py --ip 192.168.1.55
        """
    )
    parser.add_argument(
        "--ip", 
        type=str, 
        required=True, 
        help="Target Device IP Address (must exist in database)"
    )
    args = parser.parse_args()
    
    simulate(args.ip)
