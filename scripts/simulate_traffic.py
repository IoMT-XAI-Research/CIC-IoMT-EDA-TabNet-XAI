#!/usr/bin/env python3
"""
IoMT Traffic Simulator - Chaos Mode Edition v2.0
=================================================
This script simulates realistic network traffic by mixing packets from
multiple attack/benign CSV files randomly. This provides a more realistic
test scenario compared to sequential file replay.

Features:
- Traffic Mixer: Loads samples from ALL CSVs and shuffles them
- Mobile-Ready Payloads: JSON format optimized for Flutter app
- SHAP Explanations: Top features for AI transparency
- Console Feedback: Shows source file for each packet
- Confidence Threshold: Filters out low-confidence predictions
- Ground Truth Validation: Compares predictions vs filename labels
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
import shap

# ============================================================================
# CONFIGURATION
# ============================================================================
BACKEND_URL = "https://iomtbackend.space/ws/internal/report-attack"
ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data/raw/WiFi_and_MQTT/test"

# Sampling Configuration (to avoid memory overload)
MAX_ROWS_PER_FILE = 1000  # Load up to 1000 rows per file for mixing
ATTACK_DELAY = 1.5        # Seconds to wait after sending an attack alert
BENIGN_DELAY = 0.05       # Faster processing for benign traffic

# Confidence Threshold - predictions below this are treated as Benign
MIN_CONFIDENCE_THRESHOLD = 0.60  # 60%

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_artifacts():
    """Load the trained TabNet model and preprocessing artifacts."""
    print("📦 [INFO] Loading AI artifacts...")
    model_path = os.path.join(ARTIFACTS_DIR, "tabnet_model.zip")
    if not os.path.exists(model_path):
        print(f"❌ [ERROR] Model not found at {model_path}")
        sys.exit(1)
    
    clf = TabNetClassifier()
    clf.load_model(model_path)
    
    encoders_path = os.path.join(ARTIFACTS_DIR, "label_encoders.pkl")
    label_encoders = joblib.load(encoders_path)
    
    features_path = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")
    feature_names = joblib.load(features_path)
    
    print("✅ [INFO] Artifacts loaded successfully!")
    return clf, label_encoders, feature_names


def preprocess_row(row, label_encoders, feature_names):
    """Preprocess a single row for model inference."""
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
    
    # Label encode categorical columns
    for col, le in label_encoders.items():
        if col in row.index and col != 'Label':
            val = str(row[col])
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0
    
    row = pd.to_numeric(row, errors='coerce').fillna(0)
    return row.values.reshape(1, -1)


def safe_scalar(val):
    """Safely convert numpy/torch values to Python float."""
    try:
        if val is None:
            return 0.0
        if hasattr(val, 'item'):
            return float(val.item())
        if isinstance(val, (list, np.ndarray)):
            val = np.array(val)
            if val.size == 0:
                return 0.0
            if val.size == 1:
                return float(val.item())
            return float(val.flatten()[0])
        return float(val)
    except Exception:
        return 0.0


def generate_shap_explanation(clf, X_sample, feature_names):
    """
    Generate SHAP explanation for a prediction.
    Includes robust error handling for IndexError issues.
    """
    try:
        background = np.zeros((1, X_sample.shape[1]))
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        
        probs = clf.predict_proba(X_sample)[0]
        pred_idx = np.argmax(probs)
        
        # Handle different SHAP output formats with safe indexing
        vals = None
        if isinstance(shap_values, list):
            if len(shap_values) == 0:
                # Empty SHAP values - return default
                return _default_explanation()
            
            # Safe index access
            safe_idx = min(pred_idx, len(shap_values) - 1)
            safe_idx = max(0, safe_idx)  # Ensure non-negative
            
            if len(shap_values[safe_idx]) > 0:
                vals = shap_values[safe_idx][0]
            else:
                vals = np.zeros(len(feature_names))
        else:
            if shap_values is not None and len(shap_values) > 0:
                vals = shap_values[0]
            else:
                vals = np.zeros(len(feature_names))
        
        # Validate vals length matches feature_names
        if vals is None or len(vals) != len(feature_names):
            print(f"   ⚠️ SHAP length mismatch: {len(vals) if vals is not None else 0} vs {len(feature_names)}")
            return _default_explanation()

        # Build feature importance list
        feature_importance = []
        for name, val in zip(feature_names, vals):
            scalar_val = safe_scalar(val)
            feature_importance.append({
                "name": name,
                "contribution": round(scalar_val, 4),
                "impact": "positive" if scalar_val > 0 else "negative"
            })
            
        feature_importance.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Return top 5 features
        return feature_importance[:5]
        
    except Exception as e:
        print(f"   ⚠️ SHAP Error: {e}")
        return _default_explanation()


def _default_explanation():
    """Return a default explanation when SHAP fails."""
    return [
        {"name": "Unknown Feature", "contribution": 0.0, "impact": "neutral"}
    ]


def get_ground_truth_from_filename(filename: str) -> str:
    """
    Infer ground truth label from the source filename.
    Returns 'BENIGN' or 'ATTACK' based on filename patterns.
    """
    filename_lower = filename.lower()
    if 'benign' in filename_lower:
        return 'BENIGN'
    else:
        # Anything else (DDoS, Spoofing, Recon, etc.) is an attack
        return 'ATTACK'


def load_and_mix_traffic():
    """
    CHAOS MODE: Load samples from ALL CSV files and shuffle them randomly.
    This simulates real-world mixed traffic patterns.
    """
    print("🌀 [INFO] Loading and mixing traffic data (CHAOS MODE)...")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        print("❌ [ERROR] No CSV files found in", DATA_DIR)
        sys.exit(1)
    
    all_samples = []
    
    for file_path in csv_files:
        basename = os.path.basename(file_path)
        try:
            # Load limited rows for memory efficiency
            df = pd.read_csv(file_path, nrows=MAX_ROWS_PER_FILE)
            df['_source_file'] = basename  # Track source for console output
            all_samples.append(df)
            print(f"   📄 Loaded {len(df)} rows from {basename}")
        except Exception as e:
            print(f"   ⚠️ Failed to load {basename}: {e}")
            continue
    
    if not all_samples:
        print("❌ [ERROR] Could not load any data!")
        sys.exit(1)
    
    # Combine all DataFrames
    combined_df = pd.concat(all_samples, ignore_index=True)
    print(f"\n🎲 [INFO] Total packets loaded: {len(combined_df)}")
    
    # SHUFFLE for chaos!
    combined_df = combined_df.sample(frac=1, random_state=None).reset_index(drop=True)
    print("🔀 [INFO] Traffic shuffled randomly for realistic simulation!")
    print(f"🎯 [INFO] Confidence Threshold: {MIN_CONFIDENCE_THRESHOLD:.0%}\n")
    
    return combined_df


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def simulate(target_ip):
    """Main simulation loop with chaos mode traffic mixing."""
    clf, label_encoders, feature_names = load_artifacts()
    mixed_traffic = load_and_mix_traffic()
    
    print(f"🚀 [INFO] Starting simulation for Device IP: {target_ip}")
    print("=" * 70)
    
    # Counters
    attack_count = 0
    benign_count = 0
    suppressed_count = 0  # Low confidence attacks ignored
    false_positive_count = 0
    false_negative_count = 0
    
    for index, row in mixed_traffic.iterrows():
        source_file = row.get('_source_file', 'Unknown')
        ground_truth = get_ground_truth_from_filename(source_file)
        
        try:
            X = preprocess_row(row, label_encoders, feature_names)
            probs = clf.predict_proba(X)[0]
            pred_idx = np.argmax(probs)
            pred_label = label_encoders['Label'].inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            
            # Determine if model thinks it's an attack
            model_says_attack = pred_label != "Benign"
            
            # Apply confidence threshold
            if model_says_attack and confidence < MIN_CONFIDENCE_THRESHOLD:
                # LOW CONFIDENCE - Suppress alert
                suppressed_count += 1
                print(f"🟡 [{source_file}] Packet {index}: {pred_label} ({confidence:.2%}) -> IGNORED (Low Confidence)")
                time.sleep(BENIGN_DELAY)
                continue
            
            # Final decision
            is_attack = model_says_attack and confidence >= MIN_CONFIDENCE_THRESHOLD
            
            # Ground Truth Validation
            gt_mismatch = ""
            if is_attack and ground_truth == 'BENIGN':
                false_positive_count += 1
                gt_mismatch = " ❌ FALSE POSITIVE!"
            elif not is_attack and ground_truth == 'ATTACK':
                false_negative_count += 1
                gt_mismatch = " ❌ FALSE NEGATIVE!"
            
            if is_attack:
                attack_count += 1
                # Color-coded console output
                print(f"🔴 [{source_file}] Packet {index}: {pred_label} (Confidence: {confidence:.2%}){gt_mismatch}")
                print("   🚨 ATTACK DETECTED! Generating Alert...")
                
                # Generate SHAP explanation (with error handling)
                top_features = generate_shap_explanation(clf, X, feature_names)
                
                # Build Mobile-Ready Payload
                payload = {
                    "device_ip": target_ip,
                    "attack_type": pred_label,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prediction": {
                        "is_attack": True,
                        "probability": round(confidence * 100, 2),  # 0-100 format
                        "label": pred_label
                    },
                    "explanation": top_features,
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
                # Minimal output for benign traffic (every 50th packet)
                if index % 50 == 0:
                    print(f"🟢 [{source_file}] Packet {index}: {pred_label} (Safe){gt_mismatch}")
                time.sleep(BENIGN_DELAY)
                    
        except Exception as e:
            print(f"⚠️ [ERROR] Processing packet {index}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SIMULATION COMPLETE")
    print(f"   🔴 Attacks Sent:       {attack_count}")
    print(f"   🟢 Benign Packets:     {benign_count}")
    print(f"   🟡 Suppressed (Low):   {suppressed_count}")
    print(f"   ❌ False Positives:    {false_positive_count}")
    print(f"   ❌ False Negatives:    {false_negative_count}")
    print(f"   📦 Total Processed:    {attack_count + benign_count + suppressed_count}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IoMT Traffic Simulator - Chaos Mode Edition v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simulate_traffic.py --ip 192.168.1.55
  python simulate_traffic.py --ip 10.0.0.100
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