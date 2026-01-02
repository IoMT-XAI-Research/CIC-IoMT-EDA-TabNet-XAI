#!/usr/bin/env python3
"""
IoMT Traffic Simulator - Benign Only Edition
=============================================
This script simulates ONLY benign (normal) network traffic to test
the model's stability and measure False Positive rates.

Purpose:
- Test system behavior under normal network conditions
- Identify and count False Positives
- Validate model accuracy on legitimate traffic

Features:
- Loads ONLY files containing "Benign" in their filename
- Maintains chaos mode architecture (sampling, shuffling)
- Reports False Positive Rate at the end
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

# Sampling Configuration
MAX_ROWS_PER_FILE = 2000  # Load more rows since we only have benign files
ATTACK_DELAY = 1.5        # Delay after sending false positive alert
BENIGN_DELAY = 0.02       # Faster processing for normal traffic

# Confidence Threshold
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
    
    for col in feature_names:
        if col not in row.index:
            row[col] = 0 
    row = row[feature_names]
    
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
    """Generate SHAP explanation with robust error handling."""
    try:
        background = np.zeros((1, X_sample.shape[1]))
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        
        probs = clf.predict_proba(X_sample)[0]
        pred_idx = np.argmax(probs)
        
        vals = None
        if isinstance(shap_values, list):
            if len(shap_values) == 0:
                return _default_explanation()
            safe_idx = min(max(0, pred_idx), len(shap_values) - 1)
            if len(shap_values[safe_idx]) > 0:
                vals = shap_values[safe_idx][0]
            else:
                vals = np.zeros(len(feature_names))
        else:
            if shap_values is not None and len(shap_values) > 0:
                vals = shap_values[0]
            else:
                vals = np.zeros(len(feature_names))
        
        if vals is None or len(vals) != len(feature_names):
            return _default_explanation()

        feature_importance = []
        for name, val in zip(feature_names, vals):
            scalar_val = safe_scalar(val)
            feature_importance.append({
                "name": name,
                "contribution": round(scalar_val, 4),
                "impact": "positive" if scalar_val > 0 else "negative"
            })
            
        feature_importance.sort(key=lambda x: abs(x['contribution']), reverse=True)
        return feature_importance[:5]
        
    except Exception as e:
        print(f"   ⚠️ SHAP Error: {e}")
        return _default_explanation()


def _default_explanation():
    """Return a default explanation when SHAP fails."""
    return [{"name": "Unknown Feature", "contribution": 0.0, "impact": "neutral"}]


def load_benign_traffic_only():
    """
    Load ONLY Benign CSV files for False Positive testing.
    """
    print("🟢 [INFO] Loading BENIGN traffic only (False Positive Test Mode)...")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    # Filter to only include files with "Benign" in the name
    benign_files = [f for f in csv_files if "benign" in os.path.basename(f).lower()]
    
    if not benign_files:
        print("❌ [ERROR] No Benign CSV files found in", DATA_DIR)
        sys.exit(1)
    
    print(f"   📁 Found {len(benign_files)} Benign file(s)")
    
    all_samples = []
    
    for file_path in benign_files:
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
    print(f"\n🎲 [INFO] Total BENIGN packets loaded: {len(combined_df)}")
    
    # Shuffle for randomness
    combined_df = combined_df.sample(frac=1, random_state=None).reset_index(drop=True)
    print("🔀 [INFO] Traffic shuffled randomly!")
    print(f"🎯 [INFO] Confidence Threshold: {MIN_CONFIDENCE_THRESHOLD:.0%}\n")
    
    return combined_df


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def simulate_benign(target_ip):
    """Simulate benign traffic to test False Positive rate."""
    clf, label_encoders, feature_names = load_artifacts()
    benign_traffic = load_benign_traffic_only()
    
    print(f"🚀 [INFO] Starting BENIGN simulation for Device IP: {target_ip}")
    print("=" * 70)
    print("⚠️  All traffic is BENIGN. Any attack detection is a FALSE POSITIVE!")
    print("=" * 70)
    
    # Counters
    total_processed = 0
    benign_correct = 0
    false_positives = 0
    suppressed_low_conf = 0
    
    for index, row in benign_traffic.iterrows():
        source_file = row.get('_source_file', 'Unknown')
        total_processed += 1
        
        try:
            X = preprocess_row(row, label_encoders, feature_names)
            probs = clf.predict_proba(X)[0]
            pred_idx = np.argmax(probs)
            pred_label = label_encoders['Label'].inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            
            model_says_attack = pred_label != "Benign"
            
            if model_says_attack:
                if confidence < MIN_CONFIDENCE_THRESHOLD:
                    # Low confidence - suppress
                    suppressed_low_conf += 1
                    print(f"🟡 [{source_file}] Packet {index}: {pred_label} ({confidence:.2%}) -> SUPPRESSED (Low Confidence)")
                    time.sleep(BENIGN_DELAY)
                else:
                    # FALSE POSITIVE! Model thinks it's an attack but it's benign data
                    false_positives += 1
                    print(f"❌ [{source_file}] Packet {index}: {pred_label} ({confidence:.2%}) -> FALSE POSITIVE!")
                    
                    # ═══════════════════════════════════════════════════════════════
                    # DEEP DIVE LOGGER - Analyze WHY the model made this mistake
                    # ═══════════════════════════════════════════════════════════════
                    
                    # 1. Probability Split - Show competition between classes
                    all_labels = label_encoders['Label'].classes_
                    prob_pairs = list(zip(all_labels, probs))
                    prob_pairs.sort(key=lambda x: x[1], reverse=True)
                    
                    top1_label, top1_prob = prob_pairs[0]
                    top2_label, top2_prob = prob_pairs[1] if len(prob_pairs) > 1 else ("N/A", 0.0)
                    
                    print(f"   ⚖️  Probability Split: {top1_label}: {top1_prob:.0%} vs {top2_label}: {top2_prob:.0%}")
                    
                    # 2. Top 3 Features that caused this decision
                    top_features = generate_shap_explanation(clf, X, feature_names)
                    
                    if top_features and len(top_features) >= 3:
                        f1 = top_features[0]
                        f2 = top_features[1]
                        f3 = top_features[2]
                        print(f"   ❓ Why {pred_label}? 1. {f1['name']} ({f1['contribution']:.2f}) | "
                              f"2. {f2['name']} ({f2['contribution']:.2f}) | "
                              f"3. {f3['name']} ({f3['contribution']:.2f})")
                    elif top_features:
                        feature_str = " | ".join([f"{i+1}. {f['name']} ({f['contribution']:.2f})" 
                                                   for i, f in enumerate(top_features[:3])])
                        print(f"   ❓ Why {pred_label}? {feature_str}")
                    
                    print("   🚨 Sending FALSE POSITIVE alert to test mobile app...")
                    
                    payload = {
                        "device_ip": target_ip,
                        "attack_type": pred_label,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "prediction": {
                            "is_attack": True,
                            "probability": round(confidence * 100, 2),
                            "label": pred_label
                        },
                        "explanation": top_features,
                        "source_file": source_file,
                        "probability_split": {
                            "predicted": {"label": top1_label, "probability": round(top1_prob * 100, 2)},
                            "runner_up": {"label": top2_label, "probability": round(top2_prob * 100, 2)}
                        },
                        "message": f"⚠️ FALSE POSITIVE: {pred_label} tespit edildi!"
                    }
                    
                    try:
                        response = requests.post(BACKEND_URL, json=payload, timeout=5)
                        print(f"   ➡️ Alert Sent: {response.status_code}")
                    except requests.exceptions.RequestException as e:
                        print(f"   ❌ Failed to send alert: {e}")
                    
                    time.sleep(ATTACK_DELAY)
            else:
                # Correctly classified as Benign
                benign_correct += 1
                if index % 100 == 0:
                    print(f"🟢 [{source_file}] Packet {index}: {pred_label} (Correct)")
                time.sleep(BENIGN_DELAY)
                    
        except Exception as e:
            print(f"⚠️ [ERROR] Processing packet {index}: {e}")
            continue
    
    # Calculate False Positive Rate
    fp_rate = (false_positives / total_processed * 100) if total_processed > 0 else 0
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 BENIGN SIMULATION COMPLETE - FALSE POSITIVE ANALYSIS")
    print("=" * 70)
    print(f"   📦 Total Benign Packets Processed: {total_processed}")
    print(f"   ✅ Correctly Classified (Benign):  {benign_correct}")
    print(f"   🟡 Suppressed (Low Confidence):    {suppressed_low_conf}")
    print(f"   ❌ FALSE POSITIVES (Sent Alerts):  {false_positives}")
    print("-" * 70)
    print(f"   📈 FALSE POSITIVE RATE: {fp_rate:.2f}%")
    print("=" * 70)
    
    if fp_rate < 5:
        print("   🏆 EXCELLENT! Model has very low false positive rate.")
    elif fp_rate < 15:
        print("   ⚠️ ACCEPTABLE. Some false positives detected.")
    else:
        print("   🚨 WARNING! High false positive rate - model may need retraining.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IoMT Traffic Simulator - Benign Only (False Positive Test)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Purpose:
  Test the model's behavior with ONLY benign traffic.
  Any attack detection is a FALSE POSITIVE.

Examples:
  python simulate_traffic_benign.py --ip 192.168.1.55
        """
    )
    parser.add_argument(
        "--ip", 
        type=str, 
        required=True, 
        help="Target Device IP Address (must exist in database)"
    )
    args = parser.parse_args()
    
    simulate_benign(args.ip)
