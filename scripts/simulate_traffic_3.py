#!/usr/bin/env python3
"""
IoMT Traffic Simulator - Synthetic 90/10 Edition
=================================================
This script generates SYNTHETIC network traffic with a controlled ratio:
- 90% Benign traffic
- 10% Attack traffic (randomly selected from DDoS, DoS, MQTT, Recon, Spoofing)

Purpose: Test the model's False Positive/Negative rates with known ground truth.

Model: Uses multiclass_model.zip (NOT binary)
"""

import time
import sys
import os
import random
import requests
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

# Model Files
MODEL_FILE = "multiclass_model.zip"
LABEL_ENCODER_FILE = "label_encoder.pkl"
SCALER_FILE = "scaler.pkl"
FEATURE_NAMES_FILE = "feature_names.pkl"

# Traffic Ratio
BENIGN_RATIO = 0.90  # 90% Benign
ATTACK_RATIO = 0.10  # 10% Attack

# Attack Types (excluding Benign)
ATTACK_TYPES = ['DDoS', 'DoS', 'MQTT', 'Recon', 'Spoofing']

# Simulation Settings
TOTAL_PACKETS = 500  # Total synthetic packets to generate
ATTACK_DELAY = 1.0
BENIGN_DELAY = 0.02

# ============================================================================
# ARTIFACT LOADING
# ============================================================================

def load_artifacts():
    """Load the MULTICLASS TabNet model and preprocessing artifacts."""
    print("📦 [INFO] Loading MULTICLASS model artifacts...")
    
    model_path = os.path.join(ARTIFACTS_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"❌ [ERROR] Multiclass model not found at {model_path}")
        sys.exit(1)
    
    clf = TabNetClassifier()
    clf.load_model(model_path)
    print(f"   ✅ Loaded model: {MODEL_FILE}")
    
    encoder_path = os.path.join(ARTIFACTS_DIR, LABEL_ENCODER_FILE)
    label_encoder = joblib.load(encoder_path)
    print(f"   ✅ Loaded encoder: {LABEL_ENCODER_FILE}")
    print(f"   📋 Classes: {list(label_encoder.classes_)}")
    
    scaler_path = os.path.join(ARTIFACTS_DIR, SCALER_FILE)
    scaler = joblib.load(scaler_path)
    print(f"   ✅ Loaded scaler: {SCALER_FILE}")
    
    features_path = os.path.join(ARTIFACTS_DIR, FEATURE_NAMES_FILE)
    feature_names = joblib.load(features_path)
    print(f"   ✅ Loaded features: {len(feature_names)} features")
    
    print("✅ [INFO] All artifacts loaded successfully!\n")
    return clf, label_encoder, scaler, feature_names


# ============================================================================
# SYNTHETIC FEATURE GENERATION
# ============================================================================

def generate_benign_features(num_features):
    """
    Generate synthetic BENIGN traffic features.
    Characteristics: Normal packet sizes, standard intervals, low packet counts.
    """
    features = np.zeros(num_features)
    
    # Typical network flow characteristics for benign traffic
    # These indices are approximate - adjust based on actual feature order
    
    # Flow Duration: Normal range (0.1 - 5 seconds)
    features[0] = np.random.uniform(100000, 5000000)
    
    # Packet counts: Low to moderate
    features[1] = np.random.randint(1, 50)      # Total Fwd Packets
    features[2] = np.random.randint(1, 30)      # Total Bwd Packets
    
    # Packet sizes: Normal range
    features[3] = np.random.uniform(40, 500)    # Fwd Packet Length Mean
    features[4] = np.random.uniform(40, 500)    # Bwd Packet Length Mean
    
    # Flow bytes: Moderate
    features[5] = np.random.uniform(500, 5000)  # Flow Bytes/s
    features[6] = np.random.uniform(10, 100)    # Flow Packets/s
    
    # IAT (Inter-Arrival Time): Normal ranges
    features[7] = np.random.uniform(10000, 100000)  # Flow IAT Mean
    features[8] = np.random.uniform(1000, 50000)    # Fwd IAT Mean
    
    # Add some noise to other features
    for i in range(9, num_features):
        features[i] = np.random.uniform(-0.5, 0.5)
    
    return features


def generate_attack_features(num_features, attack_type):
    """
    Generate synthetic ATTACK traffic features.
    Each attack type has distinct characteristics.
    """
    features = np.zeros(num_features)
    
    if attack_type == 'DDoS':
        # DDoS: Very high packet counts, short duration, high rates
        features[0] = np.random.uniform(1000, 100000)       # Short Flow Duration
        features[1] = np.random.randint(1000, 50000)        # Very high Fwd Packets
        features[2] = np.random.randint(100, 5000)          # High Bwd Packets
        features[3] = np.random.uniform(40, 100)            # Small packet size (amplification)
        features[5] = np.random.uniform(100000, 10000000)   # Very high Bytes/s
        features[6] = np.random.uniform(1000, 100000)       # Very high Packets/s
        features[7] = np.random.uniform(1, 1000)            # Very low IAT
        
    elif attack_type == 'DoS':
        # DoS: High packet counts, moderate duration
        features[0] = np.random.uniform(10000, 500000)      # Moderate Flow Duration
        features[1] = np.random.randint(500, 10000)         # High Fwd Packets
        features[2] = np.random.randint(50, 1000)           # Moderate Bwd Packets
        features[5] = np.random.uniform(50000, 5000000)     # High Bytes/s
        features[6] = np.random.uniform(500, 50000)         # High Packets/s
        
    elif attack_type == 'Recon':
        # Recon: Port scanning - many unique ports, low packet per flow
        features[0] = np.random.uniform(1000, 50000)        # Short flows
        features[1] = np.random.randint(1, 5)               # Very few packets per flow
        features[2] = np.random.randint(0, 2)               # Minimal response
        features[3] = np.random.uniform(40, 60)             # SYN packet size
        features[9] = np.random.randint(1, 65535)           # Random ports
        
    elif attack_type == 'Spoofing':
        # Spoofing: ARP/IP spoofing - unusual MAC/IP patterns
        features[0] = np.random.uniform(1000, 100000)
        features[1] = np.random.randint(10, 100)
        features[3] = np.random.uniform(28, 60)             # ARP packet sizes
        features[10] = np.random.uniform(100, 1000)         # Unusual patterns
        
    elif attack_type == 'MQTT':
        # MQTT: Protocol-specific attacks
        features[0] = np.random.uniform(10000, 500000)
        features[1] = np.random.randint(50, 500)
        features[3] = np.random.uniform(10, 100)            # Small MQTT packets
        features[11] = np.random.uniform(0, 1)              # MQTT specific
    
    # Add some noise to remaining features
    for i in range(12, num_features):
        features[i] = np.random.uniform(-1, 1)
    
    return features


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def simulate(target_ip):
    """Main simulation loop with synthetic 90/10 traffic."""
    clf, label_encoder, scaler, feature_names = load_artifacts()
    num_features = len(feature_names)
    classes = list(label_encoder.classes_)
    
    print(f"🏷️ [INFO] Model Classes: {classes}")
    print(f"📊 [INFO] Traffic Ratio: {BENIGN_RATIO*100:.0f}% Benign / {ATTACK_RATIO*100:.0f}% Attack")
    print(f"🎯 [INFO] Total Packets to Generate: {TOTAL_PACKETS}")
    print(f"🚀 [INFO] Starting SYNTHETIC simulation for Device IP: {target_ip}")
    print("=" * 80)
    
    # Counters
    success_count = 0
    false_positive_count = 0
    false_negative_count = 0
    attack_alerts_sent = 0
    
    ground_truth_counts = {'Benign': 0}
    for at in ATTACK_TYPES:
        ground_truth_counts[at] = 0
    
    for i in range(TOTAL_PACKETS):
        # Decide ground truth based on ratio
        if random.random() < BENIGN_RATIO:
            ground_truth = 'Benign'
            raw_features = generate_benign_features(num_features)
        else:
            ground_truth = random.choice(ATTACK_TYPES)
            raw_features = generate_attack_features(num_features, ground_truth)
        
        ground_truth_counts[ground_truth] += 1
        
        try:
            # Scale features
            X = raw_features.reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            # Predict
            probs = clf.predict_proba(X_scaled)[0]
            pred_idx = np.argmax(probs)
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            
            # Determine if prediction matches ground truth
            gt_is_attack = ground_truth != 'Benign'
            pred_is_attack = pred_label.upper() not in ['BENIGN', 'NORMAL', 'SAFE']
            
            # Evaluate result
            if gt_is_attack and pred_is_attack:
                result = "✅ TRUE POSITIVE"
                success_count += 1
            elif not gt_is_attack and not pred_is_attack:
                result = "✅ TRUE NEGATIVE"
                success_count += 1
            elif not gt_is_attack and pred_is_attack:
                result = "❌ FALSE POSITIVE"
                false_positive_count += 1
            else:  # gt_is_attack and not pred_is_attack
                result = "❌ FALSE NEGATIVE"
                false_negative_count += 1
            
            # Log output
            print(f"[{i+1:4}/{TOTAL_PACKETS}] Sending: {ground_truth:10} | Prediction: {pred_label:10} ({confidence:.0%}) | {result}")
            
            # Send to backend ONLY if predicted as attack
            if pred_is_attack:
                attack_alerts_sent += 1
                payload = {
                    "device_ip": target_ip,
                    "attack_type": pred_label,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prediction": {
                        "is_attack": True,
                        "probability": round(confidence * 100, 2),
                        "label": pred_label,
                        "ground_truth": ground_truth
                    },
                    "message": f"⚠️ {pred_label} tespit edildi! (GT: {ground_truth})"
                }
                
                try:
                    response = requests.post(BACKEND_URL, json=payload, timeout=5)
                    print(f"         ➡️ Alert Sent: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"         ❌ Send Failed: {e}")
                
                time.sleep(ATTACK_DELAY)
            else:
                time.sleep(BENIGN_DELAY)
                
        except Exception as e:
            print(f"⚠️ [ERROR] Packet {i}: {e}")
            continue
    
    # Summary
    total = success_count + false_positive_count + false_negative_count
    accuracy = success_count / total * 100 if total > 0 else 0
    fp_rate = false_positive_count / ground_truth_counts['Benign'] * 100 if ground_truth_counts['Benign'] > 0 else 0
    
    print("\n" + "=" * 80)
    print("📊 SYNTHETIC SIMULATION COMPLETE - GROUND TRUTH ANALYSIS")
    print("=" * 80)
    print("\nGround Truth Distribution:")
    for gt, count in sorted(ground_truth_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / TOTAL_PACKETS * 100
        bar = "█" * int(pct / 2)
        print(f"   {gt:12} : {count:4} ({pct:5.1f}%) {bar}")
    
    print("\n" + "-" * 80)
    print("Prediction Results:")
    print(f"   ✅ Correct Predictions:  {success_count:4} ({accuracy:.1f}%)")
    print(f"   ❌ False Positives:      {false_positive_count:4} ({fp_rate:.1f}% FP Rate)")
    print(f"   ❌ False Negatives:      {false_negative_count:4}")
    print(f"   📤 Attack Alerts Sent:  {attack_alerts_sent:4}")
    print("=" * 80)
    
    if fp_rate < 5:
        print("🏆 EXCELLENT! Very low False Positive rate.")
    elif fp_rate < 15:
        print("⚠️ ACCEPTABLE. Some false positives detected.")
    else:
        print("🚨 WARNING! High False Positive rate - model may need tuning.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IoMT Traffic Simulator - Synthetic 90/10 Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Generates SYNTHETIC traffic with controlled ratio:
  - 90% Benign
  - 10% Random Attack (DDoS, DoS, MQTT, Recon, Spoofing)

Compares Ground Truth vs Prediction to measure:
  - True Positives / True Negatives
  - False Positives / False Negatives

Examples:
  python simulate_traffic_3.py --ip 192.168.1.55
  python simulate_traffic_3.py --ip 10.0.0.100 --packets 1000
        """
    )
    parser.add_argument(
        "--ip", 
        type=str, 
        required=True, 
        help="Target Device IP Address"
    )
    parser.add_argument(
        "--packets",
        type=int,
        default=500,
        help="Total packets to generate (default: 500)"
    )
    args = parser.parse_args()
    
    TOTAL_PACKETS = args.packets
    simulate(args.ip)
