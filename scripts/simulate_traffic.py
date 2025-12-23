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
import argparse  # Argümanları okumak için gerekli
from pytorch_tabnet.tab_model import TabNetClassifier
import shap

# Configuration
# Azure Backend URL
BACKEND_URL = "https://iomtbackend.space/ws/internal/report-attack"
ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data/raw/WiFi_and_MQTT/test"

# Simulation Constants
HOSPITAL_ID = 1
DEVICE_ID = 1

def load_artifacts():
    print("[INFO] Loading artifacts...")
    model_path = os.path.join(ARTIFACTS_DIR, "tabnet_model.zip")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        sys.exit(1)
    
    clf = TabNetClassifier()
    clf.load_model(model_path)
    
    encoders_path = os.path.join(ARTIFACTS_DIR, "label_encoders.pkl")
    label_encoders = joblib.load(encoders_path)
    
    features_path = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")
    feature_names = joblib.load(features_path)
    
    return clf, label_encoders, feature_names

def preprocess_row(row, label_encoders, feature_names):
    cols_to_drop = ['source_file', 'basename', 'SubType', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label', 'Protocol']
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
    try:
        if hasattr(val, 'item'):
            return float(val.item())
        if isinstance(val, (list, np.ndarray)):
            val = np.array(val)
            if val.size == 1:
                return float(val.item())
            return float(val.flatten()[0])
        return float(val)
    except Exception:
        return 0.0

def generate_shap_plot(clf, X_sample, feature_names):
    background = np.zeros((1, X_sample.shape[1]))
    explainer = shap.KernelExplainer(clf.predict_proba, background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    
    probs = clf.predict_proba(X_sample)[0]
    pred_idx = np.argmax(probs)
    
    vals = None
    if isinstance(shap_values, list):
        if pred_idx < len(shap_values):
            vals = shap_values[pred_idx][0]
        else:
            vals = shap_values[0][0] if len(shap_values) == 1 else shap_values[-1][0]
    else:
        vals = shap_values[0]

    feature_importance = []
    for name, val in zip(feature_names, vals):
        scalar_val = safe_scalar(val)
        feature_importance.append({
            "name": name,
            "percentage": float(abs(scalar_val)),
            "value_desc": f"{scalar_val:.4f}"
        })
        
    feature_importance.sort(key=lambda x: x['percentage'], reverse=True)
    
    ev = explainer.expected_value
    base_value = safe_scalar(ev[pred_idx] if isinstance(ev, list) else ev)
    sum_shap = np.sum(vals)
    final_value = safe_scalar(sum_shap) + base_value

    xai_plot = {
        "base_value": base_value,
        "final_value": final_value,
        "features": [{"name": n, "value": safe_scalar(v)} for n, v in zip(feature_names, X_sample[0])]
    }
    
    return xai_plot, feature_importance[:5]

# simulate fonksiyonuna target_ip parametresi eklendi
def simulate(target_ip):
    clf, label_encoders, feature_names = load_artifacts()
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("[ERROR] No CSV files found.")
        return

    print(f"[INFO] Starting simulation for Device IP: {target_ip}")
    
    for file_path in csv_files:
        print(f"[INFO] Replaying file: {file_path}")
        df = pd.read_csv(file_path)
        
        for index, row in df.iterrows():
            try:
                X = preprocess_row(row, label_encoders, feature_names)
                probs = clf.predict_proba(X)[0]
                pred_idx = np.argmax(probs)
                pred_label = label_encoders['Label'].inverse_transform([pred_idx])[0]
                confidence = float(probs[pred_idx])
                
                print(f"Packet {index}: {pred_label} ({confidence:.2f})")
                
                if pred_label != "Benign" and confidence > 0.5:
                    print("🚨 ATTACK DETECTED! Generating Alert...")
                    
                    xai_plot, top_features = generate_shap_plot(clf, X, feature_names)
                    
                    # PAYLOAD GÜNCELLENDİ: Azure Backend formatına uygun
                    payload = {
                        # ID'leri GÖNDERMİYORUZ (Backend IP'den bulacak)
                        #"hospital_id": HOSPITAL_ID,
                        #"device_id": DEVICE_ID,
                        "device_ip": target_ip, 
                        "prediction": {
                            "is_attack": True,
                            "probability": float(confidence),
                        },
                        "explanation": top_features,
                        "flow_details": row.to_dict(),
                        "message": f"Saldırı Tespit Edildi: {pred_label}"
                    }
                    
                    try:
                        response = requests.post(BACKEND_URL, json=payload)
                        print(f"   -> Alert Sent: {response.status_code}")
                    except Exception as e:
                        print(f"   -> Failed to send alert: {e}")
                        
                    time.sleep(2)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"[ERROR] Processing row {index}: {e}")
                continue

if __name__ == "__main__":
    # Argümanları burada alıp fonksiyona gönderiyoruz
    parser = argparse.ArgumentParser(description="IoT Traffic Simulator")
    parser.add_argument("--ip", type=str, required=True, help="Target Device IP Address")
    args = parser.parse_args()
    
    simulate(args.ip)