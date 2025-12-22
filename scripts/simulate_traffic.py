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
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import shap

# Configuration
# Render adresini buraya yapıştırıyoruz (https olmasına dikkat et)
BACKEND_URL = "https://cic-iomt-eda-tabnet-xai.onrender.com/ws/internal/report-attack"
ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data/raw/WiFi_and_MQTT/test"
HOSPITAL_ID = 1 # Dummy ID from seed.py
DEVICE_ID = 1   # Dummy ID from seed.py

def load_artifacts():
    print("[INFO] Loading artifacts...")
    
    # Load Model
    model_path = os.path.join(ARTIFACTS_DIR, "tabnet_model.zip")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        sys.exit(1)
    
    clf = TabNetClassifier()
    clf.load_model(model_path)
    
    # Load Encoders
    encoders_path = os.path.join(ARTIFACTS_DIR, "label_encoders.pkl")
    label_encoders = joblib.load(encoders_path)
    
    # Load Feature Names
    features_path = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")
    feature_names = joblib.load(features_path)
    
    return clf, label_encoders, feature_names

def preprocess_row(row, label_encoders, feature_names):
    # Drop metadata
    cols_to_drop = ['source_file', 'basename', 'SubType', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label', 'Protocol']
    # Note: 'Label' is the target, we drop it for inference. 'Protocol' might be used?
    # In train_test_run.py:
    # cols_to_drop = ['source_file', 'basename', 'SubType', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    # It keeps 'Protocol' and 'Label' (target).
    # We must drop 'Label' for X.
    
    # Create a copy
    row = row.copy()
    
    # Handle missing columns if any
    for col in cols_to_drop:
        if col in row.index:
            row = row.drop(col)
            
    # Ensure all feature columns exist
    for col in feature_names:
        if col not in row.index:
            row[col] = 0 # Default value
            
    # Reorder columns to match training
    row = row[feature_names]
    
    # Encode Categorical
    # We need to know which columns are categorical.
    # We can infer from label_encoders keys.
    for col, le in label_encoders.items():
        if col in row.index and col != 'Label':
            val = str(row[col])
            # Handle unseen labels
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0 # Default or unknown
                
    # Convert to numeric
    row = pd.to_numeric(row, errors='coerce').fillna(0)
    
    return row.values.reshape(1, -1)

def safe_scalar(val):
    """Safely convert numpy arrays/scalars to Python float."""
    try:
        if hasattr(val, 'item'):
            return float(val.item())
        if isinstance(val, (list, np.ndarray)):
            val = np.array(val)
            if val.size == 1:
                return float(val.item())
            # If array has multiple elements, take the first one (fallback)
            return float(val.flatten()[0])
        return float(val)
    except Exception:
        return 0.0

def generate_shap_plot(clf, X_sample, feature_names):
    # Calculate SHAP for this sample
    # Using a small background (zeros) for speed
    background = np.zeros((1, X_sample.shape[1]))
    explainer = shap.KernelExplainer(clf.predict_proba, background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    
    # shap_values can be a list (multi-class) or array (binary/single output)
    # We need to handle both cases safely.
    
    probs = clf.predict_proba(X_sample)[0]
    pred_idx = np.argmax(probs)
    
    vals = None
    
    if isinstance(shap_values, list):
        # Multi-output case (e.g. classifier with multiple classes)
        if pred_idx < len(shap_values):
            vals = shap_values[pred_idx][0]
        else:
            # Fallback
            if len(shap_values) == 1:
                 vals = shap_values[0][0]
            else:
                 vals = shap_values[-1][0]
    else:
        # Single output case (ndarray)
        vals = shap_values[0]

    # Create Feature Importance List
    feature_importance = []
    for name, val in zip(feature_names, vals):
        scalar_val = safe_scalar(val)
        feature_importance.append({
            "name": name,
            "percentage": float(abs(scalar_val)),
            "value_desc": f"{scalar_val:.4f}"
        })
        
    # Sort by importance
    feature_importance.sort(key=lambda x: x['percentage'], reverse=True)
    
    # Handle expected_value safely
    ev = explainer.expected_value
    base_value = 0.0
    
    if isinstance(ev, list):
        if pred_idx < len(ev):
            base_value = safe_scalar(ev[pred_idx])
        else:
            base_value = safe_scalar(ev[-1] if ev else 0.0)
    elif isinstance(ev, np.ndarray):
        if ev.size > 1 and pred_idx < ev.size:
             base_value = safe_scalar(ev.flatten()[pred_idx])
        else:
             base_value = safe_scalar(ev)
    else:
        base_value = safe_scalar(ev)

    # Calculate final value
    # Sum of SHAP values + base value
    sum_shap = np.sum(vals)
    final_value = safe_scalar(sum_shap) + base_value

    # Mock Force Plot data
    xai_plot = {
        "base_value": base_value,
        "final_value": final_value,
        "features": [{"name": n, "value": safe_scalar(v)} for n, v in zip(feature_names, X_sample[0])]
    }
    
    return xai_plot, feature_importance[:5]

def simulate():
    clf, label_encoders, feature_names = load_artifacts()
    
    # Load Test Data
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print("[ERROR] No CSV files found.")
        return

    print(f"[INFO] Found {len(csv_files)} test files. Starting simulation...")
    
    # Pick a random file or iterate
    for file_path in csv_files:
        print(f"[INFO] Replaying file: {file_path}")
        df = pd.read_csv(file_path)
        
        # Iterate rows
        for index, row in df.iterrows():
            try:
                # Preprocess
                X = preprocess_row(row, label_encoders, feature_names)
                
                # Predict
                probs = clf.predict_proba(X)[0]
                pred_idx = np.argmax(probs)
                pred_label = label_encoders['Label'].inverse_transform([pred_idx])[0]
                confidence = float(probs[pred_idx])
                
                print(f"Packet {index}: {pred_label} ({confidence:.2f})")
                
                # If Attack
                if pred_label != "Benign" and confidence > 0.5:
                    print("🚨 ATTACK DETECTED! Generating Alert...")
                    
                    # Generate SHAP
                    xai_plot, top_features = generate_shap_plot(clf, X, feature_names)
                    
                    # Construct Payload
                    payload = {
                        "hospital_id": HOSPITAL_ID,
                        "device_id": DEVICE_ID,
                        "type": "ALERT",
                        "message": f"Detected {pred_label} Attack!",
                        "timestamp": str(pd.Timestamp.now()),
                        "analysis": {
                            "device_status": "DANGER",
                            "risk_score": confidence * 100,
                            "summary_text": f"High confidence {pred_label} attack detected.",
                            "xai_force_plot": xai_plot,
                            "feature_importance_list": top_features
                        }
                    }
                    
                    # Send to Backend
                    try:
                        response = requests.post(BACKEND_URL, json=payload)
                        print(f"   -> Alert Sent: {response.status_code}")
                    except Exception as e:
                        print(f"   -> Failed to send alert: {e}")
                        
                    # Sleep to simulate real-time and allow UI to update
                    time.sleep(2)
                else:
                    # Sleep briefly for benign
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"[ERROR] Processing row {index}: {e}")
                continue

if __name__ == "__main__":
    simulate()
