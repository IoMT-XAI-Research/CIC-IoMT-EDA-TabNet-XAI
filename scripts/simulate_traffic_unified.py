"""
simulate_traffic_unified.py
========================
GRAND FINALE - Unified IoMT IDS Simulation
- Rhythmic traffic pattern (10 Benign → 5 Attack per type)
- Cascaded classification (Binary → Multiclass)
- Dual-persona XAI (Doctor + Admin)
- Production-ready demonstration
"""
import requests
import time
import sys
import os
import json
import glob
import random
import pandas as pd
import numpy as np
import joblib
import argparse
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=UserWarning)
from pytorch_tabnet.tab_model import TabNetClassifier
# Configuration
BACKEND_URL = "https://iomtbackend.space/ws/internal/report-attack"
ARTIFACTS_DIR = "artifacts"
DATA_DIR_PRIMARY = "data/all_traffics"
DATA_DIR_FALLBACK = "data/raw/WiFi_and_MQTT/test"
def get_ground_truth(filename):
   """Extract ground truth from filename."""
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

def extract_traffic_details(packet_row):
   """Extract traffic details for mobile app dropdown."""
   details = {'rate': 0.0, 'size': 0.0, 'protocol': 0.0}
  
   # Rate (Traffic Speed)
   for col in ['Rate', 'Flow Bytes/s', 'Fwd Bytes/s']:
       if col in packet_row.index:
           val = packet_row[col]
           if pd.notna(val) and val != np.inf and val != -np.inf:
               details['rate'] = float(val)
               break
  
   # Size (Packet Size)
   for col in ['Tot size', 'Tot Size', 'Total Length', 'Max', 'Len', 'Pkt Len Mean']:
       if col in packet_row.index:
           val = packet_row[col]
           if pd.notna(val) and val != np.inf and val != -np.inf:
               details['size'] = float(val)
               break
  
   # Protocol
   for col in ['Protocol Type', 'Protocol', 'Proto']:
       if col in packet_row.index:
           val = packet_row[col]
           if pd.notna(val):
               details['protocol'] = float(val) if isinstance(val, (int, float)) else str(val)
               break
  
   return details
def load_intelligence():
   """Load multiclass model and XAI artifacts."""
   print("="*70)
   print("[INTELLIGENCE CORE] Loading Model & XAI Artifacts")
   print("="*70)
  
   # Core preprocessing
   feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, "final_feature_names.pkl"))
   scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "final_preprocessor.pkl"))
  
   # Encoder (Multiclass only)
   label_encoder_multi = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder_multiclass.pkl"))
  
   print(f"  ✓ Feature Map: {len(feature_names)} columns")
   print(f"  ✓ Scaler: StandardScaler")
  
   # Multiclass Model (Direct Inference)
   multi_model = TabNetClassifier()
   multi_model.load_model(os.path.join(ARTIFACTS_DIR, "multiclass_model.zip"))
   print("  ✓ Multiclass Model (Direct Expert)")
  
   # XAI artifacts
   xai = {}
   xai_files = {
       'statistics': 'feature_statistics.json',
       'descriptions': 'feature_descriptions.json',
       'importance': 'global_importance.json'
   }
  
   for key, filename in xai_files.items():
       path = os.path.join(ARTIFACTS_DIR, filename)
       if os.path.exists(path):
           with open(path, 'r') as f:
               xai[key] = json.load(f)
           print(f"  ✓ {filename}")
       else:
           xai[key] = {}
           print(f"  ⚠ {filename} (missing)")
  
   return {
       'multi_model': multi_model,
       'scaler': scaler,
       'label_encoder_multi': label_encoder_multi,
       'feature_names': feature_names,
       'xai': xai
   }
def categorize_files(data_dir):
   """Categorize CSV files by attack type."""
   print(f"\n[DATA LOADER] Scanning {data_dir}...")
  
   csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
  
   if not csv_files:
       return None
  
   categories = {
       'Benign': [],
       'DDoS': [],
       'DoS': [],
       'MQTT': [],
       'Recon': [],
       'Spoofing': []
   }
  
   for f in csv_files:
       filename = os.path.basename(f)
       label = get_ground_truth(filename)
      
       if label in categories:
           categories[label].append(f)
  
   for label, files in categories.items():
       print(f"  - {label}: {len(files)} files")
  
   return categories
def load_packets(file_list, label, max_packets=100):
   """Load packets from files."""
   packets = []
  
   for f in file_list[:2]:  # Limit files
       try:
           df = pd.read_csv(f, nrows=max_packets)
           for _, row in df.iterrows():
               packets.append({'data': row, 'label': label})
       except Exception as e:
           print(f"  [Warning] Failed to load {f}: {e}")
  
   return packets
def build_rhythmic_playlist(categories):
   """
   Build rhythmic playlist: 10 Benign → 5 Attack (for each type).
   Pattern: Benign→DDoS→Benign→DoS→Benign→MQTT→Benign→Recon→Benign→Spoofing
   """
   print("\n[PLAYLIST] Building Rhythmic Pattern...")
   print("  Pattern: 10 Benign → 5 Attack (per type)")
  
   # Load packets
   packet_pools = {}
   for label, files in categories.items():
       if files:
           packet_pools[label] = load_packets(files, label)
           print(f"  ✓ Loaded {len(packet_pools[label])} {label} packets")
  
   # Build playlist
   playlist = []
   attack_order = ['DDoS', 'DoS', 'MQTT', 'Recon', 'Spoofing']
  
   for attack_type in attack_order:
       if attack_type not in packet_pools or len(packet_pools[attack_type]) == 0:
           print(f"  ⚠ Skipping {attack_type} (no data)")
           continue
      
       # Add 10 benign
       benign_burst = random.sample(
           packet_pools['Benign'],
           min(10, len(packet_pools['Benign']))
       )
       playlist.extend(benign_burst)
      
       # Add 5 attack
       attack_burst = random.sample(
           packet_pools[attack_type],
           min(5, len(packet_pools[attack_type]))
       )
       playlist.extend(attack_burst)
      
       print(f"  + 10 Benign → 5 {attack_type}")
  
   # End with 10 benign packets
   final_benign = random.sample(
       packet_pools['Benign'],
       min(10, len(packet_pools['Benign']))
   )
   playlist.extend(final_benign)
   print(f"  + 10 Benign (final cooldown)")
  
   print(f"\n[PLAYLIST] Total: {len(playlist)} packets (ending with 10 benign)")
  
   return playlist
def preprocess_packet(row, scaler, feature_names):
   """Preprocess packet with exact alignment."""
   if isinstance(row, pd.Series):
       df = pd.DataFrame([row])
   else:
       df = pd.DataFrame([row])
  
   # Drop leakage columns
   cols_to_drop = ['source_file', 'basename', 'SubType', 'Flow ID',
                   'Src IP', 'Dst IP', 'Source IP', 'Destination IP',
                   'Timestamp', 'Label', 'Protocol', 'Flow Duration',
                   'Source Port', 'Destination Port', 'Time', 'MAC']
  
   df.columns = df.columns.str.strip()
   df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
  
   # Convert objects to numeric
   for col in df.columns:
       if df[col].dtype == 'object':
           df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
  
   df = df.astype(float)
  
   # CRITICAL: Sanitize infinity and NaN values
   df.replace([np.inf, -np.inf], 0, inplace=True)
   df.fillna(0, inplace=True)
  
   # CRITICAL: Reindex to training columns
   df = df.reindex(columns=feature_names, fill_value=0.0)
  
   # Scale
   X_scaled = scaler.transform(df.values)
   X_original = df.values
  
   return X_scaled, X_original
def direct_predict(X_scaled, intelligence):
   """Direct multiclass prediction with confidence threshold."""
   multi_model = intelligence['multi_model']
   label_encoder_multi = intelligence['label_encoder_multi']
  
   # Direct Multiclass Prediction
   probs = multi_model.predict_proba(X_scaled)[0]
   pred_idx = np.argmax(probs)
   pred_label = label_encoder_multi.inverse_transform([pred_idx])[0]
   confidence = float(np.max(probs))
  
   # Confidence Threshold: Suppress low-confidence attacks
   suppressed = False
   if pred_label != "Benign" and confidence < 0.60:
       pred_label = "Benign"
       suppressed = True
  
   return {
       'final': pred_label,
       'confidence': confidence,
       'source': 'multiclass_direct',
       'suppressed': suppressed
   }
def generate_dual_xai(packet_raw, pred_label, intelligence):
   """Generate Doctor and Admin XAI views."""
   xai = intelligence['xai']
   feature_names = intelligence['feature_names']
  
   statistics = xai.get('statistics', {})
   descriptions = xai.get('descriptions', {})
   importance = xai.get('importance', {})
  
   benign_stats = statistics.get('Benign', {})
  
   if not benign_stats or 'mean' not in benign_stats:
       return {
           'doctor': "Unusual network behavior detected.",
           'admin': "Insufficient baseline data."
       }
  
   # Find most anomalous feature
   max_deviation = 0
   critical_feature = None
   critical_value = 0
   normal_value = 0
   z_score = 0
  
   top_features = list(importance.keys())[:5] if importance else feature_names[:5]
  
   for i, feat_name in enumerate(feature_names):
       if i >= len(packet_raw[0]) or feat_name not in top_features:
           continue
      
       current_val = float(packet_raw[0][i])
      
       if i < len(benign_stats['mean']):
           mean = benign_stats['mean'][i]
           std = benign_stats['std'][i]
          
           if mean != 0:
               deviation = abs(current_val - mean) / (abs(mean) + 1e-6)
              
               if deviation > max_deviation:
                   max_deviation = deviation
                   critical_feature = feat_name
                   critical_value = current_val
                   normal_value = mean
                   if std > 0:
                       z_score = (current_val - mean) / std
  
   if critical_feature is None:
       return {
           'doctor': "Unusual network behavior detected.",
           'admin': "No significant anomalies identified."
       }
  
   # Doctor View (Simplified)
   human_name = descriptions.get(critical_feature, critical_feature.replace('_', ' ').title())
  
   if max_deviation > 5:
       severity = "ekstremdir"
       factor = int(max_deviation)
   elif max_deviation > 2:
       severity = "yüksektir"
       factor = int(max_deviation)
   else:
       severity = "düzensizdir"
       factor = 2
  
   doctor_msg = f"{human_name} son derece {severity}. Normal hasta izleme trafiğine kıyasla {factor}x kat daha fazladır."
  
   # Admin View (Technical)
   rank = list(importance.keys()).index(critical_feature) + 1 if critical_feature in importance else None
   rank_str = f"Rank: #{rank} | " if rank else ""
  
   admin_msg = f"Feature '{critical_feature}' | {rank_str}Z-Score: {z_score:+.1f} | Value: {critical_value:.1f}"
  
   return {
       'doctor': doctor_msg,
       'admin': admin_msg
   }
def display_packet_result(idx, ground_truth, result, xai=None):
   """Display packet result with optional XAI."""
   pred_label = result['final']
   confidence = result['confidence']
   suppressed = result.get('suppressed', False)
  
   if pred_label == "Benign":
       marker = "🟢" if ground_truth == "Benign" else "⚠️"
       suffix = " - Safe"
       if suppressed:
           suffix = " - Safe (Low conf attack suppressed)"
       print(f"  [{idx:3d}] {marker} [ACTUAL: {ground_truth}] -> [PRED: {pred_label} ({confidence:.0%})]{suffix}")
   else:
       marker = "🚨"
       print(f"  [{idx:3d}] {marker} [ACTUAL: {ground_truth}] -> [PRED: {pred_label} ({confidence:.0%})] - ATTACK DETECTED!")
      
       if xai:
           print(f"        {'-'*60}")
           print(f"        👨‍⚕️ DOCTOR: \"{xai['doctor']}\"")
           print(f"        🛡️ ADMIN:  {xai['admin']}")
           print(f"        {'-'*60}")
def simulate(target_ip, send_alerts=True):
   """Main unified simulation."""
   print("\n" + "="*70)
   print("UNIFIED IoMT IDS SIMULATION - Grand Finale")
   print("="*70)
  
   # Load intelligence
   intelligence = load_intelligence()
  
   # Find data
   data_dir = DATA_DIR_PRIMARY if os.path.exists(DATA_DIR_PRIMARY) else DATA_DIR_FALLBACK
  
   if not os.path.exists(data_dir):
       print(f"\n[ERROR] Data directory not found: {data_dir}")
       return
  
   # Categorize files
   categories = categorize_files(data_dir)
  
   if not categories or not categories.get('Benign'):
       print("\n[ERROR] No valid data files found.")
       return
  
   # Build playlist
   playlist = build_rhythmic_playlist(categories)
  
   # Simulate
   print("\n" + "="*70)
   print("[SIMULATION] Starting Rhythmic Pattern Demo")
   print(f"[SIMULATION] Device IP: {target_ip}")
   print(f"[SIMULATION] Alerts: {'ENABLED' if send_alerts else 'DISABLED'}")
   print("="*70)
  
   stats = {
       'total': 0,
       'correct': 0,
       'attacks_detected': 0,
       'false_positives': 0,
       'false_negatives': 0
   }
  
   for idx, packet_info in enumerate(playlist, 1):
       try:
           row = packet_info['data']
           ground_truth = packet_info['label']
          
           # Preprocess
           X_scaled, X_raw = preprocess_packet(
               row,
               intelligence['scaler'],
               intelligence['feature_names']
           )
          
           # Predict (Direct Multiclass)
           result = direct_predict(X_scaled, intelligence)
          
           stats['total'] += 1
          
           if ground_truth == result['final']:
               stats['correct'] += 1
           elif ground_truth == "Benign" and result['final'] != "Benign":
               stats['false_positives'] += 1
           elif ground_truth != "Benign" and result['final'] == "Benign":
               stats['false_negatives'] += 1
          
           # Generate XAI for attacks
           xai_data = None
           xai_payload = None
           if result['final'] != "Benign":
               stats['attacks_detected'] += 1
               xai_data = generate_dual_xai(X_raw, result['final'], intelligence)
               xai_payload = {
                   "tech_view": xai_data['doctor'],
                   "admin_view": xai_data['admin']
               }
          
           # Extract traffic details for mobile dropdown
           traffic_details = extract_traffic_details(row)
          
           # Send EVERY packet to backend (Always-On Sync)
           if send_alerts:
               try:
                   payload = {
                       "device_ip": target_ip,
                       "prediction": {
                           "is_attack": (result['final'] != "Benign"),
                           "probability": result['confidence']
                       },
                       "attack_type": result['final'],
                       "traffic_data": traffic_details,
                       "xai": xai_payload
                   }
                   requests.post(BACKEND_URL, json=payload, timeout=2)
                   print(".", end="", flush=True)  # Heartbeat success
               except:
                   print("!", end="", flush=True)  # Heartbeat failure
          
           # Display
           print()  # Newline after heartbeat
           display_packet_result(idx, ground_truth, result, xai_data)
          
           # Pacing
           if result['final'] != "Benign":
               time.sleep(0.5)
           else:
               time.sleep(0.02)
          
       except Exception as e:
           print(f"  [{idx:3d}] ❌ [ERROR] {str(e)[:50]}")
  
   # Summary
   print("\n" + "="*70)
   print("[SUMMARY] Unified Simulation Complete")
   print("="*70)
   total = max(1, stats['total'])
   print(f"  Total Packets:       {stats['total']}")
   print(f"  ✅ Correct:           {stats['correct']} ({stats['correct']/total*100:.1f}%)")
   print(f"  🚨 Attacks Detected:  {stats['attacks_detected']}")
   print(f"  ❌ False Positives:   {stats['false_positives']}")
   print(f"  ⚠️ False Negatives:   {stats['false_negatives']}")
  
   print("\n" + "="*70)
   print("Dual-Persona XAI demonstrated for all detected attacks.")
   print("="*70)
if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Unified IoMT IDS Simulation")
   parser.add_argument("--ip", type=str, required=True, help="Target Device IP")
   parser.add_argument("--no-alerts", action="store_true", help="Disable backend alerts")
   args = parser.parse_args()
  
   try:
       simulate(args.ip, send_alerts=not args.no_alerts)
   except KeyboardInterrupt:
       print("\n\n[INTERRUPTED] Simulation stopped by user.")
   except Exception as e:
       print(f"\n\n[FATAL ERROR] {e}")
       import traceback
       traceback.print_exc()

