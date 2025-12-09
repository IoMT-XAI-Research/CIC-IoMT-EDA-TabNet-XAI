import os
import sys
import joblib
import pandas as pd
import numpy as np
import shap
from pytorch_tabnet.tab_model import TabNetClassifier
from .. import schemas

# Helper for safe scalar conversion
def safe_scalar(val):
    """Safely convert numpy arrays/scalars to Python float."""
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

class AIEngine:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.clf = None
        self.label_encoders = None
        self.feature_names = None
        self._is_loaded = False

    def load_model(self):
        print("[INFO] Loading AI Model Artifacts...")
        try:
            # Load Model
            model_path = os.path.join(self.artifacts_dir, "tabnet_model.zip")
            if not os.path.exists(model_path):
                # Fallback to absolute path if running from src/backend
                # Assuming standard project structure
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
                model_path = os.path.join(base_dir, "artifacts", "tabnet_model.zip")
                self.artifacts_dir = os.path.join(base_dir, "artifacts")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")

            self.clf = TabNetClassifier()
            self.clf.load_model(model_path)

            # Load Encoders
            encoders_path = os.path.join(self.artifacts_dir, "label_encoders.pkl")
            self.label_encoders = joblib.load(encoders_path)

            # Load Feature Names
            features_path = os.path.join(self.artifacts_dir, "feature_names.pkl")
            self.feature_names = joblib.load(features_path)

            self._is_loaded = True
            print("[INFO] AI Model Loaded Successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load AI model: {e}")
            self._is_loaded = False

    def preprocess_features(self, features: dict) -> np.ndarray:
        # Convert dict to DataFrame
        row = pd.DataFrame([features])
        
        # Consistent preprocessing logic
        cols_to_drop = ['source_file', 'basename', 'SubType', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label', 'Protocol']
        
        for col in cols_to_drop:
            if col in row.columns:
                row = row.drop(columns=[col])
        
        # Ensure all feature columns exist
        for col in self.feature_names:
            if col not in row.columns:
                row[col] = 0
                
        # Reorder
        row = row[self.feature_names]
        
        # Encode Categorical
        for col, le in self.label_encoders.items():
            if col in row.columns and col != 'Label':
                val = str(row[col].iloc[0])
                if val in le.classes_:
                    row[col] = le.transform([val])[0]
                else:
                    row[col] = 0
                    
        # Numeric conversion
        row = row.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return row.values.reshape(1, -1)

    def generate_shap_explanation(self, X_sample, pred_idx):
        # Background for SHAP (zeros)
        background = np.zeros((1, X_sample.shape[1]))
        explainer = shap.KernelExplainer(self.clf.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        
        vals = None
        if isinstance(shap_values, list):
            if pred_idx < len(shap_values):
                vals = shap_values[pred_idx][0]
            else:
                 vals = shap_values[0][0] if len(shap_values) == 1 else shap_values[-1][0]
        else:
            vals = shap_values[0]

        feature_importance = []
        for name, val in zip(self.feature_names, vals):
            scalar_val = safe_scalar(val)
            feature_importance.append({
                "name": name,
                "percentage": float(abs(scalar_val)),
                "value_desc": f"{scalar_val:.4f}"
            })
        feature_importance.sort(key=lambda x: x['percentage'], reverse=True)

        # Force Plot Data
        ev = explainer.expected_value
        base_value = 0.0
        if isinstance(ev, list):
            base_value = safe_scalar(ev[pred_idx]) if pred_idx < len(ev) else safe_scalar(ev[-1] if ev else 0.0)
        elif isinstance(ev, np.ndarray):
            base_value = safe_scalar(ev.flatten()[pred_idx]) if ev.size > 1 and pred_idx < ev.size else safe_scalar(ev)
        else:
            base_value = safe_scalar(ev)

        sum_shap = np.sum(vals)
        final_value = safe_scalar(sum_shap) + base_value

        xai_plot = schemas.XAIForcePlot(
            base_value=base_value,
            final_value=final_value,
            features=[
                {"name": n, "value": safe_scalar(v)} 
                for n, v in zip(self.feature_names, X_sample[0])
            ]
        )
        
        return xai_plot, feature_importance[:5]

    def predict(self, features: dict) -> schemas.AnalysisResponse:
        if not self._is_loaded:
            # Try loading if not loaded (e.g. first request)
            self.load_model()
            if not self._is_loaded:
                # Mock response if model fails to load
                return self._mock_response()

        try:
            X = self.preprocess_features(features)
            
            probs = self.clf.predict_proba(X)[0]
            pred_idx = np.argmax(probs)
            pred_label = self.label_encoders['Label'].inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            
            # Determine status
            status = "ATTACK" if (pred_label != "Benign" and confidence > 0.5) else "SAFE"
            
            # XAI (Only if Attack or high risk, but let's do it always for demo or just for attack?)
            # Doing it always is expensive. Let's do it if status is ATTACK.
            xai_plot = None
            feature_importance = []
            
            if status == "ATTACK":
                xai_plot, feature_importance = self.generate_shap_explanation(X, pred_idx)
            else:
                # Minimal placeholder for SAFE
                xai_plot = schemas.XAIForcePlot(base_value=0, final_value=0, features=[])
                feature_importance = []

            return schemas.AnalysisResponse(
                device_status=status,
                risk_score=confidence if status == "ATTACK" else (1.0 - confidence), # Risk is low if benign high conf
                summary_text=f"Detected {pred_label} with {confidence:.2f} confidence." if status == "ATTACK" else "Normal traffic detected.",
                xai_force_plot=xai_plot,
                feature_importance_list=feature_importance
            )
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return self._mock_response()

    def _mock_response(self):
        return schemas.AnalysisResponse(
            device_status="SAFE",
            risk_score=0.0,
            summary_text="AI Engine unavailable. Returning safe mock.",
            xai_force_plot=schemas.XAIForcePlot(base_value=0, final_value=0, features=[]),
            feature_importance_list=[]
        )

# Global Instance
ai_engine = AIEngine()
