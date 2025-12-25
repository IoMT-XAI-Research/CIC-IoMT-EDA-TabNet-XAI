import sys
import os
import time
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.loader import DataLoader
from pyspark.sql.functions import col, when, lit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import mutual_info_classif
from pytorch_tabnet.tab_model import TabNetClassifier

# XAI Libraries
try:
    import shap
    from alibi.explainers import AnchorTabular
    import dice_ml
except ImportError:
    print("[WARNING] XAI libraries (shap, alibi, dice_ml) not found. XAI steps will be skipped.")
    shap = None
    AnchorTabular = None
    dice_ml = None

class DualTrack_IDS_Pipeline:
    def __init__(self, base_path, artifacts_dir="artifacts"):
        self.base_path = base_path
        self.artifacts_dir = artifacts_dir
        if not os.path.exists(artifacts_dir):
            os.makedirs(artifacts_dir)
            
        self.preprocessor = None
        self.feature_names = None
        self.label_encoder_multi = None
        
    def load_and_clean(self, fraction=0.7):
        print("\n[STEP 1] Loading and Cleaning Data...")
        loader = DataLoader(self.base_path)
        df_spark = loader.load_data()
        
        if df_spark is None:
            sys.exit(1)
            
        # Sampling
        print(f"  - Sampling {fraction*100}% of data...")
        df_sampled = df_spark.sample(withReplacement=False, fraction=fraction, seed=42)
        
        # Convert to Pandas
        df = df_sampled.toPandas()
        
        # Aggressive Anti-Leakage
        print("  - [Anti-Leakage] Dropping Forbidden Columns...")
        df.columns = df.columns.str.strip()
        
        # 1. Drop Metadata & Leakage Columns
        # Protocol is dropped because it contains label info (e.g. 'MQTT', 'Benign')
        leakage_cols = ['Source IP', 'Destination IP', 'Timestamp', 'Flow ID', 'Source Port', 'Destination Port', 'Unnamed: 0', 
                        'source_file', 'basename', 'SubType', 'Src IP', 'Dst IP', 'MAC', 'Prob', 'Flow Duration', 'Protocol']
        
        existing_drop_cols = [c for c in leakage_cols if c in df.columns]
        df = df.drop(columns=existing_drop_cols)
        
        # Fill NaNs
        df = df.fillna(0)
        
        return df

    def preprocess(self, df):
        print("\n[STEP 2] Preprocessing (Log-Transform + Scaling)...")
        X = df.drop(columns=['Label'])
        y = df['Label']
        
        # 2. Feature Restoration: Rate
        # Ensure Rate is treated as numeric and log-transformed
        # Also apply log to other skewed features if present
        skewed_features = ['Rate', 'Variance', 'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Kurtosis', 'Skewness']
        
        # Identify types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"  - Numeric: {len(numeric_features)}")
        print(f"  - Categorical: {len(categorical_features)}")
        
        # Safety Check for High Cardinality
        for col in categorical_features:
            unique_count = X[col].nunique()
            if unique_count > 50:
                print(f"  - [WARNING] Dropping high-cardinality column: {col} ({unique_count})")
                X = X.drop(columns=[col])
                categorical_features.remove(col)
        
        # Define Log Transformer
        log_transformer = FunctionTransformer(np.log1p, validate=True)
        
        # Split numeric into skewed and normal (simplified: treat all numeric as candidates for scaling, 
        # but apply log to specific ones if they exist)
        # Actually, let's just use StandardScaler for all numeric, but pre-apply log to 'Rate' and others manually or via transformer.
        # User asked for "Logarithmic Transformation (log1p) or RobustScaler".
        # Let's apply log1p to 'Rate' specifically if it exists.
        
        for col in skewed_features:
            if col in X.columns:
                # Apply log1p directly to the dataframe before ColumnTransformer
                # This is easier than complex pipeline for now
                X[col] = np.log1p(X[col])
                print(f"  - Applied log1p to {col}")
        
        # Transform
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            verbose_feature_names_out=False
        )
        
        X_processed = self.preprocessor.fit_transform(X)
        
        # Get names
        try:
            self.feature_names = self.preprocessor.get_feature_names_out().tolist()
        except:
            self.feature_names = numeric_features + list(self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
            
        # Save feature names and preprocessor
        joblib.dump(self.feature_names, os.path.join(self.artifacts_dir, "feature_names_v4.pkl"))
        joblib.dump(self.preprocessor, os.path.join(self.artifacts_dir, "preprocessor.pkl"))
        
        return X_processed, y, self.feature_names

    def prepare_targets(self, y):
        print("\n[STEP 3] Preparing Dual-Track Targets...")
        
        # Track A: Binary (0=Benign, 1=Attack)
        y_binary = y.apply(lambda x: 0 if x == 'Benign' else 1).values
        
        # Track B: Multiclass
        # Ensure classes are: ['Benign', 'DDoS', 'DoS', 'MQTT', 'Recon', 'Spoofing']
        self.label_encoder_multi = LabelEncoder()
        y_multi = self.label_encoder_multi.fit_transform(y)
        joblib.dump(self.label_encoder_multi, os.path.join(self.artifacts_dir, "label_encoder_multi.pkl"))
        
        print(f"  - Binary Classes: {np.unique(y_binary)}")
        print(f"  - Multiclass Classes: {self.label_encoder_multi.classes_}")
        
        return y_binary, y_multi

    def train_model(self, X_train, y_train, X_val, y_val, name, is_multiclass=False):
        print(f"\n[STEP 4] Training {name}...")
        
        # Class Weights
        classes = np.unique(y_train)
        if is_multiclass:
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weights = dict(zip(classes, weights))
        else:
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weights = dict(zip(classes, weights))
            
        clf = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax',
            verbose=1
        )
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_name=['train', 'valid'],
            eval_metric=['accuracy'],
            max_epochs=50,
            patience=15,
            batch_size=16384,
            virtual_batch_size=1024,
            num_workers=0,
            drop_last=False,
            weights=class_weights
        )
        
        # Save
        clf.save_model(os.path.join(self.artifacts_dir, name))
        # Zip it manually if needed, but save_model creates zip
        print(f"  - Saved {name}")
        
        return clf

    def evaluate(self, clf, X_test, y_test, name, target_names=None):
        print(f"\n[Evaluation] {name}")
        y_pred = clf.predict(X_test)
        
        # Fix: Check if target_names is not None explicitly
        if target_names is not None:
            print(classification_report(y_test, y_pred, target_names=target_names))
        else:
            print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names if target_names is not None else "auto", 
                    yticklabels=target_names if target_names is not None else "auto")
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(os.path.join(self.artifacts_dir, f"cm_{name}.png"))
        plt.close()

    def run_xai(self, clf, X_train, X_test, feature_names):
        if not (shap and AnchorTabular and dice_ml):
            return

        print("\n[STEP 5] Running XAI Pipeline (Binary Model)...")
        
        # 1. SHAP Beeswarm
        print("  - [SHAP] Generating Beeswarm Plot...")
        # Use small background
        background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        shap_values = explainer.shap_values(X_test[:20]) 
        
        # shap_values is list of arrays (one per class). Index 1 = Attack.
        vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        plt.figure()
        shap.summary_plot(vals, X_test[:20], feature_names=feature_names, show=False)
        plt.savefig(os.path.join(self.artifacts_dir, "shap_beeswarm_v4.png"))
        plt.close()
        
        # 2. Anchors
        print("  - [Anchors] Generating Rules...")
        predict_fn = lambda x: clf.predict(x)
        explainer_anchor = AnchorTabular(predict_fn, feature_names)
        explainer_anchor.fit(X_train)
        
        attacks = np.where(clf.predict(X_test) == 1)[0]
        if len(attacks) > 0:
            # Fix: threshold=0.85, delta=0.15
            try:
                exp = explainer_anchor.explain(X_test[attacks[0]], threshold=0.85, delta=0.15)
                print(f"    > Anchor Rule: {exp.anchor}")
            except Exception as e:
                print(f"    > Anchor Error: {e}")
            
        # 3. DiCE
        print("  - [DiCE] Generating Counterfactuals...")
        # Fix: Wrap predict_proba for DiCE
        class DiCEWrapper:
            def __init__(self, model):
                self.model = model
            def predict_proba(self, x):
                data = x.values if hasattr(x, "values") else x
                return self.model.predict_proba(data)
                
        dice_model = dice_ml.Model(model=DiCEWrapper(clf), backend="sklearn")
        
        # Fix: Explicit continuous features
        d = dice_ml.Data(dataframe=pd.DataFrame(X_train, columns=feature_names).assign(Label=clf.predict(X_train)), 
                         continuous_features=feature_names, 
                         outcome_name='Label')
                         
        exp_dice = dice_ml.Dice(d, dice_model, method="random")
        
        if len(attacks) > 0:
            q = pd.DataFrame([X_test[attacks[0]]], columns=feature_names)
            try:
                cf = exp_dice.generate_counterfactuals(q, total_CFs=1, desired_class=0)
                print("    > Counterfactual generated.")
                # Save to file instead of visualize to avoid display issues
                cf_json = cf.to_json()
                with open(os.path.join(self.artifacts_dir, "dice_cf.json"), "w") as f:
                    f.write(cf_json)
            except Exception as e:
                print(f"    > DiCE Error: {e}")

    def run(self):
        # 1. Load
        df = self.load_and_clean(fraction=0.7)
        
        # 2. Preprocess
        X, y, feature_names = self.preprocess(df)
        
        # 3. Targets
        y_binary, y_multi = self.prepare_targets(y)
        
        # 4. Split
        X_train, X_test, y_b_train, y_b_test, y_m_train, y_m_test = train_test_split(
            X, y_binary, y_multi, test_size=0.3, stratify=y_multi, random_state=42
        )
        X_val, X_test, y_b_val, y_b_test, y_m_val, y_m_test = train_test_split(
            X_test, y_b_test, y_m_test, test_size=0.5, stratify=y_m_test, random_state=42
        )
        
        # 5. Track A: Binary
        clf_binary = self.train_model(X_train, y_b_train, X_val, y_b_val, "clf_binary", is_multiclass=False)
        self.evaluate(clf_binary, X_test, y_b_test, "clf_binary", target_names=['Benign', 'Attack'])
        
        # 6. Track B: Multiclass
        clf_multi = self.train_model(X_train, y_m_train, X_val, y_m_val, "clf_multiclass", is_multiclass=True)
        self.evaluate(clf_multi, X_test, y_m_test, "clf_multiclass", target_names=self.label_encoder_multi.classes_)
        
        # 7. XAI (Binary)
        self.run_xai(clf_binary, X_train, X_test, feature_names)
        
        print("\n[DONE] Dual-Track Pipeline Complete.")

if __name__ == "__main__":
    pipeline = DualTrack_IDS_Pipeline(base_path="data/raw/WiFi_and_MQTT")
    pipeline.run()
