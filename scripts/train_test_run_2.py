import sys
import os

# Add the project root to the python path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.loader import DataLoader
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import zipfile

def train_test_run():
    print("===============================================================")
    print("   IDS Project - Production Training Run (70% Data)   ")
    print("===============================================================")

    # Define paths and load data
    base_path = os.path.abspath("data/raw/WiFi_and_MQTT")
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[ERROR] Train or Test directory not found in {base_path}")
        sys.exit(1)

    # 1. Load Data Separately
    print("\n[STEP 1] Loading Train and Test Data Separately...")
    
    # Train Loader
    print(f"  - Loading Training Data from: {train_path}")
    train_loader = DataLoader(train_path)
    df_train_spark = train_loader.load_data()
    
    # Test Loader
    print(f"  - Loading Test Data from: {test_path}")
    test_loader = DataLoader(test_path)
    df_test_spark = test_loader.load_data()
    
    if df_train_spark is None or df_test_spark is None:
        print("[ERROR] Failed to load data.")
        sys.exit(1)

    # Preprocess in Spark (Handle Infinity/NaN)
    print("\n[STEP 2] Preprocessing in Spark...")
    df_train_spark = train_loader.preprocess(df_train_spark)
    df_test_spark = test_loader.preprocess(df_test_spark)

    # NEW: Interaction Features (DoS vs DDoS without IPs)
    print("\n[STEP 2.5] Calculating Interaction Features (Rate * Dynamics)...")
    
    # Logic: Create interaction features to capture traffic dynamics
    # Rate_Variance = Rate * Variance (Amplifies chaotic nature)
    # Rate_IAT = Rate * IAT (Volume vs Timing)
    # Size_Rate_Ratio = Tot size / (Rate + 1e-5) (Packet size efficiency)
    
    from pyspark.sql.functions import col, log1p
    
    def add_interaction_features(df):
        # Interaction Features
        df = df.withColumn("Rate_Variance", col("Rate") * col("Variance"))
        df = df.withColumn("Rate_IAT", col("Rate") * col("IAT"))
        df = df.withColumn("Size_Rate_Ratio", col("Tot size") / (col("Rate") + 1e-5))
        
        # NEW: Logarithmic Features (Compress Scale)
        df = df.withColumn("Log_Rate", log1p(col("Rate")))
        df = df.withColumn("Log_Variance", log1p(col("Variance")))
        df = df.withColumn("Log_Tot_Size", log1p(col("Tot size")))
        df = df.withColumn("Log_Duration", log1p(col("Number") * col("IAT"))) # Proxy for duration
        
        # NEW: Ratio Features (Fusion)
        df = df.withColumn("Size_Per_Packet", col("Tot size") / (col("Number") + 1))
        
        # Handle NaNs and Infs created by these operations
        # Replace Inf with NaN, then fill NaN with 0
        from pyspark.sql.functions import when, lit
        new_cols = ["Rate_Variance", "Rate_IAT", "Size_Rate_Ratio", "Log_Rate", "Log_Variance", "Log_Tot_Size", "Log_Duration", "Size_Per_Packet"]
        
        for c_name in new_cols:
            df = df.withColumn(c_name, 
                               when(col(c_name) == float("inf"), 0.0)
                               .when(col(c_name) == float("-inf"), 0.0)
                               .when(col(c_name).isNull(), 0.0)
                               .when(col(c_name).isNaN(), 0.0)
                               .otherwise(col(c_name)))
        
        return df

    df_train_spark = add_interaction_features(df_train_spark)
    df_test_spark = add_interaction_features(df_test_spark)

    # Merge DataFrames
    print("\n[STEP 3] Merging Train and Test Data for Strict Split...")
    df_full_spark = df_train_spark.union(df_test_spark)
    
    # Sample 70% for Production
    fraction = 0.7
    print(f"\n[STEP 3.1] Sampling {fraction*100}% of data for Production Run...")
    df_sampled = df_full_spark.sample(withReplacement=False, fraction=fraction, seed=42)
    
    print(f"[INFO] Converting to Pandas...")
    df_full = df_sampled.toPandas()
    print(f"  - Full Sample Shape: {df_full.shape}")

    # 2. Preprocessing & Anti-Leakage
    print("\n[STEP 4] Preprocessing for Training...")
    
    # Clean Data
    print("[INFO] Cleaning Data (Removing invalid labels)...")
    valid_labels = df_full['Label'].notna() & (df_full['Label'] != "") & (df_full['Label'] != "0")
    df_full = df_full[valid_labels]
    
    cols_to_drop = ['source_file', 'basename', 'SubType', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    df_full = df_full.drop(columns=[c for c in cols_to_drop if c in df_full.columns])
    
    # Encoding
    object_cols = df_full.select_dtypes(include=['object']).columns.tolist()
    print(f"[INFO] Categorical columns to encode: {object_cols}")
    
    label_encoders = {}
    for col in object_cols:
        le = LabelEncoder()
        df_full[col] = le.fit_transform(df_full[col].astype(str))
        label_encoders[col] = le

    # Define X and y
    target_col = 'Label'
    X = df_full.drop(columns=[target_col])
    y = df_full[target_col]
    
    feature_names = X.columns.tolist()
    X = X.values
    y = y.values

    # 3. Strict 60/20/20 Split
    print("\n[STEP 5] Performing Strict 60/20/20 Stratified Split...")
    
    # First split: Train (60%) vs Temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    
    # Second split: Val (50% of Temp -> 20% total) vs Test (50% of Temp -> 20% total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    print(f"  - Train shape: {X_train.shape} (60%)")
    print(f"  - Val shape:   {X_val.shape} (20%)")
    print(f"  - Test shape:  {X_test.shape} (20%)")

    # 4. Oversampling Benign (Train Set ONLY)
    print("\n[STEP 6] Oversampling Benign Class in Training Set (Pandas)...")
    # Reconstruct Train DF for easy manipulation
    df_train_processed = pd.DataFrame(X_train, columns=feature_names)
    df_train_processed['Label'] = y_train
    
    # Identify Benign Class ID
    benign_label = "Benign"
    if benign_label in label_encoders[target_col].classes_:
        benign_id = label_encoders[target_col].transform([benign_label])[0]
        
        # Filter Benign
        df_benign = df_train_processed[df_train_processed['Label'] == benign_id]
        
        if not df_benign.empty:
            print(f"  - Found {len(df_benign)} Benign samples in Train.")
            # Duplicate 20x
            df_benign_oversampled = pd.concat([df_benign] * 20, ignore_index=True)
            
            # Concat and Shuffle
            df_train_processed = pd.concat([df_train_processed, df_benign_oversampled], ignore_index=True)
            df_train_processed = df_train_processed.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"  - New Train shape after Oversampling: {df_train_processed.shape}")
            
            # Update X_train, y_train
            X_train = df_train_processed.drop(columns=['Label']).values
            y_train = df_train_processed['Label'].values
        else:
            print("  - [WARNING] No Benign samples found in Training set to oversample!")
    else:
        print("  - [WARNING] 'Benign' class not found in Label Encoder!")

    # 5. Feature Selection (Mutual Information)
    print("\n[STEP 7] Feature Selection using Mutual Information...")
    from sklearn.feature_selection import mutual_info_classif
    
    # Use a sample of Train for speed if it's too large
    mi_sample_size = 50000
    if X_train.shape[0] > mi_sample_size:
        print(f"  - [OPTIMIZATION] Sampling {mi_sample_size} rows for MI calculation (Full dataset too large)...")
        # Random indices
        mi_indices = np.random.choice(X_train.shape[0], mi_sample_size, replace=False)
        X_mi = X_train[mi_indices]
        y_mi = y_train[mi_indices]
    else:
        X_mi = X_train
        y_mi = y_train

    print("  - Calculating MI scores...")
    mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
    
    # Create DataFrame for visualization
    mi_df = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})
    mi_df = mi_df.sort_values(by='MI_Score', ascending=False)
    
    print("\n  - Top 10 Features:")
    print(mi_df.head(10))
    print("\n  - Bottom 10 Features:")
    print(mi_df.tail(10))
    
    # Drop features with MI < 0.01
    threshold = 0.01
    drop_features = mi_df[mi_df['MI_Score'] < threshold]['Feature'].tolist()
    
    if drop_features:
        print(f"\n  - Dropping {len(drop_features)} features with MI < {threshold}:")
        print(f"    {drop_features}")
        
        # Get indices to keep
        keep_indices = [i for i, f in enumerate(feature_names) if f not in drop_features]
        keep_features = [f for f in feature_names if f not in drop_features]
        
        # Filter X arrays
        X_train = X_train[:, keep_indices]
        X_val = X_val[:, keep_indices]
        X_test = X_test[:, keep_indices]
        
        # Update feature names
        feature_names = keep_features
        print(f"  - New Feature Count: {len(feature_names)}")
    else:
        print("  - No features dropped.")

    # 6. Class Weights
    print("\n[STEP 8] Computing Class Weights...")
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    # Soften weights: sqrt to avoid extreme penalties
    weights = np.sqrt(weights)
    class_weights = dict(zip(classes, weights))
    
    print("  - Class Weights:")
    for cls, weight in class_weights.items():
        cls_name = label_encoders[target_col].inverse_transform([cls])[0]
        print(f"    {cls_name}: {weight:.4f}")

    # 5. Training
    print("\n[STEP 7] Training TabNetClassifier with Class Weights...")
    
    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        verbose=1
    )
    
    # FIX 2: Removed redundant fit call. Only calling once with weights.
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_name=['train', 'valid'],
        eval_metric=['accuracy'],
        max_epochs=50, # Production: 50 epochs
        patience=15,
        batch_size=16384, # Production: 16k batch size
        virtual_batch_size=1024,
        num_workers=0,
        drop_last=False,
        weights=class_weights 
    )

    # 6. Evaluation
    print("\n[STEP 8] Evaluation on Test Set...")
    
    y_pred = clf.predict(X_test)
    target_names = [str(c) for c in label_encoders[target_col].classes_]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 7. Confusion Matrix Plot
    print("\n[STEP 9] Generating Confusion Matrix...")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    artifacts_dir = "artifacts"
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)
        
    save_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"[INFO] Confusion Matrix saved to: {save_path}")

    # FIX 4: Save Model and Encoders (Moved before SHAP)
    print("\n[STEP 10] Saving Model and Artifacts...")
    
    # Save TabNet Model
    model_save_path = os.path.join(artifacts_dir, "tabnet_model")
    clf.save_model(model_save_path)
    print(f"[INFO] Model saved to: {model_save_path}.zip")
    
    # Save Label Encoders and Feature Names
    joblib.dump(label_encoders, os.path.join(artifacts_dir, "label_encoders.pkl"))
    joblib.dump(feature_names, os.path.join(artifacts_dir, "feature_names.pkl"))
    print(f"[INFO] Encoders and Feature Names saved to artifacts/")

    # FIX 3: SHAP Visualization
    print("\n[STEP 11] Generating SHAP Explanation (Sampled)...")
    
    # Sample 100 random instances from Test set for SHAP
    # Ensure we don't sample more than available
    n_shap_samples = min(100, X_test.shape[0])
    indices = np.random.choice(X_test.shape[0], n_shap_samples, replace=False)
    X_shap = X_test[indices]
    
    print(f"  - Calculating SHAP values for {n_shap_samples} samples...")
    
    # Using KernelExplainer with predict_proba
    # We need a background dataset. Using a small summary of train data (e.g. kmeans) or just a small sample of train.
    # For speed, let's use a small sample of X_train as background (e.g. 50 samples).
    background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
    
    explainer = shap.KernelExplainer(clf.predict_proba, background)
    shap_values = explainer.shap_values(X_shap)
    
    # Plotting
    plt.figure()
    
    print("  - Saving SHAP Summary Plot...")
    shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
    shap_save_path = os.path.join(artifacts_dir, "shap_summary.png")
    plt.savefig(shap_save_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] SHAP Summary Plot saved to: {shap_save_path}")

    print("\n===============================================================")
    print("   Final Run Complete   ")
    print("===============================================================")

if __name__ == "__main__":
    train_test_run()
