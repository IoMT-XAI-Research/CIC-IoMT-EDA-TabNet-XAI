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

def train_test_run():
    print("===============================================================")
    print("   IDS Project - Final Training Run (TabNet + SHAP)   ")
    print("===============================================================")

    # Define paths
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

    # Sample 10%
    fraction = 0.10
    print(f"\n[STEP 3] Sampling {fraction*100}% of data...")
    
    df_train_sampled = df_train_spark.sample(withReplacement=False, fraction=fraction, seed=42)
    df_test_sampled = df_test_spark.sample(withReplacement=False, fraction=fraction, seed=42)
    
    print(f"[INFO] Converting to Pandas...")
    df_train = df_train_sampled.toPandas()
    df_test = df_test_sampled.toPandas()
    
    print(f"  - Train Sample Shape: {df_train.shape}")
    print(f"  - Test Sample Shape:  {df_test.shape}")

    # 2. Preprocessing & Anti-Leakage
    print("\n[STEP 4] Preprocessing for Training...")
    
    # FIX 1: Phantom Class (Data Cleaning)
    print("[INFO] Cleaning Data (Removing invalid labels)...")
    valid_labels = df_train['Label'].notna() & (df_train['Label'] != "") & (df_train['Label'] != "0")
    df_train = df_train[valid_labels]
    
    valid_labels_test = df_test['Label'].notna() & (df_test['Label'] != "") & (df_test['Label'] != "0")
    df_test = df_test[valid_labels_test]
    
    print(f"  - Cleaned Train Shape: {df_train.shape}")
    print(f"  - Cleaned Test Shape:  {df_test.shape}")
    
    cols_to_drop = ['source_file', 'basename', 'SubType', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
    
    # Drop columns
    df_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
    df_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])
    
    # Encoding
    object_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    print(f"[INFO] Categorical columns to encode: {object_cols}")
    
    label_encoders = {}
    for col in object_cols:
        le = LabelEncoder()
        # Fit on Train
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        
        # Transform Test - Handle unseen labels
        train_classes = set(le.classes_)
        test_values = set(df_test[col].astype(str).unique())
        diff = test_values - train_classes
        if diff:
            print(f"[WARNING] Test set has unseen labels in {col}: {diff}")
            le.fit(pd.concat([df_train[col].astype(str), df_test[col].astype(str)]))
            df_train[col] = le.transform(df_train[col].astype(str))
        
        df_test[col] = le.transform(df_test[col].astype(str))
        label_encoders[col] = le

    # Define X and y
    target_col = 'Label'
    
    X_train_full = df_train.drop(columns=[target_col]).values
    y_train_full = df_train[target_col].values
    
    X_test = df_test.drop(columns=[target_col]).values
    y_test = df_test[target_col].values
    
    feature_names = df_train.drop(columns=[target_col]).columns.tolist()

    # 3. Split Train into Train/Val (80/20)
    print("\n[STEP 5] Splitting Training Data (80/20 Train/Val)...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )
    
    print(f"  - Final Train shape: {X_train.shape}")
    print(f"  - Final Val shape:   {X_val.shape}")
    print(f"  - Test shape:        {X_test.shape}")

    # 4. Class Weights
    print("\n[STEP 6] Computing Class Weights...")
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
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
        max_epochs=50,
        patience=10,
        batch_size=1024, 
        virtual_batch_size=128,
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

    # FIX 3: SHAP Visualization
    print("\n[STEP 10] Generating SHAP Explanation (Sampled)...")
    
    # Sample 100 random instances from Test set for SHAP
    # Ensure we don't sample more than available
    n_shap_samples = min(100, X_test.shape[0])
    indices = np.random.choice(X_test.shape[0], n_shap_samples, replace=False)
    X_shap = X_test[indices]
    
    # TabNet explainability
    # TabNet has its own explainability, but user asked for SHAP.
    # We can use shap.KernelExplainer or shap.Explainer if supported.
    # TabNetClassifier has a predict_proba method which SHAP can use.
    # However, KernelExplainer is slow. 
    # TabNet is a neural network, so maybe DeepExplainer? But TabNet is complex.
    # KernelExplainer is safest for black-box.
    
    print(f"  - Calculating SHAP values for {n_shap_samples} samples...")
    
    # Using KernelExplainer with predict_proba
    # We need a background dataset. Using a small summary of train data (e.g. kmeans) or just a small sample of train.
    # For speed, let's use a small sample of X_train as background (e.g. 50 samples).
    background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
    
    explainer = shap.KernelExplainer(clf.predict_proba, background)
    shap_values = explainer.shap_values(X_shap)
    
    # Plotting
    plt.figure()
    # shap.summary_plot usually creates its own figure.
    # We want a beeswarm plot.
    # For multi-class, shap_values is a list of arrays (one for each class).
    # We can plot for a specific class (e.g. DDoS) or all.
    # Summary plot for all classes is messy. 
    # Let's plot for the dominant class or just the summary plot which handles multi-class by color or overlay.
    # shap.summary_plot(shap_values, X_shap, feature_names=feature_names) works for multi-class?
    # It usually plots for Class 1 vs Rest or similar.
    # Let's try plotting for the first class or just calling summary_plot and letting SHAP handle it.
    # Ideally, we want to see what drives the model generally.
    
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
