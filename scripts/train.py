from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import yaml
import torch

# config yukle
with open("configs/tabnet.yaml") as f:
    cfg = yaml.safe_load(f)

# veriyi oku
data = pd.read_parquet(cfg["data"]["train_path"])
X = data.drop(columns=[cfg["data"]["label_col"]])
y = data[cfg["data"]["label_col"]]

# label encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# training ve validation bolmesi
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=cfg["data"]["test_size"], random_state=cfg["seed"])

# TabNerClassifier olusturulurken tum mimari parametrelerini yaml'dan cek
model = TabNetClassifier(
    n_d=cfg["model"]["n_d"],
    n_a=cfg["model"]["n_a"],
    n_steps=cfg["model"]["n_steps"],
    gamma=cfg["model"]["gamma"],
    lambda_sparse=cfg["model"]["lambda_sparse"],
    mask_type=cfg["model"]["mask_type"],
    seed=cfg["seed"]
)

# Egitimi baslar patience varsa egitimi bitir
model.fit(
    X_train.values, y_train,
    eval_set=[(X_val.values, y_val)],
    eval_name=["val"],
    eval_metric=["accuracy"],
    max_epochs=cfg["train"]["max_epochs"],
    patience=cfg["train"]["patience"],
    batch_size=cfg["train"]["batch_size"],
    virtual_batch_size=cfg["train"]["virtual_batch_size"]
)

# degerlendirme F1 Macrolari
pred = model.predict(X_val.values)
f1 = f1_score(y_val, pred, average="macro")
print(f"F1 Macro: {f1:.4f}")
print(classification_report(y_val, pred, target_names=le.classes_))

# modeli kaydet
model.save_model("artifacts/tabnet_model.zip")
joblib.dump(le, "artifacts/label_encoder.pkl")
print("Model ve encoder kaydedildi.")
