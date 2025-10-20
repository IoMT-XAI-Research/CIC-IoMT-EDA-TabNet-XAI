# CIC-IoMT-EDA-TabNet-XAI

### 🔍 Project Overview
This project performs **Exploratory Data Analysis (EDA)** and **Explainable Deep Learning (XAI)** on the **CIC-IoMT2024 dataset**, a benchmark dataset for **multi-protocol IoMT (Internet of Medical Things) network security assessment**.  
The study aims to understand, model, and explain **cyberattacks on IoMT devices** using a **Transformer-based TabNet model** and **SHAP (Shapley Additive Explanations)**.

---

## 🎯 Objectives
- Conduct **Exploratory Data Analysis (EDA)** to identify data characteristics, missing values, and outliers.  
- Perform **feature correlation** and visualize distributions of numerical and categorical variables.  
- Train a **TabNet model** for intrusion detection on IoMT traffic data.  
- Apply **Explainable AI (XAI)** methods (SHAP) to interpret model decisions.  
- Provide reproducible and transparent results following good data science practices.

---

## 📊 Dataset Information

- **Name:** CIC-IoMT2024 (University of New Brunswick, 2024)  
- **Protocols:** MQTT, Bluetooth, Wi-Fi, TCP/IP  
- **Attacks Covered:**  
  - *DoS, DDoS, ARP Spoofing, Reconnaissance, Malformed Packets, etc.*  
- **Official Source:**  
  🔗 [https://www.unb.ca/cic/datasets/iomt-dataset-2024.html](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html)

> ⚠️ Due to dataset size (~15 GB), only a **compressed sample (`merged_sample.csv.gz`)** is included for demonstration.  
> Full dataset can be downloaded from the official UNB link above.

---

## ⚙️ Installation & Setup

### 1- Clone the repository
git clone https://github.com/<yourusername>/CIC-IoMT2024-EDA-TabNet-XAI.git
cd CIC-IoMT2024-EDA-TabNet-XAI

### 2- Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # (Linux/Mac)
### or
.venv\Scripts\activate     # (Windows)

### 3- Install dependencies
pip install -r requirements.txt

### 4- Start Jupyter Notebook
jupyter notebook

## Technologies Used

Python 3.9+
Pandas, NumPy, Matplotlib, Seaborn
PyTorch, PyTorch-TabNet
SHAP
Scikit-Learn
Jupyter Notebook
