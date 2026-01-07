# 🏥 IoMT Intrusion Detection System (IDS) with XAI

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-TabNet-red.svg)](https://github.com/dreamquark-ai/tabnet)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Proje Özeti

Bu proje, **IoMT (Internet of Medical Things)** ortamlarında oluşan ağ trafiğini analiz ederek siber saldırıları tespit eden, **TabNet tabanlı** ve **açıklanabilir yapay zekâ (XAI)** destekli bir **Saldırı Tespit Sistemi (IDS)** geliştirmeyi amaçlamaktadır.

Sistem, ağ trafiğini **çok sınıflı** olarak analiz etmekte ve saldırı tespit edildiğinde kararın nedenlerini **backend tarafında açıklanabilir şekilde** üretmektedir. Üretilen sonuçlar, **mobil uygulama** üzerinden gerçek zamanlı olarak görüntülenebilmektedir.

---

## 🎯 Desteklenen Saldırı Sınıfları

Model, ağ trafiğini aşağıdaki **6 sınıf** altında doğrudan sınıflandırmaktadır:

- **Benign**
- **DoS**
- **DDoS**
- **MQTT**
- **Recon**
- **Spoofing**

---

## 📊 Kullanılan Veri Seti

- **Dataset**: CIC-IoMT-2024  
- **Kaynak**: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html  
- **Format**: CSV   
- **Not**: Veri seti büyük olduğu için repoya eklenmemiştir.

---

## 🏗️ Proje Mimarisi

<p align="center">
  <img src="" width="800"/>
</p>

---

### 2️⃣ Sanal Ortam Oluşturun
python -m venv venv
source venv/bin/activate   # Linux / macOS
# veya
venv\Scripts\activate      # Windows

### 3️⃣ Bağımlılıkları Yükleyin
pip install -r requirements.txt

🚀 Model Eğitimi

TabNet tabanlı çok sınıflı IDS modelini eğitmek için:

python train_test_run_5.py

Bu işlem sonunda aşağıdaki dosyalar üretilir:
- multiclass_model.zip → Eğitilmiş model
- scaler.pkl → Ölçekleme modeli
- final_feature_names.pkl → Beklenen feature listesi
- label_encoder_multiclass.pkl → Sınıf eşleştirmeleri

---

## 🔍 Gerçek Zamanlı Trafik Simülasyonu

Eğitilen modelin sahada nasıl çalıştığını görmek için:
python simulate_traffic_unified.py

Bu script:
-Örnek bir trafik girdisi alır
-Model ile saldırı tahmini yapar
-Güven skoru üretir
-XAI açıklamasını backend mantığıyla oluşturur

---

## 🧠 Açıklanabilir Yapay Zekâ (XAI) Yaklaşımı
Sistemde iki seviyeli XAI yaklaşımı uygulanmaktadır:

###🔹 Global XAI (Offline)
SHAP Beeswarm grafikleri ile modelin genel feature importance analizi
Modelin hangi ağ özelliklerine daha fazla önem verdiği gösterilir

###🔹 Olay Bazlı XAI (Online)
Model tahmini sonrası, trafik örneği normal (benign) trafik istatistikleriyle karşılaştırılır
En çok sapma gösteren 3–5 özellik seçilerek açıklama metni üretilir
Açıklamalar backend tarafında kural ve istatistik tabanlı olarak oluşturulur

---

### 📱 Mobil Uygulama Entegrasyonu
Mobil uygulama Flutter ile geliştirilmiştir
Backend üzerinden gelen sonuçlar REST API ile gösterilir
Rol bazlı görünüm:
Doctor → sade açıklama
Admin → teknik detay + etkili feature listesi

---

### 📄 Lisans
Bu proje MIT Lisansı ile lisanslanmıştır.
Detaylar için LICENSE dosyasına bakınız.

---

### 👤 İletişim
Geliştiriciler: Emir Sözer, Simay Avcı
Proje Repo: https://github.com/IoMT-XAI-Research/CIC-IoMT-EDA-TabNet-XAI

⭐ Bu proje akademik amaçlarla geliştirilmiştir.





