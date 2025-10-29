# 🏥 IoMT Intrusion Detection System (IDS)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Proje Özeti

Bu proje, **CIC-IoMT-2024 veri seti** üzerinde **Transformer tabanlı TabNet** derin öğrenme modeli kullanarak IoMT (Internet of Medical Things) cihazlarındaki siber saldırıları tespit eden ve **Shapley Değerleri** ile açıklanabilir yapay zeka (XAI) sağlayan kapsamlı bir sistemdir.

### 🔍 Ana Özellikler

- **🤖 Gelişmiş ML Modeli**: Transformer tabanlı TabNet ile yüksek performanslı saldırı tespiti
- **🧠 Açıklanabilir AI**: SHAP değerleri ile model kararlarının şeffaf açıklaması
- **📱 Mobil Uygulama**: Real-time IoMT trafik analizi ve kullanıcı bildirimleri
- **⚡ Adaptif Sistem**: Duruma göre uyarlanan XAI açıklamaları
- **🔄 Real-time Processing**: Canlı veri akışı analizi

## 🎯 Desteklenen Saldırı Türleri

| Kategori | Saldırı Türü | Protokol |
|----------|---------------|----------|
| **DoS/DDoS** | SYN Flood, UDP Flood, ICMP Flood | TCP/IP |
| **MQTT Saldırıları** | Connect Flood, Publish Flood | MQTT |
| **Reconnaissance** | Port Scan, OS Scan, Ping Sweep | TCP/IP |
| **ARP Saldırıları** | ARP Spoofing | ARP |
| **Bluetooth Saldırıları** | DoS, Malformed Packets | Bluetooth |

## 📊 Veri Seti Bilgileri

- **Kaynak**: [CIC-IoMT-2024 Dataset](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html)
- **Boyut**: ~15 GB (tam veri seti)
- **Protokoller**: MQTT, Bluetooth, Wi-Fi, TCP/IP
- **Saldırı Sayısı**: 20+ farklı saldırı türü
- **Örnek Veri**: `data/processed/merged_sample.csv.gz` (demo için)

## 🏗️ Proje Yapısı

```
IoMT_IDS/
├── 📁 src/                          # Ana kaynak kod
│   ├── 📁 data/                     # Veri işleme
│   ├── 📁 models/                   # Model tanımları
│   ├── 📁 training/                 # Eğitim pipeline
│   ├── 📁 xai/                      # Açıklanabilir AI
│   ├── 📁 streaming/                # Real-time işleme
│   ├── 📁 feature_engineering/      # Özellik mühendisliği
│   ├── 📁 alerting/                # Uyarı sistemi
│   └── 📁 monitoring/               # Model izleme
├── 📁 service/                      # Backend servisler
│   ├── 📁 api/                      # REST API
│   ├── 📁 stream_processor/         # Veri akışı işleyici
│   ├── 📁 notification/             # Bildirim servisi
│   └── 📁 models/                   # Model servisi
├── 📁 configs/                      # Konfigürasyon dosyaları
├── 📁 data/                         # Veri setleri
│   ├── 📁 raw/                      # Ham veri
│   ├── 📁 processed/                # İşlenmiş veri
│   └── 📁 interim/                  # Ara veri
├── 📁 notebooks/                    # Jupyter notebook'lar
├── 📁 scripts/                      # Yardımcı scriptler
├── 📁 tests/                        # Test dosyaları
├── 📁 docs/                         # Dokümantasyon
├── 📁 artifacts/                    # Model ve sonuçlar
└── 📁 logs/                         # Log dosyaları
```

## ⚙️ Kurulum

### 1. Repository'yi Klonlayın
```bash
git clone https://github.com/yourusername/IoMT_IDS.git
cd IoMT_IDS
```

### 2. Sanal Ortam Oluşturun
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# veya
.venv\Scripts\activate     # Windows
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Veri Setini Hazırlayın
```bash
# Demo veri seti zaten mevcut
# Tam veri seti için: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
```

## 🚀 Kullanım

### Temel Eğitim
```bash
python scripts/train.py
```

### XAI Analizi
```bash
python src/xai/explain_predictions.py
```

### API Servisi Başlatma
```bash
python service/api/main.py
```

### Mobil Uygulama
```bash
# React Native uygulaması için
cd mobile_app
npm install
npm start
```

## 📈 Model Performansı (TAHMİN!!)

| Metrik | Değer |
|--------|-------|
| **F1-Score (Macro)** | 0.95+ |
| **Precision** | 0.94+ |
| **Recall** | 0.93+ |
| **Accuracy** | 0.96+ |

## 🧠 XAI Özellikleri

- **SHAP Değerleri**: Her özelliğin karar üzerindeki etkisi
- **Adaptif Açıklamalar**: Kullanıcı seviyesine göre açıklama detayı
- **Görsel Analiz**: Interaktif SHAP grafikleri
- **Real-time Açıklama**: Canlı tahmin açıklamaları

## 📱 Mobil Uygulama

- **Real-time Monitoring**: Canlı trafik analizi
- **Push Notifications**: Anında saldırı uyarıları
- **XAI Dashboard**: Açıklanabilir AI görselleştirmeleri
- **Offline Mode**: İnternet bağlantısı olmadan çalışma

## 🔧 Konfigürasyon

### Model Ayarları (`configs/tabnet.yaml`)
```yaml
model:
  n_d: 32              # Decision layer boyutu
  n_a: 32              # Attention layer boyutu
  n_steps: 5           # Decision step sayısı
  gamma: 1.5           # Sparsity parametresi
  lambda_sparse: 1e-4  # Sparsity regularization
```

### XAI Ayarları (`configs/xai.yaml`)
```yaml
shap:
  max_samples: 1000    # SHAP hesaplama için örnek sayısı
  background_size: 100 # Background veri boyutu
  explainer_type: "tree" # SHAP explainer türü
```

## 📊 Örnek Kullanım

### Veri Yükleme ve Ön İşleme
```python
from src.data.preprocess import load_and_clean_data

# Veri yükleme
data = load_and_clean_data("data/processed/merged_clean.parquet")
print(f"Veri boyutu: {data.shape}")
```

### Model Eğitimi
```python
from src.training.trainer import TabNetTrainer

trainer = TabNetTrainer(config_path="configs/tabnet.yaml")
model = trainer.train(data)
```

### XAI Analizi
```python
from src.xai.explainer import SHAPExplainer

explainer = SHAPExplainer(model)
explanations = explainer.explain_predictions(X_test)
explainer.visualize_explanations(explanations)
```

## 🧪 Test Etme

```bash
# Tüm testleri çalıştır
pytest tests/

# Belirli test kategorisi
pytest tests/unit/
pytest tests/integration/
```

## 📚 Dokümantasyon

- [API Dokümantasyonu](docs/api/)
- [Kullanıcı Kılavuzu](docs/user_guide/)
- [Model Açıklamaları](docs/model_explanations/)

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 📞 İletişim

- **Proje Sahibi**: Emir SÖZER/Simay AVCI
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile]

## 🙏 Teşekkürler

- [CIC-IoMT-2024 Dataset](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) - Veri seti sağlayıcısı
- [PyTorch TabNet](https://github.com/dreamquark-ai/tabnet) - TabNet implementasyonu
- [SHAP](https://github.com/slundberg/shap) - Açıklanabilir AI kütüphanesi

---

⭐ **Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**