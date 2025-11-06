# 🤖 IoMT Intrusion Detection System (IDS)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org)
[![Spark](https://img.shields.io/badge/Spark-3.x-orange.svg)](https://spark.apache.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

IoMT IDS, **CIC-IoMT-2024** veri seti üzerinde, **Transformer‑tabanlı TabNet** ve **Spark data pipeline** ile saldırı tespiti yapar; kararları **SHAP (XAI)** ile açıklanabilir kılar.

> Not: Bölüm akışı, `OpenDriveLab/ETA` benzeri düzeni takip eder. Referans: [OpenDriveLab/ETA](https://github.com/OpenDriveLab/ETA).

---

## 📚 İçindekiler
1. Highlight
2. News
3. Results
4. Model Architecture
5. Getting Started (Training & Evaluation)
6. Dataset
7. Configuration
8. Project Structure
9. Feature Engineering (Rationale)
10. License & Citation

---

## ✨ Highlight
- Spark tabanlı veri pipeline: büyük veride hızlı temizlik ve ön‑işleme
- TabNet + Transformer tabanlı modelleme
- SHAP (XAI) ile açıklanabilirlik ve adaptif açıklamalar
- Real‑time stream processing ve alerting

---

## 🗞️ News
- 2025/11 – Spark pipeline entegre edildi; karşılaştırmalı MI raporları eklendi
- 2025/10 – Proje yapısı ve dokümantasyon güncellendi

---

## 🏁 Results

Buraya sonuç görselleri gelecek (confusion matrix, ROC, PR, per‑class F1).


Örnek metrik özeti:
- Accuracy: 0.96+
- F1 (Macro): 0.95+
- Precision/Recall: 0.94+/0.93+

---

## 🧱 Model Architecture

Buraya mimari diyagram gelecek (TabNet + FE + XAI + Stream pipeline).


---

## 🚀 Getting Started

### Training
```bash
python scripts/train.py
```

### Evaluation
```bash
# Spark tabanlı karşılaştırmalı FE raporu
python scripts/test_pipeline_spark.py

# Pandas tabanlı karşılaştırmalı FE raporu
python scripts/test_pipeline_compare.py
```

### Docker
```bash
# 1) Image oluştur
docker build -t iomt-ids:latest .

# 2) Çalıştır (veri klasörünü mount ederek)
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  -p 8000:8000 \
  iomt-ids:latest

# 3) Container içinde örnek komutlar
#   python scripts/test_pipeline_spark.py
#   python scripts/test_pipeline_compare.py
#   python scripts/train.py
#   python service/api/main.py

# Tek komutla çalıştır ve çık
docker run --rm -it -v $(pwd)/data:/app/data iomt-ids:latest \
  bash -lc "python scripts/test_pipeline_spark.py"
```

## 📦 Dataset
- Kaynak: [CIC‑IoMT‑2024 Dataset](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html)
- Protokoller: MQTT, Bluetooth, Wi‑Fi, TCP/IP
- Örnek: `data/processed/merged_sample.csv`

## 🗂️ Project Structure

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

## ⚙️ Configuration
- `configs/tabnet.yaml` – Model ve eğitim parametreleri
- `configs/streaming.yaml` – Stream/alert ayarları
- `configs/api.yaml` – Servis/API ayarları
- `configs/mobile.yaml` – Mobil istemci ayarları

## 🧪 Feature Engineering (Rationale)
FE; ağ saldırı dinamikleri ve istatistiksel davranışa dayanır. Test scriptleri MI (mutual information) ve gerekçeyi raporlar.

- Protokol oranları (`tcp_ratio`, `http_ratio`): flood/anomali dağılım dengesizlikleri
- TCP bayrak dinamikleri (`flag_diversity`, `syn_ack_ratio`, `rst_ratio`): SYN/ACK dengesizlikleri
- Hız/oynaklık (`packet_rate_mean/std/cv`): DoS/DDoS’ta hız ve varyans yükselir
- IAT istatistikleri (`iat_mean/std/cv`, pencere tabanlı `_min/_max/_iqr`): aralık instabilitesi
- Zaman pencereli istatistikler (rolling mean/std/min/max/q25/q75/iqr)
- Etkileşim/ratio/polynomial: çoklu sinyallerin birlikte etkisi ve nonlineer büyüklükler

Raporlar:
- `artifacts/results/feature_engineering_spark_comparison.json`
- `artifacts/results/feature_engineering_comparison.json`

## 📈 Results (Örnek)
Buraya sonuç görsellerini koyabilirsin (tablo/grafik).

> "buraya resmi koyabilirsin"

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

## 📚 Docs
- [API](docs/api/)
- [User Guide](docs/user_guide/)
- [Model Notes](docs/model_explanations/)

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📜 License & Citation
Bu proje MIT lisansı ile lisanslanmıştır. Ayrıntılar için [LICENSE](LICENSE).

Bu README’nin bölüm düzeni için ilham: [OpenDriveLab/ETA](https://github.com/OpenDriveLab/ETA).

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