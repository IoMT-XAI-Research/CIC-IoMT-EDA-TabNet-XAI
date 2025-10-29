# ⚡ Hızlı Başlangıç

Bu rehber, IoMT IDS sistemini hızlıca çalıştırmak için gerekli adımları açıklar.

## 🎯 5 Dakikada Başlangıç

### 1. Sistemi Başlatın

```bash
# API servisini başlat
python service/api/main.py

# Başka bir terminalde stream processor'ı başlat
python service/stream_processor/main.py
```

### 2. Temel Kullanım

```python
# Python'da temel kullanım
from src.models.inference import ModelInference
from src.xai.explainer import SHAPExplainer

# Model yükle
inference = ModelInference("artifacts/models/tabnet_model.zip")
explainer = SHAPExplainer(inference.model)

# Tahmin yap
prediction = inference.predict(sample_data)
explanation = explainer.explain(sample_data)

print(f"Tahmin: {prediction}")
print(f"Açıklama: {explanation}")
```

### 3. API Kullanımı

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Tahmin yap
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, 3.4, 5.6, ...]}'

# Açıklama al
curl -X POST http://localhost:8000/api/v1/explain \
  -H "Content-Type: application/json" \
  -d '{"features": [1.2, 3.4, 5.6, ...]}'
```

## 📊 Veri Analizi

### Jupyter Notebook ile

```bash
# Jupyter başlat
jupyter notebook

# notebooks/ klasöründeki notebook'ları açın:
# - ARP_Spoofing_Analysis.ipynb
# - MQTT_DDoS_Analysis.ipynb
```

### Veri Görselleştirme

```python
import pandas as pd
import matplotlib.pyplot as plt
from src.data.preprocess import load_and_clean_data

# Veri yükle
data = load_and_clean_data("data/processed/merged_clean.parquet")

# Temel istatistikler
print(data.describe())

# Saldırı türlerinin dağılımı
data['attack_type'].value_counts().plot(kind='bar')
plt.title('Saldırı Türleri Dağılımı')
plt.show()
```

## 🔍 XAI Analizi

### SHAP Açıklamaları

```python
from src.xai.shap_explainer import SHAPExplainer
import shap

# Explainer oluştur
explainer = SHAPExplainer(model, X_train)

# Açıklama al
shap_values = explainer.explain(X_test[:10])

# Görselleştir
shap.summary_plot(shap_values, X_test[:10])
shap.waterfall_plot(shap_values[0])
```

### Adaptif Açıklamalar

```python
from src.xai.adaptive_explainer import AdaptiveExplainer

# Kullanıcı seviyesine göre açıklama
explainer = AdaptiveExplainer(model, user_level="beginner")
explanation = explainer.explain(sample_data, user_level="beginner")

print(explanation.summary)  # Basit açıklama
print(explanation.details)  # Detaylı açıklama
```

## 📱 Mobil Uygulama

### React Native ile

```bash
cd mobile_app
npm install
npm start
```

### API Entegrasyonu

```javascript
// API çağrısı
const response = await fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    features: extractedFeatures
  })
});

const result = await response.json();
console.log('Tahmin:', result.prediction);
console.log('Güven:', result.confidence);
```

## 🚨 Real-time Monitoring

### Stream Processing

```python
from src.streaming.processor import StreamProcessor

# Stream processor başlat
processor = StreamProcessor()
processor.start()

# Real-time analiz
processor.analyze_stream()
```

### Alerting

```python
from src.alerting.manager import AlertManager

# Alert manager oluştur
alert_manager = AlertManager()

# Alert gönder
alert_manager.send_alert(
    attack_type="DoS",
    confidence=0.95,
    details="SYN flood attack detected"
)
```

## 📈 Performans İzleme

### Model Metrikleri

```python
from src.monitoring.metrics import ModelMonitor

# Model izleme
monitor = ModelMonitor()
metrics = monitor.get_metrics()

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
```

### Dashboard

```bash
# Monitoring dashboard başlat
python src/monitoring/dashboard.py
```

## 🔧 Gelişmiş Kullanım

### Hiperparametre Optimizasyonu

```python
from src.training.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner()
best_params = tuner.optimize(X_train, y_train)
```

### Model Ensemble

```python
from src.models.ensemble import EnsembleModel

# Ensemble model oluştur
ensemble = EnsembleModel([model1, model2, model3])
predictions = ensemble.predict(X_test)
```

## 📚 Sonraki Adımlar

1. [API Dokümantasyonu](../api/endpoints.md) - Detaylı API kullanımı
2. [Model Açıklamaları](../model_explanations/tabnet.md) - TabNet modeli
3. [XAI Yöntemleri](../model_explanations/xai_methods.md) - Açıklanabilir AI
4. [Sorun Giderme](troubleshooting.md) - Yaygın sorunlar ve çözümleri









