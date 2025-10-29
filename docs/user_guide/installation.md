# 🔧 Kurulum Rehberi

Bu rehber, IoMT IDS sistemini kurmak için gerekli adımları açıklar.

## 📋 Sistem Gereksinimleri

### Minimum Gereksinimler
- **İşletim Sistemi**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Python**: 3.9 veya üzeri
- **RAM**: 8 GB (16 GB önerilen)
- **Disk Alanı**: 20 GB boş alan
- **GPU**: CUDA destekli GPU (opsiyonel, hızlandırma için)

### Önerilen Gereksinimler
- **RAM**: 32 GB
- **GPU**: NVIDIA RTX 3080 veya üzeri
- **SSD**: NVMe SSD (hızlı veri okuma/yazma için)

## 🚀 Kurulum Adımları

### 1. Repository'yi Klonlayın

```bash
git clone GITLINK
cd IoMT_IDS
```

### 2. Sanal Ortam Oluşturun

```bash
# Python sanal ortamı oluştur
python -m venv .venv

# Sanal ortamı aktifleştir
# Linux/macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### 3. Bağımlılıkları Yükleyin

```bash
# Temel bağımlılıkları yükle
pip install -r requirements.txt

# GPU desteği için (opsiyonel)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Konfigürasyonu Ayarlayın

```bash
# Environment dosyasını kopyala
cp env.example .env

# Konfigürasyon dosyalarını düzenle
nano .env
```

### 5. Veri Setini Hazırlayın

```bash
# Demo veri seti zaten mevcut
# Tam veri seti için:
# 1. https://www.unb.ca/cic/datasets/iomt-dataset-2024.html adresinden indirin
# 2. data/raw/ klasörüne yerleştirin
# 3. Veri işleme scriptini çalıştırın:

python scripts/merge_to_parquet.py
```

### 6. Modeli Eğitin

```bash
# Temel model eğitimi
python scripts/train.py

# Gelişmiş eğitim (hiperparametre optimizasyonu ile)
python scripts/train_advanced.py
```

## 🔧 Gelişmiş Kurulum

### Docker ile Kurulum

```bash
# Docker image oluştur
docker build -t iomt-ids .

# Container çalıştır
docker run -p 8000:8000 -v $(pwd)/data:/app/data iomt-ids
```

### Kubernetes ile Kurulum

```bash
# Kubernetes deployment
kubectl apply -f k8s/
```

## 🧪 Kurulumu Test Etme

```bash
# Temel testler
python -m pytest tests/unit/

# Tüm testler
python -m pytest tests/

# API testi
python -m pytest tests/integration/test_api.py
```

## ❗ Sorun Giderme

### Yaygın Sorunlar

1. **CUDA Hatası**: GPU sürücülerini güncelleyin
2. **Memory Hatası**: Batch size'ı küçültün
3. **Import Hatası**: Sanal ortamın aktif olduğundan emin olun

### Log Dosyaları

```bash
# Log dosyalarını kontrol et
tail -f logs/app.log
```

## 📞 Destek

Kurulum sorunları için:
- GitHub Issues: [Issues sayfası](https://github.com/LINK/IoMT_IDS/issues)
- Email: MAIL









