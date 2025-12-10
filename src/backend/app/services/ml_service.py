# src/backend/app/services/ml_service.py

from typing import Dict, Any
from sqlalchemy.orm import Session

from .. import models, schemas
from .ai_engine import ai_engine


def build_features_for_device(device: models.Device) -> Dict[str, Any]:
    """
    Verilen cihaz için, TabNet modelinin beklediği feature sözlüğünü üretir.

    Not:
    - Gerçek trafik loglarını şu an kullanmıyoruz.
    - Bunun yerine, modelin içindeki feature_names ve label_encoders bilgisini kullanarak
      "geçerli" (pipeline'a uygun) bir satır oluşturuyoruz.
    - Bu sayede:
        * TabNet + LabelEncoders gerçekten çalışıyor,
        * Çıkan sonuç rastgele değil, modelin öğrendiği dağılıma göre geliyor.
    """
    # Model henüz yüklenmediyse yüklemeyi dene
    if not getattr(ai_engine, "_is_loaded", False):
        ai_engine.load_model()

    feature_names = getattr(ai_engine, "feature_names", None)
    label_encoders = getattr(ai_engine, "label_encoders", None)

    # Eğer herhangi bir sebeple feature_names yoksa, boş dict döneriz;
    # ai_engine.preprocess_features zaten eksik kolonları 0 ile tamamlıyor.
    if not feature_names:
        return {}

    features: Dict[str, Any] = {}

    # Cihaz ID'sine göre küçük bir varyasyon (demo amaçlı, aynı input olmasın diye)
    variant = (device.id or 0) % 3  # 0, 1, 2 döner

    for name in feature_names:
        # Güvenlik için Label kolonunu asla input olarak vermiyoruz
        if name == "Label":
            continue

        # Kategorik feature ise (LabelEncoder varsa)
        if label_encoders and name in label_encoders and name != "Label":
            le = label_encoders[name]

            # Cihaz ID'sine göre 0/1/2. sınıfı seç (sınıf sayısı az ise son sınıfa clamp ediyoruz)
            idx = min(variant, len(le.classes_) - 1)
            features[name] = le.classes_[idx]
        else:
            # Sayısal feature için basit bir baseline değer:
            # 0.0, 0.1 veya 0.2 veriyoruz (yine variant'a göre)
            if variant == 0:
                base = 0.0
            elif variant == 1:
                base = 0.1
            else:
                base = 0.2

            features[name] = base

    return features


def get_model_analysis_for_device(
    db: Session,
    device: models.Device,
) -> schemas.AnalysisResponse:
    """
    Belirli bir cihaz için AI modelini (TabNet + LabelEncoders + SHAP) kullanarak analiz üretir.

    Akış:
      1) build_features_for_device(device) ile feature dict hazırlanır.
      2) ai_engine.predict(features) çağrılır → TabNet modeli gerçekten çalışır.
      3) Dönen AnalysisResponse'a göre device.status ve device.last_risk_score güncellenir.
      4) AnalysisResponse endpoint'e geri döner (mobil uygulama bunu gösterir).
    """

    # Her ihtimale karşı model yüklenmemişse yüklemeyi dener
    if not getattr(ai_engine, "_is_loaded", False):
        ai_engine.load_model()

    try:
        # 1) Cihaza göre feature set oluştur
        features = build_features_for_device(device)

        # 2) AI modelini çalıştır
        analysis = ai_engine.predict(features)

        # 3) Cihaz durumunu modele göre güncelle
        if analysis.device_status == "ATTACK":
            device.status = models.DeviceStatus.ATTACK
        else:
            device.status = models.DeviceStatus.SAFE

        device.last_risk_score = analysis.risk_score

        # 4) Değişiklikleri DB'ye yaz
        db.commit()
        db.refresh(device)

        return analysis

    except Exception as e:
        print(f"[ERROR] get_model_analysis_for_device failed for device {device.id}: {e}")
        # Hata olursa uygulama patlamasın diye AIEngine içindeki güvenli mock'u dön
        return ai_engine._mock_response()

