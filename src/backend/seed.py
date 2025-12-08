from app.database import SessionLocal, engine, Base
from app import models
from passlib.context import CryptContext

# Veritabanı tablolarını oluştur (garanti olsun)
Base.metadata.create_all(bind=engine)

db = SessionLocal()

def seed_data():
    # 1. Önce Hastane Var mı Kontrol Et
    hospital = db.query(models.Hospital).filter(models.Hospital.code == "HST-001").first()
    
    if not hospital:
        print("🏥 Hastane oluşturuluyor...")
        hospital = models.Hospital(
            name="Merkez Şehir Hastanesi",
            code="HST-001"
        )
        db.add(hospital)
        db.commit()
        db.refresh(hospital)
    else:
        print("✅ Hastane zaten var.")

    # 2. Örnek Cihazları Ekle
    if db.query(models.Device).count() == 0:
        print("📟 Cihazlar ekleniyor...")
        devices = [
            models.Device(name="Oksijen Sensörü - Oda 302", ip_address="192.168.1.10", status="SAFE", hospital_id=hospital.id),
            models.Device(name="Akıllı Tansiyon Cihazı", ip_address="192.168.1.11", status="SAFE", hospital_id=hospital.id),
            models.Device(name="İlaç Pompası", ip_address="192.168.1.12", status="SAFE", hospital_id=hospital.id),
        ]
        db.add_all(devices)
        db.commit()
        print("✅ 3 adet cihaz eklendi.")
    else:
        print("✅ Cihazlar zaten ekli.")

    print("\n🎉 Kurulum Tamamlandı! Şimdi kayıt olabilirsiniz.")
    db.close()

if __name__ == "__main__":
    seed_data()