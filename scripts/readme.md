build_manifest.py: data/raw klasorundeki tum ham verileri tarar. Bunlari tek bir tabloya kaydeder: data/manifest.csv

merged_sample.py	: manifest.csv dosyasindaki yollari kullanarak tum verileri okur ve gelen verileri tek birlestirilmis CSV haline getirir: data/processed/merged_sample.csv

merge_to_parquet.py: merged_sample.csv dosyasini alir, veri temizleme islemleri uygular ve parquet formatina donusturur: data/manifest_processed.csv ve data/processed/merged_clean.parquet (ML icin sıkıstırılmıs daha hizli)

simple_test.py: Basit testler calistirmak icin kullanilir, veri yollarinin dogru calisip calismadigini veya bagimliliklarin yuklu olup olmadigini kontrol eder.

test_data_pipeline.py: Tum pipeline (veri hattini) uctan uca test eder: data/processed/merged_sample.csv dosyasini test eder.

train.py: TabNet modelini egitir, configs/tabnet.yaml dosyasindan model hiperparametrelerini okur., egitim verisini merged_clean.parquet'ten yukler. Egitim tamamlandiktan sonra modeli artifacts/tabnet_model.zip olarak kaydeder, label encoder'i ise artifacts/label_encoder.pkl olarak kaydeder. Son olarak egitim metriklerini (F1, Accuarcy vb.) terminale yazar.


