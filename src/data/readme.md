data_loader.py: Temel amaci temiz veriyi alip model icin hazirlamak, bolmek, olcmek ve kodlamaktir.
Yani tamamen modelin veri girisine hazir hale gelmesi icin tum asamalari yonetir. Veri nerede sorusunun cevabini bulup sisteme getirir.

preprocess.py: merged_sample.csv dosyasini okur inf degerleri NaN ile degistirdikten sonra median ile doldurur, outliner degerleri %99.5 diliminde keser. Temizlenen veriyi data/processed/merged_clean.parquet oalrak kaydeder.
Yani veriyi egitime uygun hale getirir.

_init_.py: Bu bir connector (baglayici) gorevindedir, python'a bu klasorun bir modul oldugunu soyler.
Yani bu klasorun icinde hangi araclar disaridan erisebilir.

Pipeline Flow:
raw data -> preprocess.py -> merged_clean.parquet -> data_loader.py -> model-ready dataset



