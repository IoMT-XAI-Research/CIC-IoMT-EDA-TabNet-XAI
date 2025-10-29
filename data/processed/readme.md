manifest_processed.csv: Tum islenmis verilerin meta (ozet) bilgilerini icerir.
Yani hangi dosya islendi kac satir var temiz mi, egitime hazir mi

merged_sample.csv: Ham verilerin birlestirilmis ama henuz tamamen temizlenmemis versiyonudur. Model egitiminden hemen onceki ham birlestirme asamasidir. Henuz inf degerleri ve outlier'lar temizlenmemistir.

manifest_processed.csv: Bu dosya artik model egitimine hazir verilerin ozetidir.
Yani artik bu dosyalar model egitimi icin kullanilabilir durumda.

merged_clean.parquet: Modelin dogrudan egitimde kullandigi son ve temiz veri setidir. (sıkıstirilmis veri ML icin daha hizli)


