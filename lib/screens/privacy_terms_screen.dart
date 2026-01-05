import 'package:flutter/material.dart';

// Reusing colors from main.dart to ensure theme consistency
// Ideally these should be in a separate constants file, but for now we follow the existing pattern.
const Color darkBackground = Color(0xFF121212);
const Color cardColor = Color(0xFF242424);
const Color neonGreen = Color(0xFF00E676);
const Color textLight = Color(0xFFE0E0E0);
const Color textMuted = Color(0xFFAAAAAA);

class PrivacyTermsScreen extends StatelessWidget {
  const PrivacyTermsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: darkBackground,
      appBar: AppBar(
        title: const Text('Gizlilik ve Şartlar',
            style: TextStyle(fontWeight: FontWeight.bold, color: textLight)),
        backgroundColor: darkBackground,
        elevation: 0,
        iconTheme: const IconThemeData(color: textLight),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "KULLANIM KOŞULLARI VE GİZLİLİK POLİTİKASI",
              style: TextStyle(
                  fontSize: 20, fontWeight: FontWeight.bold, color: neonGreen),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            const Center(
              child: Text(
                "Son Güncelleme: 05.01.2026",
                style: TextStyle(color: textMuted, fontStyle: FontStyle.italic),
              ),
            ),
            const SizedBox(height: 25),
            _buildSection(
              "1. GENEL HÜKÜMLER",
              "İşbu Kullanım Koşulları ve Gizlilik Politikası (\"Sözleşme\"), TÜBİTAK projesi kapsamında geliştirilen \"IoMT Saldırı Tespit Sistemi\" (\"Uygulama\") kullanıcısı ile geliştirici ekip arasındaki yasal hak ve yükümlülükleri belirler. Uygulamayı indirerek veya kullanarak, bu sözleşmenin tamamını okuduğunuzu, anladığınızı ve kabul ettiğinizi beyan edersiniz.",
            ),
            _buildSection(
              "2. VERİ GİZLİLİĞİ VE KVKK AYDINLATMA METNİ",
              "Uygulama, Tıbbi Nesnelerin İnterneti (IoMT) cihazlarınızın ağ trafiğini analiz etmek amacıyla çalışır.\n\n"
                  "• Veri İşleme: Sistem, saldırı tespiti yapabilmek için ağ paketlerini (packet sniffing) analiz eder. Bu veriler yalnızca anomali tespiti amacıyla işlenir.\n\n"
                  "• Kişisel Sağlık Verileri (PHI): Uygulama, hastaların kimlik bilgilerini veya doğrudan tıbbi kayıtlarını sunuculara kaydetmez. Analiz edilen veriler, anonimleştirilmiş ağ trafik öznitelikleridir.\n\n"
                  "• Üçüncü Taraflar: Toplanan anonim teknik veriler, akademik araştırma ve sistem iyileştirme amaçları dışında üçüncü taraflarla paylaşılmaz.",
            ),
            _buildSection(
              "3. YAPAY ZEKA VE SORUMLULUK REDDİ (DISCLAIMER)",
              "Bu sistem, Derin Öğrenme (Deep Learning) ve Makine Öğrenmesi (TabNet/XAI) algoritmaları kullanılarak geliştirilmiştir.\n\n"
                  "• Karar Destek Sistemi: Uygulama bir \"karar destek sistemi\" niteliğindedir. Sağlanan güvenlik skorları ve saldırı alarmları olasılıksal temellere dayanır.\n\n"
                  "• Garanti Yoktur: Algoritmalar %100 doğruluk oranı vaat etmez. Yanlış Pozitif (False Positive) veya Yanlış Negatif (False Negative) sonuçlar oluşabilir.\n\n"
                  "• Kullanıcı Sorumluluğu: Bu uygulama, profesyonel siber güvenlik donanımlarının veya uzman denetiminin yerini alamaz. Kritik tıbbi altyapılarda tek güvenlik önlemi olarak kullanılmamalıdır.",
            ),
            _buildSection(
              "4. TELİF HAKLARI VE FİKRİ MÜLKİYET",
              "Uygulamanın kaynak kodları, kullanılan yapay zeka modelleri ve arayüz tasarımları proje ekibine aittir. İzinsiz kopyalanması, tersine mühendislik (reverse engineering) yapılması veya ticari amaçla dağıtılması yasaktır.",
            ),
            _buildSection(
              "5. İLETİŞİM",
              "Güvenlik açığı bildirimleri veya hukuki sorularınız için proje ekibiyle iletişime geçebilirsiniz.\n(E-posta: simayavci2022@gmail.com - emirsozeres@gmail.com)",
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildSection(String title, String content) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 25.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: neonGreen,
            ),
          ),
          const SizedBox(height: 10),
          Text(
            content,
            style: const TextStyle(
              fontSize: 15,
              color: textLight,
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }
}
