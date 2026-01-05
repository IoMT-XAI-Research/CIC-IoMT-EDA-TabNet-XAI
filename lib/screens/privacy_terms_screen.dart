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
            _buildSection(
              "1. Giriş",
              "Bu uygulama (IoMT IDS), Tıbbi Nesnelerin İnterneti (IoMT) cihazlarınızın güvenliğini sağlamak için geliştirilmiştir. Uygulamayı kullanarak aşağıdaki şartları kabul etmiş olursunuz.",
            ),
            _buildSection(
              "2. Veri Güvenliği",
              "Kullanıcı verileri, en son şifreleme standartları kullanılarak korunmaktadır. Sistem, cihazlarınızdan gelen ağ trafiğini analiz eder ancak kişisel sağlık verilerinizi (PHI) toplamaz veya üçüncü taraflarla paylaşmaz. Tüm analizler anonimleştirilmiş trafik verileri üzerinden yapılır.",
            ),
            _buildSection(
              "3. Kullanım Şartları",
              "Uygulamanın sağladığı güvenlik uyarıları bilgilendirme amaçlıdır. Sistem %100 koruma garantisi vermez. Kritik altyapılarınızda ek güvenlik önlemleri almanız önerilir. Uygulamanın kötüye kullanımı, tersine mühendislik yapılması veya yetkisiz erişim denemeleri yasaktır.",
            ),
            _buildSection(
              "4. Sorumluluk Reddi",
              "Geliştirici ekip, uygulamanın kullanımı sırasında oluşabilecek doğrudan veya dolaylı zararlardan (veri kaybı, cihaz arızası vb.) sorumlu tutulamaz. Kullanıcı, sistemi kendi sorumluluğunda kullanır.",
            ),
            _buildSection(
              "5. İletişim",
              "Gizlilik politikası veya kullanım şartları ile ilgili sorularınız için bizimle iletişime geçebilirsiniz.\nE-posta: support@iomtsescurity.com",
            ),
            const SizedBox(height: 30),
            const Center(
              child: Text(
                "Son Güncelleme: 05 Ocak 2026",
                style: TextStyle(color: textMuted, fontStyle: FontStyle.italic),
              ),
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
