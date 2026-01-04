import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'firebase_options.dart';
import 'api_service.dart';
import 'dart:async';
import 'dart:convert';
import 'dart:io';

// ANA RENKLER

const Color darkBackground = Color(0xFF121212);
const Color cardColor = Color(0xFF242424);
const Color neonGreen = Color(0xFF00E676);
const Color neonRed = Color(0xFFFF0055);
const Color neonBlue = Color(0xFF00E5FF);
const Color neonYellow = Color(0xFFFFD600);
const Color textLight = Color(0xFFE0E0E0);
const Color textMuted = Color(0xFFAAAAAA);
const Color accentBlue = Color(0xFF00BFFF);

// =====================================================
// GLOBAL HELPER: Broad Threat Detection
// If status is NOT Safe and NOT Benign, it's an attack
// This handles: "Spoofing", "DDoS", "Botnet", "ATTACK", etc.
// =====================================================
bool isDeviceUnderAttack(String? status) {
  final s = (status ?? 'SAFE').toString().trim().toUpperCase();
  return s != 'SAFE' && s != 'BENIGN';
}

//ANA UYGULAMA BAŞLANGICI VE FIREBASE

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  try {
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform,
    );
  } catch (e) {
    debugPrint('Firebase Initialization Error: $e');
  }

  runApp(const IoMTIDSApp());
}

class IoMTIDSApp extends StatelessWidget {
  const IoMTIDSApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'IoMT IDS',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: darkBackground,
        textTheme: const TextTheme(
          bodyMedium: TextStyle(color: textLight),
          titleMedium: TextStyle(color: textLight),
        ),
        inputDecorationTheme: const InputDecorationTheme(
          filled: true,
          fillColor: cardColor,
          labelStyle: TextStyle(color: textMuted),
          hintStyle: TextStyle(color: textMuted),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.all(Radius.circular(8.0)),
            borderSide: BorderSide.none,
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.all(Radius.circular(8.0)),
            borderSide: BorderSide(color: neonGreen, width: 2),
          ),
        ),
      ),
      home: const SplashScreen(),
    );
  }
}

// 3. SPLASH SCREEN VE AUTH WRAPPER

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    Future.delayed(const Duration(seconds: 3), () {
      if (mounted) {
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(builder: (context) => const AuthWrapper()),
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      backgroundColor: darkBackground,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Icon(Icons.security, size: 100, color: neonGreen),
            SizedBox(height: 20),
            Text(
              'IoMT IDS YÖNETİMİ',
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.bold,
                color: textLight,
                letterSpacing: 1.5,
              ),
            ),
            SizedBox(height: 8),
            Text(
              'Sistem Yükleniyor...',
              style: TextStyle(
                fontSize: 16,
                color: textMuted,
              ),
            ),
            SizedBox(height: 50),
            CircularProgressIndicator(color: neonGreen),
          ],
        ),
      ),
    );
  }
}

class AuthWrapper extends StatelessWidget {
  const AuthWrapper({super.key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder<User?>(
      stream: FirebaseAuth.instance.authStateChanges(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator(color: neonGreen)),
            backgroundColor: darkBackground,
          );
        }

        final user = snapshot.data;

        if (user != null) {
          if (!user.emailVerified) {
            return const EmailVerificationScreen();
          }
          return const DashboardScreen();
        }

        return const LoginScreen();
      },
    );
  }
}

// 4. KAYIT OL EKRANI

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  State<SignUpScreen> createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _hospitalCodeController = TextEditingController();
  String? _errorText;
  bool _isLoading = false;
  bool _isPasswordVisible = false;

  // Role Selection
  String _selectedRole = 'TECH_STAFF'; // Default

  Future<void> _signUp() async {
    // Basic Validation
    if (_emailController.text.trim().isEmpty ||
        _passwordController.text.trim().isEmpty) {
      setState(() => _errorText = 'Lütfen e-posta ve şifre girin.');
      return;
    }

    if (_selectedRole == 'TECH_STAFF' &&
        _hospitalCodeController.text.trim().isEmpty) {
      setState(() => _errorText = 'Personel için Hastane Kodu zorunludur.');
      return;
    }

    setState(() {
      _isLoading = true;
      _errorText = null;
    });

    try {
      await ApiService().register(
        email: _emailController.text.trim(),
        password: _passwordController.text.trim(),
        role: _selectedRole,
        hospitalCode: _selectedRole == 'TECH_STAFF'
            ? _hospitalCodeController.text.trim()
            : null,
      );

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Text('Kayıt Başarılı! Lütfen giriş yapın.'),
            backgroundColor: neonGreen,
          ),
        );
        Navigator.of(context).pop(); // Return to Login Screen
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorText =
              'Kayıt Hatası: ${e.toString().replaceAll("Exception: ", "")}';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Widget _buildRoleCard(String title, String role, IconData icon) {
    final isSelected = _selectedRole == role;
    return GestureDetector(
      onTap: () {
        setState(() {
          _selectedRole = role;
          _errorText = null;
        });
      },
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isSelected ? neonGreen.withOpacity(0.2) : cardColor,
          border: Border.all(
            color: isSelected ? neonGreen : Colors.transparent,
            width: 2,
          ),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          children: [
            Icon(icon, color: isSelected ? neonGreen : textMuted, size: 32),
            const SizedBox(height: 8),
            Text(
              title,
              style: TextStyle(
                color: isSelected ? neonGreen : textMuted,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Yeni Hesap Oluştur',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: darkBackground,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(30.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Icon(Icons.person_add, size: 60, color: neonGreen),
            const SizedBox(height: 30),

            // Role Selection
            const Text(
              "Rol Seçimi",
              style: TextStyle(color: textLight, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 15),
            Row(
              children: [
                Expanded(
                  child: _buildRoleCard(
                      "Teknik Personel", "TECH_STAFF", Icons.engineering),
                ),
                const SizedBox(width: 15),
                Expanded(
                  child: _buildRoleCard(
                      "Yönetici (Admin)", "ADMIN", Icons.admin_panel_settings),
                ),
              ],
            ),
            const SizedBox(height: 30),

            if (_errorText != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 15),
                child: Text(
                  _errorText!,
                  style: TextStyle(color: neonRed, fontWeight: FontWeight.bold),
                  textAlign: TextAlign.center,
                ),
              ),

            // Email & Password
            TextFormField(
              controller: _emailController,
              keyboardType: TextInputType.emailAddress,
              decoration: const InputDecoration(
                labelText: 'E-posta Adresi',
                prefixIcon: Icon(Icons.email, color: textMuted),
              ),
              style: const TextStyle(color: textLight),
            ),
            const SizedBox(height: 20),
            TextFormField(
              controller: _passwordController,
              obscureText: !_isPasswordVisible,
              decoration: InputDecoration(
                labelText: 'Şifre (Min. 6 Karakter)',
                prefixIcon: const Icon(Icons.lock, color: textMuted),
                suffixIcon: IconButton(
                  icon: Icon(
                    _isPasswordVisible
                        ? Icons.visibility
                        : Icons.visibility_off,
                    color: textMuted,
                  ),
                  onPressed: () {
                    setState(() {
                      _isPasswordVisible = !_isPasswordVisible;
                    });
                  },
                ),
              ),
              style: const TextStyle(color: textLight),
            ),

            // Conditional Hospital Code
            if (_selectedRole == 'TECH_STAFF') ...[
              const SizedBox(height: 20),
              TextFormField(
                controller: _hospitalCodeController,
                decoration: const InputDecoration(
                  labelText: 'Hastane Kodu',
                  prefixIcon: Icon(Icons.local_hospital, color: textMuted),
                  helperText: "Yöneticinizden aldığınız kod",
                  helperStyle: TextStyle(color: textMuted),
                ),
                style: const TextStyle(color: textLight),
              ),
            ],

            const SizedBox(height: 40),
            _isLoading
                ? const Center(
                    child: CircularProgressIndicator(color: neonGreen))
                : ElevatedButton(
                    onPressed: _signUp,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: neonGreen,
                      padding: const EdgeInsets.symmetric(vertical: 15),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8)),
                      shadowColor: neonGreen.withOpacity(0.5),
                      elevation: 10,
                    ),
                    child: const Text(
                      'HESAP OLUŞTUR',
                      style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: darkBackground),
                    ),
                  ),
          ],
        ),
      ),
    );
  }
}

// 5. GİRİŞ EKRANI

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  String? _errorText;
  bool _isLoading = false;
  bool _isPasswordVisible = false;

  Future<void> _login() async {
    if (_emailController.text.trim().isEmpty ||
        _passwordController.text.trim().isEmpty) {
      setState(() => _errorText = 'Lütfen tüm alanları doldurun.');
      return;
    }

    setState(() {
      _isLoading = true;
      _errorText = null;
    });

    try {
      await ApiService().login(
        _emailController.text.trim(),
        _passwordController.text.trim(),
      );

      if (mounted) {
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(builder: (context) => const DashboardScreen()),
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorText =
              'Giriş Başarısız: ${e.toString().replaceAll("Exception: ", "")}';
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
                'Giriş Başarısız: Kullanıcı bulunamadı veya şifre hatalı.'),
            backgroundColor: neonRed,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void _navigateToForgotPassword(BuildContext context) {
    Navigator.of(context).push(
        MaterialPageRoute(builder: (context) => const ForgotPasswordScreen()));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(30.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: <Widget>[
              Icon(Icons.security, size: 80, color: neonGreen),
              const SizedBox(height: 10),
              const Text('IoMT IDS YÖNETİMİ',
                  textAlign: TextAlign.center,
                  style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: textLight)),
              const SizedBox(height: 50),
              if (_errorText != null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 15),
                  child: Text(
                    _errorText!,
                    style:
                        TextStyle(color: neonRed, fontWeight: FontWeight.bold),
                    textAlign: TextAlign.center,
                  ),
                ),
              TextFormField(
                controller: _emailController,
                keyboardType: TextInputType.emailAddress,
                decoration: const InputDecoration(
                    labelText: 'E-posta Adresi',
                    prefixIcon: Icon(Icons.email, color: textMuted)),
                style: const TextStyle(color: textLight),
              ),
              const SizedBox(height: 20),
              TextFormField(
                controller: _passwordController,
                obscureText: !_isPasswordVisible,
                decoration: InputDecoration(
                  labelText: 'Şifre',
                  prefixIcon: const Icon(Icons.lock, color: textMuted),
                  suffixIcon: IconButton(
                    icon: Icon(
                      _isPasswordVisible
                          ? Icons.visibility
                          : Icons.visibility_off,
                      color: textMuted,
                    ),
                    onPressed: () {
                      setState(() {
                        _isPasswordVisible = !_isPasswordVisible;
                      });
                    },
                  ),
                ),
                style: const TextStyle(color: textLight),
              ),
              const SizedBox(height: 10),
              Align(
                alignment: Alignment.centerRight,
                child: TextButton(
                  onPressed: () => _navigateToForgotPassword(context),
                  child: Text('Şifremi Unuttum?',
                      style: TextStyle(color: textMuted.withOpacity(0.8))),
                ),
              ),
              const SizedBox(height: 20),
              _isLoading
                  ? const Center(
                      child: CircularProgressIndicator(color: neonGreen))
                  : ElevatedButton(
                      onPressed: _login,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: neonGreen,
                        padding: const EdgeInsets.symmetric(vertical: 15),
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8)),
                        shadowColor: neonGreen.withOpacity(0.5),
                        elevation: 10,
                      ),
                      child: const Text('GİRİŞ YAP',
                          style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              color: darkBackground)),
                    ),
              const SizedBox(height: 20),
              TextButton(
                onPressed: () {
                  Navigator.of(context).push(MaterialPageRoute(
                      builder: (context) => const SignUpScreen()));
                },
                child: const Text(
                  'Hala hesabın yok mu? Üye Ol',
                  style: TextStyle(
                    color: accentBlue,
                    fontWeight: FontWeight.bold,
                    decoration: TextDecoration.underline,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// 7. ŞİFREMİ UNUTTUM EKRANI

class ForgotPasswordScreen extends StatefulWidget {
  const ForgotPasswordScreen({super.key});

  @override
  State<ForgotPasswordScreen> createState() => _ForgotPasswordScreenState();
}

class _ForgotPasswordScreenState extends State<ForgotPasswordScreen> {
  final _emailController = TextEditingController();
  String? _errorText;
  bool _isLoading = false;
  bool _emailSent = false;

  Future<void> _sendPasswordReset() async {
    if (_emailController.text.trim().isEmpty) {
      setState(() => _errorText = 'Lütfen tüm alanları doldurun.');
      return;
    }

    setState(() {
      _isLoading = true;
      _errorText = null;
      _emailSent = false;
    });

    try {
      await FirebaseAuth.instance.sendPasswordResetEmail(
        email: _emailController.text.trim(),
      );

      if (mounted) {
        setState(() {
          _emailSent = true;
        });
      }
    } on FirebaseAuthException catch (e) {
      if (mounted) {
        setState(() {
          if (e.code == 'user-not-found' || e.code == 'invalid-email') {
            _errorText = 'Kullanıcı bulunamadı veya e-posta hatalı.';
          } else {
            _errorText = 'Hata: ${e.message}';
          }
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Şifre Sıfırlama',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: darkBackground,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(30.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Icon(Icons.lock_reset, size: 60, color: neonGreen),
            const SizedBox(height: 30),
            const Text(
              'Kayıtlı e-posta adresinizi girin. Size şifrenizi sıfırlamanız için bir bağlantı göndereceğiz.',
              textAlign: TextAlign.center,
              style: TextStyle(color: textMuted, fontSize: 16),
            ),
            const SizedBox(height: 30),
            if (_emailSent)
              Container(
                padding: const EdgeInsets.all(15),
                decoration: BoxDecoration(
                  color: neonGreen.withOpacity(0.1),
                  border: Border.all(color: neonGreen),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Bağlantı başarıyla gönderildi! Lütfen e-posta kutunuzu kontrol edin.',
                  style:
                      TextStyle(color: neonGreen, fontWeight: FontWeight.bold),
                  textAlign: TextAlign.center,
                ),
              ),
            const SizedBox(height: 20),
            if (_errorText != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 15),
                child: Text(
                  _errorText!,
                  style: TextStyle(color: neonRed, fontWeight: FontWeight.bold),
                  textAlign: TextAlign.center,
                ),
              ),
            TextFormField(
              controller: _emailController,
              keyboardType: TextInputType.emailAddress,
              decoration: const InputDecoration(
                labelText: 'Kayıtlı E-posta Adresi',
                prefixIcon: Icon(Icons.email, color: textMuted),
              ),
              style: const TextStyle(color: textLight),
            ),
            const SizedBox(height: 40),
            _isLoading
                ? const Center(
                    child: CircularProgressIndicator(color: neonGreen))
                : ElevatedButton(
                    onPressed: _sendPasswordReset,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: neonGreen,
                      padding: const EdgeInsets.symmetric(vertical: 15),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8)),
                      shadowColor: neonGreen.withOpacity(0.5),
                      elevation: 10,
                    ),
                    child: const Text(
                      'ŞİFRE SIFIRLAMA BAĞLANTISI GÖNDER',
                      style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: darkBackground),
                    ),
                  ),
          ],
        ),
      ),
    );
  }
}

// 8. E-POSTA DOĞRULAMA EKRANI

class EmailVerificationScreen extends StatelessWidget {
  const EmailVerificationScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
          title: const Text('E-posta Doğrulaması',
              style: TextStyle(fontWeight: FontWeight.bold)),
          backgroundColor: darkBackground),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(30.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Icon(Icons.mail_outline, size: 80, color: neonGreen),
              const SizedBox(height: 30),
              const Text(
                'Hesabınız Kaydedildi!',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 10),
              const Text(
                'Lütfen devam etmek için e-posta adresinize gönderilen doğrulama bağlantısına tıklayın.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 16, color: textMuted),
              ),
              const SizedBox(height: 40),
              ElevatedButton(
                onPressed: () async {
                  await FirebaseAuth.instance.currentUser?.reload();
                  if (FirebaseAuth.instance.currentUser?.emailVerified ??
                      false) {
                    if (context.mounted) {
                      Navigator.of(context).pushReplacement(
                        MaterialPageRoute(
                            builder: (context) => const AuthWrapper()),
                      );
                    }
                  } else {
                    if (context.mounted) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(
                            content: Text(
                                'Doğrulama henüz tamamlanmadı. E-postanızı kontrol edin.'),
                            backgroundColor: neonRed),
                      );
                    }
                  }
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: accentBlue,
                  padding:
                      const EdgeInsets.symmetric(vertical: 15, horizontal: 30),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8)),
                ),
                child: const Text('Doğruladım (Yeniden Kontrol Et)',
                    style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: darkBackground)),
              ),
              TextButton(
                onPressed: () async {
                  await FirebaseAuth.instance.currentUser
                      ?.sendEmailVerification();
                  if (context.mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                          content: Text('Yeni doğrulama e-postası gönderildi.'),
                          backgroundColor: neonGreen),
                    );
                  }
                },
                child: const Text('Tekrar E-posta Gönder',
                    style: TextStyle(color: textMuted)),
              ),
              const SizedBox(height: 30),
              TextButton(
                onPressed: () => FirebaseAuth.instance.signOut(),
                child: const Text('Farklı Hesapla Giriş Yap',
                    style: TextStyle(color: neonRed)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// 9. DASHBOARD EKRANI

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen>
    with SingleTickerProviderStateMixin {
  bool isAlert = false;
  bool _isDialogOpen = false;
  late AnimationController _controller;
  int _deviceCount = 0;

  // ✅ Backend ile konuşmak için ApiService
  final ApiService _apiService = ApiService();

  // ✅ Periyodik kontrol için Timer
  Timer? _pollTimer;

  void _setAlertState(bool newIsAlert) {
    if (newIsAlert == isAlert) return; // Aynı duruma tekrar geçme

    setState(() {
      isAlert = newIsAlert;
    });

    if (newIsAlert) {
      // ATTACK => popup aç
      _showAttackPopup();
    } else {
      // SAFE => popup açıksa kapat
      if (_isDialogOpen) {
        Navigator.of(context, rootNavigator: true).pop();
        _isDialogOpen = false;
      }
    }
  }

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    )..repeat(reverse: true);

    // Uygulama açılır açılmaz backend'i dinlemeye başla
    _startPollingAttackStatus();
    _loadStats();
  }

  Future<void> _loadStats() async {
    try {
      final stats = await _apiService.getStats();
      if (mounted) {
        setState(() {
          _deviceCount = stats['device_count'] ?? 0;
        });
      }
    } catch (e) {
      debugPrint('Error loading stats: $e');
    }
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _controller.dispose();
    super.dispose();
  }

  // ✅ Her 3 saniyede bir backend'ten status çek
  void _startPollingAttackStatus() {
    // İlk başta hemen bir kez çağır
    _checkAttackStatus();

    _pollTimer = Timer.periodic(const Duration(seconds: 3), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      _checkAttackStatus();
      // Periodically refresh stats too if needed, but maybe less frequent or same
      // for now just once is fine as per request context, but keeping it fresh is better
      _loadStats();
    });
  }

  // ✅ Backend'teki tüm hastane/cihazları tarayarak global saldırı durumunu kontrol et
  Future<void> _checkAttackStatus() async {
    try {
      final hospitals = await _apiService.getHospitals();
      if (hospitals.isEmpty) return;

      bool anyAttackFound = false;

      // Scan ALL hospitals and ALL devices
      for (var hospital in hospitals) {
        if (!mounted) return;
        try {
          final devices = await _apiService.getDevices(hospital['unique_code']);
          for (var device in devices) {
            if (isDeviceUnderAttack(device['status'])) {
              anyAttackFound = true;
            }
          }
        } catch (e) {
          debugPrint('Error checking hospital ${hospital['unique_code']}: $e');
        }
      }

      if (!mounted) return;

      // Update alert state based on global scan
      _setAlertState(anyAttackFound);
    } catch (e) {
      debugPrint('Attack status check error: $e');
    }
  }

  // Bu artık sadece istersen elle test için duruyor,
  // gerçek senaryoda backend'ten gelen ATTACK durumu kullanılıyor.
  void toggleAlert() {
    _setAlertState(!isAlert);
  }

  void _showAttackPopup() {
    _isDialogOpen = true;

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (BuildContext dialogContext) {
        return AlertPopup(
          onIsolate: () {
            // Kullanıcı butona basınca dialog kapansın
            _isDialogOpen = false;
            Navigator.of(dialogContext).pop();

            // İstersen burada da SAFE’e çekebilirsin:
            _setAlertState(false);
          },
        );
      },
    );
  }

  void _navigateToDetail(
      BuildContext context, String deviceName, bool isCurrentlyAlert) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => DeviceDetailScreen(
          device: {
            'name': deviceName,
            'status': isCurrentlyAlert ? 'ATTACK' : 'SAFE',
            'ip_address': '192.168.1.X', // Dummy for dash
            'room_number': 'Bilinmiyor'
          },
          userRole: 'Teknik Personel',
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final userEmail = FirebaseAuth.instance.currentUser?.email ?? 'Kullanıcı';
    // isOxygenSensorAlert variable removed

    return Scaffold(
      appBar: AppBar(
        title: const Text('IoMT Koruma Merkezi',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: darkBackground,
        elevation: 0,
        centerTitle: false,
        actions: [
          Tooltip(
            message: userEmail,
            child: const Icon(Icons.verified_user, color: neonGreen),
          ),
          IconButton(
            icon: const Icon(Icons.logout, color: textMuted),
            onPressed: () async {
              final prefs = await SharedPreferences.getInstance();
              await prefs.clear();
              if (context.mounted) {
                Navigator.of(context).pushAndRemoveUntil(
                  MaterialPageRoute(builder: (context) => const LoginScreen()),
                  (route) => false,
                );
              }
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            StatusArea(isAlert: isAlert, controller: _controller),
            const SizedBox(height: 30),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                InfoCard(title: 'Toplam Cihaz', value: '$_deviceCount'),
                const InfoCard(title: '24s Olağandışı Trafik', value: '0'),
              ],
            ),
            const SizedBox(height: 30),
            const Text('Hızlı Erişim',
                style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: textLight)),
            const Divider(color: cardColor, thickness: 2),
            const SizedBox(height: 15),

            // Demo statik liste yerine dinamik inventory ekranına yönlendirme
          ],
        ),
      ),
      bottomNavigationBar: NavBar(
        // Artık alarm backend'ten geldiği için, bu butonu kullanmak zorunda değilsin.
        onSimulate: () {
          Navigator.of(context).push(
            MaterialPageRoute(builder: (context) => const MonitoringScreen()),
          );
        },
        onNavigate: (index) {
          if (index == 2) {
            // Hospitals
            Navigator.of(context).push(MaterialPageRoute(
                builder: (context) => const HospitalManagementScreen()));
          } else if (index == 3) {
            // Activity Log
            Navigator.of(context).push(MaterialPageRoute(
                builder: (context) => const ActivityLogScreen()));
          } else if (index == 4) {
            // Devices
            Navigator.of(context).push(MaterialPageRoute(
                builder: (context) => const DeviceInventoryScreen()));
          } else if (index == 5) {
            // Settings
            Navigator.of(context).push(MaterialPageRoute(
                builder: (context) => const SettingsScreen()));
          }
        },
      ),
    );
  }
}

// 10. CİHAZ DETAY EKRANI (GÜNCELLENDİ)

class DeviceDetailScreen extends StatefulWidget {
  final Map<String, dynamic> device;
  final String userRole;

  const DeviceDetailScreen({
    Key? key,
    required this.device,
    required this.userRole,
  }) : super(key: key);

  @override
  _DeviceDetailScreenState createState() => _DeviceDetailScreenState();
}

class _DeviceDetailScreenState extends State<DeviceDetailScreen> {
  final ApiService _api = ApiService();
  late Map<String, dynamic> _device;
  Timer? _pollTimer;

  @override
  void initState() {
    super.initState();
    _device = Map<String, dynamic>.from(widget.device);
    _startPolling();
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    super.dispose();
  }

  void _startPolling() {
    // Immediate first check
    _fetchDeviceStatus();

    _pollTimer = Timer.periodic(const Duration(seconds: 3), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      _fetchDeviceStatus();
    });
  }

  Future<void> _fetchDeviceStatus() async {
    try {
      // Scan all hospitals to find this device by name
      final hospitals = await _api.getHospitals();
      for (var hospital in hospitals) {
        final devices = await _api.getDevices(hospital['unique_code']);
        final match = devices.firstWhere(
            (d) => d['name'] == widget.device['name'],
            orElse: () => null);

        if (match != null) {
          if (mounted) {
            setState(() {
              _device = Map<String, dynamic>.from(match);
            });
          }
          return;
        }
      }
    } catch (e) {
      debugPrint("DeviceDetail Poll Error: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    final deviceName = _device['name'] ?? 'Bilinmiyor';
    final ip = _device['ip_address'] ?? ' - ';
    final room = _device['room_number'] ?? 'Atanmamış';
    final rawStatus = _device['status'] ?? 'SAFE';
    bool isAlert = isDeviceUnderAttack(rawStatus);
    final displayStatus = rawStatus.toString().trim();

    return Scaffold(
      appBar: AppBar(
        title: Text(deviceName,
            style: const TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: darkBackground,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            // 1. Device Status Card
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: cardColor,
                borderRadius: BorderRadius.circular(15),
                border: Border.all(
                  color: isAlert ? neonRed : neonGreen.withOpacity(0.5),
                  width: 1,
                ),
                boxShadow: [
                  BoxShadow(
                    color: isAlert
                        ? neonRed.withOpacity(0.2)
                        : neonGreen.withOpacity(0.1),
                    blurRadius: 10,
                    offset: const Offset(0, 4),
                  )
                ],
              ),
              child: Column(
                children: [
                  Icon(
                    isAlert
                        ? Icons.warning_amber_rounded
                        : Icons.check_circle_outline,
                    size: 60,
                    color: isAlert ? neonRed : neonGreen,
                  ),
                  const SizedBox(height: 15),
                  Text(
                    isAlert ? 'TEHDİT: $displayStatus' : 'GÜVENLİ',
                    style: TextStyle(
                      color: isAlert ? neonRed : neonGreen,
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.2,
                    ),
                  ),
                  const SizedBox(height: 5),
                  Text(
                    isAlert
                        ? 'Cihaz şüpheli aktivite gösteriyor.'
                        : 'Cihaz normal çalışıyor.',
                    style: const TextStyle(color: textMuted),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // 2. Device Information Card (New)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: cardColor,
                borderRadius: BorderRadius.circular(15),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Cihaz Bilgileri",
                    style: TextStyle(
                        color: textLight,
                        fontSize: 18,
                        fontWeight: FontWeight.bold),
                  ),
                  const Divider(color: textMuted),
                  const SizedBox(height: 10),
                  _buildInfoRow('Cihaz Adı', deviceName),
                  _buildInfoRow('IP Adresi', ip),
                  _buildInfoRow('Oda Numarası', room),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: textMuted, fontSize: 16)),
          Text(value,
              style: const TextStyle(
                  color: textLight, fontWeight: FontWeight.bold, fontSize: 16)),
        ],
      ),
    );
  }
}

// 11. AYARLAR EKRANI

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  String _userEmail = 'Yükleniyor...';

  @override
  void initState() {
    super.initState();
    _loadUserEmail();
  }

  Future<void> _loadUserEmail() async {
    final prefs = await SharedPreferences.getInstance();
    if (mounted) {
      setState(() {
        _userEmail = prefs.getString('user_email') ?? 'Bilinmiyor';
      });
    }
  }

  Future<void> _logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.clear();
    if (mounted) {
      Navigator.of(context).pushAndRemoveUntil(
        MaterialPageRoute(builder: (context) => const LoginScreen()),
        (route) => false,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Uygulama Ayarları',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: darkBackground,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            // --- KULLANICI HESABI ---
            SettingsHeader('Kullanıcı Hesabı'),
            SettingsItem(
              icon: Icons.person_outline,
              title: 'Giriş Yapan E-posta',
              subtitle: _userEmail,
            ),
            SettingsItem(
              icon: Icons.vpn_key,
              title: 'Şifreyi Değiştir',
              onTap: () {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                      content: Text(
                          '$_userEmail adresine şifre sıfırlama bağlantısı gönderildi. (Demo)'),
                      backgroundColor: accentBlue),
                );
              },
            ),

            // --- SİSTEM GÜVENLİĞİ VE CİHAZ ---
            SettingsHeader('Sistem Güvenliği'),
            SettingsSwitchItem(
              icon: Icons.notifications_active_outlined,
              title: 'Kritik Saldırı Bildirimleri',
              subtitle: 'Anlık alarm bildirimlerini etkinleştir.',
              initialValue: true,
              onChanged: (value) {/* Bildirim ayarını kaydet */},
            ),
            SettingsSwitchItem(
              icon: Icons.model_training,
              title: 'Yapay Zeka Anomali Tespiti',
              subtitle: 'Makine öğrenimi modelini etkinleştir.',
              initialValue: true,
              onChanged: (value) {/* ML model durumunu kaydet */},
            ),

            SettingsHeader('Uygulama Bilgisi'),
            SettingsItem(
              icon: Icons.info_outline,
              title: 'Versiyon',
              subtitle: 'IoMT IDS v1.0 (CIC-IoMT 2024)',
            ),
            SettingsItem(
              icon: Icons.gavel,
              title: 'Gizlilik ve Şartlar',
              onTap: () {/* Yönlendirme eklenebilir */},
            ),

            const SizedBox(height: 50),
            Center(
              child: TextButton.icon(
                onPressed: _logout,
                icon: const Icon(Icons.logout, color: neonRed),
                label: const Text('Oturumu Kapat',
                    style:
                        TextStyle(color: neonRed, fontWeight: FontWeight.bold)),
              ),
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}

// 12. YARDIMCI WIDGETLAR (ÖN AD DÜZELTİLDİ ve Hepsi Alt Çizgisiz)

class SettingsHeader extends StatelessWidget {
  final String title;
  const SettingsHeader(this.title, {super.key});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 25, bottom: 10),
      child: Text(
        title.toUpperCase(),
        style: TextStyle(
          color: neonGreen,
          fontSize: 14,
          fontWeight: FontWeight.bold,
          letterSpacing: 1.5,
        ),
      ),
    );
  }
}

class SettingsItem extends StatelessWidget {
  final IconData icon;
  final String title;
  final String? subtitle;
  final VoidCallback? onTap;

  const SettingsItem(
      {super.key,
      required this.icon,
      required this.title,
      this.subtitle,
      this.onTap});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: cardColor,
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: ListTile(
        leading: Icon(icon, color: textMuted),
        title: Text(title, style: const TextStyle(color: textLight)),
        subtitle: subtitle != null
            ? Text(subtitle!, style: const TextStyle(color: textMuted))
            : null,
        trailing: onTap != null
            ? const Icon(Icons.arrow_forward_ios, color: textMuted, size: 16)
            : null,
        onTap: onTap,
      ),
    );
  }
}

class SettingsSwitchItem extends StatefulWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final bool initialValue;
  final ValueChanged<bool> onChanged;

  const SettingsSwitchItem({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.initialValue,
    required this.onChanged,
  });

  @override
  State<SettingsSwitchItem> createState() => _SettingsSwitchItemState();
}

class _SettingsSwitchItemState extends State<SettingsSwitchItem> {
  late bool _value;

  @override
  void initState() {
    super.initState();
    _value = widget.initialValue;
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      color: cardColor,
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: SwitchListTile(
        secondary: Icon(widget.icon, color: textMuted),
        title: Text(widget.title, style: const TextStyle(color: textLight)),
        subtitle:
            Text(widget.subtitle, style: const TextStyle(color: textMuted)),
        value: _value,
        activeColor: neonGreen,
        onChanged: (newValue) {
          setState(() {
            _value = newValue;
          });
          widget.onChanged(newValue);
        },
      ),
    );
  }
}

class StatusArea extends StatelessWidget {
  final bool isAlert;
  final AnimationController controller;

  const StatusArea(
      {super.key, required this.isAlert, required this.controller});

  @override
  Widget build(BuildContext context) {
    // Dynamic Logic
    final Color primaryColor = isAlert ? neonRed : neonGreen;
    final int score = isAlert ? 45 : 98;
    final String statusLabel = isAlert ? "TEHDİT ALGILANDI" : "SİSTEM GÜVENLİ";
    final String subLabel =
        isAlert ? "System Status: Critical" : "System Status: Secure";

    return Center(
      child: Column(
        children: [
          // GAUGE CIRCLE
          AnimatedBuilder(
            animation: controller,
            builder: (context, child) {
              return Container(
                width: 220,
                height: 220,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  // Gradient Background
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      primaryColor.withOpacity(0.15),
                      primaryColor.withOpacity(0.05),
                    ],
                  ),
                  // Animated Border
                  border: Border.all(
                    color: primaryColor
                        .withOpacity(0.6 + (0.4 * controller.value)),
                    width: 3,
                  ),
                  // Pulsing Glow
                  boxShadow: [
                    BoxShadow(
                      color: primaryColor.withOpacity(0.3 * controller.value),
                      blurRadius: 20 + (15 * controller.value),
                      spreadRadius: 2 * controller.value,
                    ),
                  ],
                ),
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Percentage Text
                      Text(
                        "$score%",
                        style: const TextStyle(
                            fontSize: 64,
                            fontWeight: FontWeight.bold,
                            color: textLight,
                            shadows: [
                              Shadow(
                                  color: Colors.black54,
                                  blurRadius: 4,
                                  offset: Offset(2, 2))
                            ]),
                      ),
                      const SizedBox(height: 5),
                      const Text(
                        "Güvenlik Skoru",
                        style: TextStyle(
                          color: textMuted,
                          fontSize: 14,
                          letterSpacing: 1.0,
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          ),

          const SizedBox(height: 25),

          // STATUS TEXT under the gauge
          Text(
            statusLabel,
            style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: primaryColor,
                letterSpacing: 1.5,
                shadows: [
                  Shadow(color: primaryColor.withOpacity(0.5), blurRadius: 10)
                ]),
          ),
          const SizedBox(height: 5),
          Text(
            subLabel,
            style: const TextStyle(
                color: textMuted, fontSize: 14, fontStyle: FontStyle.italic),
          ),
        ],
      ),
    );
  }
}

class InfoCard extends StatelessWidget {
  final String title;
  final String value;

  const InfoCard({super.key, required this.title, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: MediaQuery.of(context).size.width / 2 - 30,
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(8),
        boxShadow: const [
          BoxShadow(color: Colors.black38, blurRadius: 4, offset: Offset(0, 2))
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(title, style: const TextStyle(color: textMuted, fontSize: 14)),
          const SizedBox(height: 5),
          Text(value,
              style: const TextStyle(
                  fontSize: 24, fontWeight: FontWeight.bold, color: neonGreen)),
        ],
      ),
    );
  }
}

class DeviceItem extends StatelessWidget {
  final String name;
  final String status;
  final bool isAlert;
  final AnimationController? controller;

  const DeviceItem(
      {super.key,
      required this.name,
      required this.status,
      required this.isAlert,
      this.controller});

  @override
  Widget build(BuildContext context) {
    Color dotColor = isAlert ? neonRed : neonGreen;

    Widget dotWidget = Container(
      width: 12,
      height: 12,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: dotColor,
        boxShadow: [
          BoxShadow(
            color: dotColor,
            blurRadius: isAlert ? 10 : 8,
          ),
        ],
      ),
    );

    return Container(
      padding: const EdgeInsets.all(15),
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: <Widget>[
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Text(name,
                    style: const TextStyle(
                        fontWeight: FontWeight.bold, fontSize: 16)),
                const SizedBox(height: 4),
                Text(status,
                    style: TextStyle(
                        color: isAlert ? neonRed : textMuted, fontSize: 12)),
              ],
            ),
          ),
          dotWidget,
        ],
      ),
    );
  }
}

class AlertPopup extends StatelessWidget {
  final VoidCallback onIsolate;

  const AlertPopup({super.key, required this.onIsolate});
  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: neonRed.withOpacity(0.95),
      shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(15),
          side: const BorderSide(color: Colors.white, width: 2)),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: <Widget>[
          const Icon(Icons.error_outline, size: 60, color: Colors.white),
          const SizedBox(height: 15),
          const Text(
            'KRİTİK GÜVENLİK İHLALİ!',
            textAlign: TextAlign.center,
            style: TextStyle(
                fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white),
          ),

          // --- YENİ EKLENEN KISIM: DETAY BUTONU ---
          const SizedBox(height: 25),
          ElevatedButton(
            onPressed: () {
              // DİKKAT: Buraya Alarm Testi (Monitör) ekranının sınıf adını yazmalısın.
              // Dosyanın en tepesine o ekranı import etmeyi unutma!
              // Örnek: import 'screens/traffic_monitor_screen.dart';

              Navigator.push(
                context,
                MaterialPageRoute(
                    builder: (context) => const MonitoringScreen()),
              );
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.white, // Beyaz fon (Okunaklı olsun)
              foregroundColor: neonRed, // Kırmızı yazı
              elevation: 5,
              padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 12),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
            ),
            child: const Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.bar_chart, size: 20), // Grafik ikonu yakışır
                SizedBox(width: 8),
                Text('Saldırı Detayları',
                    style:
                        TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
              ],
            ),
          ),
          // ----------------------------------------

          const SizedBox(height: 15),

          // --- ESKİ BUTON (AYNEN DURUYOR) ---
          ElevatedButton(
            onPressed: onIsolate,
            style: ElevatedButton.styleFrom(
              backgroundColor: darkBackground,
              foregroundColor: neonRed,
              side: const BorderSide(color: neonRed, width: 2),
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
            ),
            child: const Text('Bağlantıyı Kes ve Durdur',
                style: TextStyle(fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }
}

class NavBar extends StatelessWidget {
  final VoidCallback onSimulate;
  final Function(int) onNavigate;

  const NavBar({super.key, required this.onSimulate, required this.onNavigate});

  @override
  Widget build(BuildContext context) {
    return BottomNavigationBar(
      backgroundColor: cardColor,
      selectedItemColor: neonGreen,
      unselectedItemColor: textMuted,
      items: [
        const BottomNavigationBarItem(
          icon: Icon(Icons.dashboard),
          label: 'Durum',
        ),
        const BottomNavigationBarItem(
          icon: Icon(
            Icons.warning_amber_rounded,
            color: neonRed,
          ),
          label: 'Alarm Testi',
        ),
        const BottomNavigationBarItem(
          icon: Icon(Icons.local_hospital),
          label: 'Hastaneler',
        ),
        const BottomNavigationBarItem(
          icon: Icon(Icons.list_alt),
          label: 'Olay Kaydı',
        ),
        const BottomNavigationBarItem(
          icon: Icon(Icons.devices_other),
          label: 'Cihazlar',
        ),
        const BottomNavigationBarItem(
          icon: Icon(Icons.settings),
          label: 'Ayarlar',
        ),
      ],
      currentIndex: 0,
      type: BottomNavigationBarType.fixed,
      onTap: (index) {
        if (index == 1) {
          onSimulate();
        } else {
          onNavigate(index);
        }
      },
    );
  }
}

class DetailCard extends StatelessWidget {
  final String title;
  final IconData icon;
  final Color iconColor;
  final Widget content;

  const DetailCard(
      {super.key,
      required this.title,
      required this.icon,
      required this.iconColor,
      required this.content});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: cardColor,
      margin: const EdgeInsets.only(bottom: 20),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      child: Padding(
        padding: const EdgeInsets.all(15.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, color: iconColor, size: 24),
                const SizedBox(width: 10),
                Text(
                  title,
                  style: const TextStyle(
                      color: textLight,
                      fontSize: 18,
                      fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const Divider(color: textMuted),
            const SizedBox(height: 10),
            content,
          ],
        ),
      ),
    );
  }
}

class ConfidenceChart extends StatelessWidget {
  final double score;

  const ConfidenceChart({super.key, required this.score});

  @override
  Widget build(BuildContext context) {
    Color color = score > 50 ? neonRed : neonGreen;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Tespit Güven Skoru: ${score.toStringAsFixed(1)}%',
          style: TextStyle(color: color, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 10),
        ClipRRect(
          borderRadius: BorderRadius.circular(5),
          child: LinearProgressIndicator(
            value: score / 100,
            backgroundColor: cardColor.withOpacity(0.5),
            color: color,
            minHeight: 15,
          ),
        ),
        const SizedBox(height: 5),
        Text(
          score > 50
              ? 'Yapay Zeka, bu analizin büyük olasılıkla bir saldırı olduğunu onayladı.'
              : 'Geçmiş verilerle uyumlu, risk düşük.',
          style: TextStyle(color: textMuted, fontSize: 12),
        )
      ],
    );
  }
}

class FactorBar extends StatelessWidget {
  final String name;
  final double percentage;
  final Color barColor;

  const FactorBar(
      {super.key,
      required this.name,
      required this.percentage,
      required this.barColor});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('$name (%${percentage.toStringAsFixed(0)})',
              style: const TextStyle(color: textLight)),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(3),
            child: LinearProgressIndicator(
              value: percentage / 100,
              backgroundColor: cardColor.withOpacity(0.7),
              color: barColor.withOpacity(0.9),
              minHeight: 10,
            ),
          ),
        ],
      ),
    );
  }
}

// 12. HASTANE YÖNETİMİ EKRANI (YENİ)
class HospitalManagementScreen extends StatefulWidget {
  const HospitalManagementScreen({super.key});

  @override
  State<HospitalManagementScreen> createState() =>
      _HospitalManagementScreenState();
}

class _HospitalManagementScreenState extends State<HospitalManagementScreen> {
  final ApiService _api = ApiService();
  List<dynamic> _hospitals = [];
  bool _isLoading = true;

  String? _userRole;

  @override
  void initState() {
    super.initState();
    _loadRole();
    _fetchHospitals();
  }

  Future<void> _loadRole() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _userRole = prefs.getString('user_role');
    });
  }

  Future<void> _fetchHospitals() async {
    try {
      final data = await _api.getHospitals();
      setState(() {
        _hospitals = data;
        _isLoading = false;
      });
    } catch (e) {
      if (mounted) {
        setState(() => _isLoading = false);
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Hata: $e')));
      }
    }
  }

  void _showAddHospitalDialog() {
    // ... dialog code ...
    final nameCtrl = TextEditingController();
    final codeCtrl = TextEditingController();

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: cardColor,
        title:
            const Text('Yeni Hastane Ekle', style: TextStyle(color: textLight)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: nameCtrl,
              style: const TextStyle(color: textLight),
              decoration: const InputDecoration(labelText: 'Hastane Adı'),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: codeCtrl,
              style: const TextStyle(color: textLight),
              decoration: const InputDecoration(
                  labelText: 'unique_code (örn: BURSA-01)'),
            ),
          ],
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('İptal', style: TextStyle(color: textMuted))),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: neonGreen),
            onPressed: () async {
              if (nameCtrl.text.isEmpty || codeCtrl.text.isEmpty) {
                ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                    content: Text('Alanlar boş bırakılamaz!'),
                    backgroundColor: neonRed));
                return;
              }

              try {
                await _api.createHospital(nameCtrl.text, codeCtrl.text);
                Navigator.pop(ctx);
                _fetchHospitals(); // Listeyi yenile
                ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                    content: Text('Hastane başarıyla eklendi!'),
                    backgroundColor: neonGreen));
              } catch (e) {
                Navigator.pop(ctx);
                ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                    content: Text('Hata: $e'), backgroundColor: neonRed));
              }
            },
            child: const Text('Ekle', style: TextStyle(color: darkBackground)),
          ),
        ],
      ),
    );
  }

  void _showEditHospitalDialog(Map<String, dynamic> hospital) {
    // ... edit dialog code ...
    final nameCtrl = TextEditingController(text: hospital['name']);
    final codeCtrl = TextEditingController(text: hospital['unique_code']);

    showDialog(
        context: context,
        builder: (ctx) => AlertDialog(
              backgroundColor: cardColor,
              title: const Text('Hastane Düzenle',
                  style: TextStyle(color: textLight)),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  TextField(
                    controller: nameCtrl,
                    style: const TextStyle(color: textLight),
                    decoration: const InputDecoration(labelText: 'Hastane Adı'),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: codeCtrl,
                    style: const TextStyle(color: textLight),
                    decoration: const InputDecoration(
                        labelText: 'unique_code (örn: BURSA-01)'),
                  ),
                  const SizedBox(height: 20),
                  // Delete Option
                  Align(
                    alignment: Alignment.centerLeft,
                    child: TextButton.icon(
                      onPressed: () async {
                        // Confirm delete
                        bool confirm = await showDialog(
                                context: context,
                                builder: (c) => AlertDialog(
                                        backgroundColor: cardColor,
                                        title: const Text('Emin misiniz?',
                                            style: TextStyle(color: textLight)),
                                        content: const Text(
                                            'Bu işlem geri alınamaz ve bağlı cihazları da silebilir.',
                                            style: TextStyle(color: textMuted)),
                                        actions: [
                                          TextButton(
                                              onPressed: () =>
                                                  Navigator.pop(c, false),
                                              child: const Text('İptal',
                                                  style: TextStyle(
                                                      color: textMuted))),
                                          TextButton(
                                              onPressed: () =>
                                                  Navigator.pop(c, true),
                                              child: const Text('SİL',
                                                  style: TextStyle(
                                                      color: neonRed,
                                                      fontWeight:
                                                          FontWeight.bold))),
                                        ])) ??
                            false;

                        if (confirm) {
                          try {
                            await _api.deleteHospital(hospital['id']);
                            Navigator.pop(ctx); // Close edit dialog
                            _fetchHospitals();
                            ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                    content: Text('Hastane silindi.'),
                                    backgroundColor: neonGreen));
                          } catch (e) {
                            ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                                content: Text('Hata: $e'),
                                backgroundColor: neonRed));
                          }
                        }
                      },
                      icon: const Icon(Icons.delete, color: neonRed),
                      label: const Text('Hastaneyi Sil',
                          style: TextStyle(color: neonRed)),
                    ),
                  )
                ],
              ),
              actions: [
                TextButton(
                    onPressed: () => Navigator.pop(ctx),
                    child: const Text('İptal',
                        style: TextStyle(color: textMuted))),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(backgroundColor: neonGreen),
                  onPressed: () async {
                    if (nameCtrl.text.isEmpty || codeCtrl.text.isEmpty) {
                      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                          content: Text('Alanlar boş bırakılamaz!'),
                          backgroundColor: neonRed));
                      return;
                    }
                    try {
                      await _api.updateHospital(
                          hospital['id'], nameCtrl.text, codeCtrl.text);
                      Navigator.pop(ctx);
                      _fetchHospitals();
                      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                          content: Text('Hastane güncellendi!'),
                          backgroundColor: neonGreen));
                    } catch (e) {
                      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                          content: Text('Hata: $e'), backgroundColor: neonRed));
                    }
                  },
                  child: const Text('Güncelle',
                      style: TextStyle(color: darkBackground)),
                ),
              ],
            ));
  }

  @override
  Widget build(BuildContext context) {
    bool isAdmin = _userRole == 'UserRole.ADMIN' || _userRole == 'ADMIN';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Hastane Yönetimi',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: darkBackground,
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator(color: neonGreen))
          : ListView.builder(
              padding: const EdgeInsets.all(15),
              itemCount: _hospitals.length,
              itemBuilder: (ctx, i) {
                final h = _hospitals[i];
                return Card(
                  color: cardColor,
                  margin: const EdgeInsets.only(bottom: 10),
                  child: ListTile(
                    leading: const Icon(Icons.location_city, color: accentBlue),
                    title: Text(h['name'] ?? '',
                        style: const TextStyle(
                            color: textLight, fontWeight: FontWeight.bold)),
                    subtitle: Text(h['unique_code'] ?? '',
                        style: const TextStyle(color: textMuted)),
                    trailing: isAdmin
                        ? IconButton(
                            icon: const Icon(Icons.edit, color: textMuted),
                            onPressed: () => _showEditHospitalDialog(h),
                          )
                        : null,
                  ),
                );
              },
            ),
      floatingActionButton: isAdmin
          ? FloatingActionButton(
              backgroundColor: neonGreen,
              onPressed: _showAddHospitalDialog,
              child: const Icon(Icons.add, color: darkBackground),
            )
          : null,
    );
  }
}

// 13. CİHAZ ENVANTERİ EKRANI (GÜNCELLENMİŞ - Multi-Tenant)
class DeviceInventoryScreen extends StatefulWidget {
  const DeviceInventoryScreen({super.key});

  @override
  State<DeviceInventoryScreen> createState() => _DeviceInventoryScreenState();
}

class _DeviceInventoryScreenState extends State<DeviceInventoryScreen> {
  final ApiService _api = ApiService();

  List<dynamic> _hospitals = [];
  List<dynamic> _devices = [];

  String? _selectedHospitalCode;
  bool _isLoading = false;

  String? _userRole;

  // Timer for real-time polling
  Timer? _pollTimer;

  @override
  void initState() {
    super.initState();
    _loadInitialData();
    _startPolling();
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    super.dispose();
  }

  void _startPolling() {
    _pollTimer = Timer.periodic(const Duration(seconds: 3), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      _refreshDevicesSilently();
    });
  }

  // Background refresh without loading spinner
  Future<void> _refreshDevicesSilently() async {
    if (_selectedHospitalCode == null) return;
    try {
      final devices = await _api.getDevices(_selectedHospitalCode!);
      if (mounted) {
        setState(() => _devices = devices);
      }
    } catch (e) {
      debugPrint('Silent refresh error: $e');
    }
  }

  Future<void> _loadInitialData() async {
    setState(() => _isLoading = true);
    final prefs = await SharedPreferences.getInstance();
    _userRole = prefs.getString('user_role');

    try {
      final hospitals = await _api.getHospitals();
      setState(() {
        _hospitals = hospitals;
        if (_hospitals.isNotEmpty) {
          _selectedHospitalCode = _hospitals[0]['unique_code'];
        }
      });
      if (_selectedHospitalCode != null) {
        await _fetchDevices();
      }
    } catch (e) {
      if (mounted)
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Hata: $e')));
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _fetchDevices() async {
    if (_selectedHospitalCode == null) return;
    setState(() => _isLoading = true);
    try {
      final devices = await _api.getDevices(_selectedHospitalCode!);
      setState(() => _devices = devices);
    } catch (e) {
      if (mounted)
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Cihazlar alınamadı: $e')));
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _showAddDeviceDialog() {
    // ... dialog code ...
    if (_selectedHospitalCode == null) {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Lütfen önce bir hastane seçin.')));
      return;
    }

    final nameCtrl = TextEditingController();
    final ipCtrl = TextEditingController();
    final roomCtrl = TextEditingController(); // New

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: cardColor,
        title:
            const Text('Yeni Cihaz Ekle', style: TextStyle(color: textLight)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: nameCtrl,
              style: const TextStyle(color: textLight),
              decoration: const InputDecoration(labelText: 'Cihaz Adı'),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: ipCtrl,
              style: const TextStyle(color: textLight),
              decoration: const InputDecoration(labelText: 'IP Adresi'),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: roomCtrl,
              style: const TextStyle(color: textLight),
              decoration:
                  const InputDecoration(labelText: 'Oda No (Opsiyonel)'),
            ),
            const SizedBox(height: 10),
            Text('Hastane: $_selectedHospitalCode',
                style: const TextStyle(color: neonGreen)),
          ],
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx), child: const Text('İptal')),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: neonGreen),
            onPressed: () async {
              try {
                final newDevice = await _api.createDevice(
                    nameCtrl.text,
                    ipCtrl.text,
                    roomCtrl.text.isEmpty ? null : roomCtrl.text, // Pass Room
                    _selectedHospitalCode!);

                Navigator.pop(ctx);

                if (newDevice != null) {
                  setState(() {
                    // Directly add the object WITH the ID to the list
                    _devices.add(newDevice);
                  });
                }

                ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                    content: Text('Cihaz Eklendi!'),
                    backgroundColor: neonGreen));
              } catch (e) {
                Navigator.pop(ctx);
                ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                    content: Text('Hata: $e'), backgroundColor: neonRed));
              }
            },
            child:
                const Text('Kaydet', style: TextStyle(color: darkBackground)),
          )
        ],
      ),
    );
  }

  void _navigateToDetail(BuildContext context, Map<String, dynamic> device) {
    // Pass full device object
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => DeviceDetailScreen(
          device: device,
          userRole: _userRole ?? 'Teknik Personel',
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    bool isAdmin = _userRole == 'UserRole.ADMIN' || _userRole == 'ADMIN';

    return Scaffold(
      appBar: AppBar(
        title: const Text('Cihaz Envanteri',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: darkBackground,
      ),
      body: Column(
        children: [
          // Hastane Seçimi
          Container(
            padding: const EdgeInsets.all(10),
            color: cardColor,
            child: Row(
              children: [
                const Icon(Icons.business, color: neonGreen),
                const SizedBox(width: 10),
                Expanded(
                  child: DropdownButtonHideUnderline(
                    child: DropdownButton<String>(
                      dropdownColor: cardColor,
                      value: _selectedHospitalCode,
                      hint: const Text('Hastane Seçiniz',
                          style: TextStyle(color: textMuted)),
                      icon: const Icon(Icons.arrow_drop_down, color: textLight),
                      isExpanded: true,
                      items: _hospitals.map<DropdownMenuItem<String>>((h) {
                        return DropdownMenuItem<String>(
                          value: h['unique_code'],
                          child: Text(h['name'] ?? 'Bilinmiyor',
                              style: const TextStyle(color: textLight)),
                        );
                      }).toList(),
                      onChanged: (val) {
                        setState(() => _selectedHospitalCode = val);
                        _fetchDevices();
                      },
                    ),
                  ),
                ),
              ],
            ),
          ),

          Expanded(
            child: _isLoading
                ? const Center(
                    child: CircularProgressIndicator(color: neonGreen))
                : _devices.isEmpty
                    ? const Center(
                        child: Text("Bu hastanede kayıtlı cihaz yok.",
                            style: TextStyle(color: textMuted)))
                    : ListView.builder(
                        padding: const EdgeInsets.all(15),
                        itemCount: _devices.length,
                        itemBuilder: (ctx, i) {
                          final d = _devices[i];
                          final rawStatus = d['status'] ?? 'SAFE';
                          final isAlert = isDeviceUnderAttack(rawStatus);
                          final displayStatus = rawStatus.toString().trim();
                          final deviceId = d['id'];

                          return Dismissible(
                            key: Key(deviceId.toString()),
                            direction: isAdmin
                                ? DismissDirection.endToStart
                                : DismissDirection.none,
                            background: Container(
                              alignment: Alignment.centerRight,
                              padding: const EdgeInsets.only(right: 20.0),
                              color: neonRed,
                              child:
                                  const Icon(Icons.delete, color: Colors.white),
                            ),
                            confirmDismiss: (direction) async {
                              return await showDialog(
                                context: context,
                                builder: (BuildContext context) {
                                  return AlertDialog(
                                    backgroundColor: cardColor,
                                    title: const Text("Cihazı Sil?",
                                        style: TextStyle(color: textLight)),
                                    content: const Text(
                                        "Bu cihazı silmek istediğinize emin misiniz?",
                                        style: TextStyle(color: textMuted)),
                                    actions: <Widget>[
                                      TextButton(
                                          onPressed: () =>
                                              Navigator.of(context).pop(false),
                                          child: const Text("İptal",
                                              style:
                                                  TextStyle(color: textMuted))),
                                      TextButton(
                                        onPressed: () =>
                                            Navigator.of(context).pop(true),
                                        child: const Text("SİL",
                                            style: TextStyle(
                                                color: neonRed,
                                                fontWeight: FontWeight.bold)),
                                      ),
                                    ],
                                  );
                                },
                              );
                            },
                            onDismissed: (direction) async {
                              try {
                                await _api.deleteDevice(deviceId);
                                setState(() {
                                  _devices.removeAt(i);
                                });
                                ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                        content: Text("Cihaz silindi"),
                                        backgroundColor: neonGreen));
                              } catch (e) {
                                _fetchDevices(); // Refresh list on error
                                ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(
                                        content: Text("Hata: $e"),
                                        backgroundColor: neonRed));
                              }
                            },
                            child: GestureDetector(
                              onTap: () => _navigateToDetail(context, d),
                              child: DeviceItem(
                                name: d['name'] ?? 'Bilinmeyen Cihaz',
                                status: isAlert ? displayStatus : 'SAFE',
                                isAlert: isAlert,
                              ),
                            ),
                          );
                        },
                      ),
          ),
        ],
      ),
      floatingActionButton: isAdmin
          ? FloatingActionButton(
              backgroundColor: neonGreen,
              onPressed: _showAddDeviceDialog,
              child: const Icon(Icons.add, color: darkBackground),
            )
          : null,
    );
  }
}

class ActivityLogScreen extends StatefulWidget {
  const ActivityLogScreen({Key? key}) : super(key: key);

  @override
  _ActivityLogScreenState createState() => _ActivityLogScreenState();
}

class _ActivityLogScreenState extends State<ActivityLogScreen> {
  final ApiService _api = ApiService();
  List<dynamic> _logs = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchLogs();
  }

  Future<void> _fetchLogs() async {
    try {
      final logs = await _api.fetchActivityLogs();
      if (mounted) {
        setState(() {
          _logs = logs;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isLoading = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Loglar yüklenemedi: $e')),
        );
      }
    }
  }

  Color _getLogColor(String type) {
    switch (type) {
      case 'DANGER':
        return neonRed;
      case 'WARNING':
        return Colors.orangeAccent;
      case 'SUCCESS':
        return neonGreen;
      case 'INFO':
      default:
        return neonBlue;
    }
  }

  IconData _getLogIcon(String type) {
    switch (type) {
      case 'DANGER':
        return Icons.dangerous;
      case 'WARNING':
        return Icons.warning_amber_rounded;
      case 'SUCCESS':
        return Icons.check_circle_outline;
      case 'INFO':
      default:
        return Icons.info_outline;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Aktivite Logları',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: darkBackground,
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator(color: neonGreen))
          : _logs.isEmpty
              ? const Center(
                  child: Text("Heniz bir aktivite kaydı yok.",
                      style: TextStyle(color: textMuted)))
              : ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: _logs.length,
                  itemBuilder: (context, index) {
                    final log = _logs[index];
                    final String type = log['log_type'] ?? 'INFO';
                    final Color color = _getLogColor(type);
                    final String title = log['title'] ?? 'İşlem';
                    final String desc = log['description'] ?? 'Detay yok';

                    String time = '';
                    if (log['timestamp'] != null) {
                      try {
                        final dt = DateTime.parse(log['timestamp']).toLocal();
                        time =
                            "${dt.day.toString().padLeft(2, '0')}/${dt.month.toString().padLeft(2, '0')}/${dt.year} ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}";
                      } catch (e) {
                        time = log['timestamp'];
                      }
                    }

                    return Card(
                      color: cardColor,
                      margin: const EdgeInsets.only(bottom: 12),
                      shape: RoundedRectangleBorder(
                        side:
                            BorderSide(color: color.withOpacity(0.5), width: 1),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: ListTile(
                        leading: CircleAvatar(
                          backgroundColor: color.withOpacity(0.2),
                          child: Icon(_getLogIcon(type), color: color),
                        ),
                        title: Text(title,
                            style: const TextStyle(
                                color: textLight, fontWeight: FontWeight.bold)),
                        subtitle: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const SizedBox(height: 5),
                            Text(desc,
                                style: const TextStyle(color: textMuted)),
                            const SizedBox(height: 5),
                            Text(time,
                                style: TextStyle(
                                    color: textMuted.withOpacity(0.5),
                                    fontSize: 12)),
                          ],
                        ),
                      ),
                    );
                  },
                ),
    );
  }
}

// 15. YARDIMCI XAI WIDGET'LARI (Aşama 3 Kodu - DOĞRU YER)

class ForcePlotSimulator extends StatelessWidget {
  final double baseValue;
  final double finalScore;
  final List<Map<String, dynamic>> contributions;
  final bool isAlert;

  const ForcePlotSimulator({
    super.key,
    required this.baseValue,
    required this.finalScore,
    required this.contributions,
    required this.isAlert,
  });

  @override
  Widget build(BuildContext context) {
    if (!isAlert || contributions.isEmpty) {
      return _buildNormalState();
    }

    List<Widget> plotBlocks = contributions.map<Widget>((contrib) {
      return _ForcePlotBlock(
        label: contrib['feature'] as String,
        value: contrib['value'] as double,
        isPositive: contrib['isPositive'] as bool,
      );
    }).toList();

    return Column(
      children: [
        Row(
          children: [
            Text('Temel Değer (${baseValue.toStringAsFixed(2)})',
                style: TextStyle(color: textMuted, fontSize: 12)),
            Expanded(
              child: Container(
                height: 2,
                margin: const EdgeInsets.symmetric(horizontal: 8),
                color: textMuted,
              ),
            ),
            Text('Nihai Skor (${finalScore.toStringAsFixed(2)})',
                style: TextStyle(
                    color: neonRed, fontSize: 12, fontWeight: FontWeight.bold)),
          ],
        ),
        const SizedBox(height: 15),
        Wrap(
          spacing: 2.0,
          runSpacing: 4.0,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            const Icon(Icons.show_chart, color: textMuted, size: 16),
            const SizedBox(width: 5),
            ...plotBlocks,
            const SizedBox(width: 5),
            const Icon(Icons.flag, color: neonRed, size: 16),
          ],
        ),
        const SizedBox(height: 15),
        const Text(
          'Kırmızı bloklar (örn: Packet Rate) skoru saldırı yönüne çekerken, mavi bloklar güvenli yönde tutar.',
          style: TextStyle(color: textMuted, fontSize: 12),
        ),
      ],
    );
  }

  Widget _buildNormalState() {
    return Row(
      children: [
        Icon(Icons.check_circle, color: neonGreen, size: 16),
        const SizedBox(width: 10),
        Text('Tüm özellikler beklenen aralıkta.',
            style: TextStyle(color: textMuted)),
      ],
    );
  }
}

class _ForcePlotBlock extends StatelessWidget {
  final String label;
  final double value;
  final bool isPositive;

  const _ForcePlotBlock({
    super.key,
    required this.label,
    required this.value,
    required this.isPositive,
  });

  @override
  Widget build(BuildContext context) {
    final double width = (value * 300).clamp(20.0, 100.0);

    return Tooltip(
      message: '$label (${isPositive ? '+' : ''}${value.toStringAsFixed(2)})',
      child: Container(
        width: width,
        height: 20,
        decoration: BoxDecoration(
          color: isPositive
              ? neonRed.withOpacity(0.8)
              : accentBlue.withOpacity(0.8),
          borderRadius: BorderRadius.circular(2),
        ),
        child: Center(
          child: Icon(
            isPositive ? Icons.arrow_right_alt : Icons.arrow_back,
            color: Colors.white,
            size: 14,
          ),
        ),
      ),
    );
  }
}

// 15. GERÇEK ZAMANLI İZLEME EKRANI (AI MONITOR)

class MonitoringScreen extends StatefulWidget {
  const MonitoringScreen({Key? key}) : super(key: key);

  @override
  _MonitoringScreenState createState() => _MonitoringScreenState();
}

class _MonitoringScreenState extends State<MonitoringScreen> {
  final ApiService _api = ApiService();
  WebSocket? _socket;
  Map<String, dynamic>? _currentData;
  bool _isConnected = false;
  String _statusMessage = "Sunucuya bağlanılıyor...";
  List<Map<String, dynamic>> _history = [];

  @override
  void initState() {
    super.initState();
    _connectToStream();
  }

  @override
  void dispose() {
    _socket?.close();
    super.dispose();
  }

  Future<void> _connectToStream() async {
    try {
      setState(() => _statusMessage = "Bağlantı başlatılıyor...");

      final prefs = await SharedPreferences.getInstance();
      final token = prefs.getString('access_token');

      if (token == null) {
        print("❌ HATA: Token bulunamadı!");
        setState(() => _statusMessage = "Hata: Giriş yapılmamış");
        return;
      }

      // Render adresin (Standart WSS)
      // Adresi '/ws/internal/ws' olarak güncelliyoruz (Simülasyon ile uyumlu olması için)
      // Yanlış olan (Internal) adresi sil, bunu yapıştır:
      final wsUrl = Uri.parse('wss://iomtbackend.space/ws/alerts?token=$token');

      print("🔗 Bağlanılıyor: $wsUrl");

      // Basit Bağlantı (Header yok, Port yok - Standart)
      _socket = await WebSocket.connect(wsUrl.toString());

      // Ping ayarı (Bağlantıyı canlı tutmak için)
      _socket!.pingInterval = const Duration(seconds: 10);

      print("✅ WEBSOCKET BAĞLANDI! (Nihayet!)");

      setState(() {
        _isConnected = true;
        _statusMessage = "Sistem Canlı İzleniyor...";
      });

      _socket!.listen(
        (data) {
          print("📥 VERİ: $data"); // Terminalde veri akışını göreceksin
          try {
            final decoded = jsonDecode(data);
            if (mounted) {
              setState(() {
                _currentData = decoded;
                _history.insert(0, decoded);
                if (_history.length > 10) _history.removeLast();
              });
            }
          } catch (e) {
            print("⚠️ JSON Hatası: $e");
          }
        },
        onError: (e) {
          print("❌ WebSocket Hatası: $e");
          if (mounted) setState(() => _statusMessage = "Hata: $e");
        },
        onDone: () {
          print("⚠️ Bağlantı Koptu");
          if (mounted) setState(() => _isConnected = false);
        },
      );
    } catch (e) {
      print("❌ BAĞLANTI REDDEDİLDİ: $e");
      if (mounted) {
        setState(() {
          _isConnected = false;
          _statusMessage = "Sunucu Bağlantıyı Reddetti (Yetki Yok)";
        });
      }
    }
  }

  Color _getRiskColor(double probability) {
    if (probability > 0.8) return neonRed;
    if (probability > 0.5) return Colors.orange;
    return neonGreen;
  }

  @override
  Widget build(BuildContext context) {
    // Extract Data
    final prediction = _currentData?['prediction'] ?? {};
    final bool isAttack = prediction['is_attack'] ?? false;
    final double rawProbability = (prediction['probability'] ?? 0.0).toDouble();
    final explanations = _currentData?['explanation'] as List? ?? [];
    final flowDetails = _currentData?['flow_details'] ?? {};

    // Normalize probability: if > 1, assume it's already 0-100 scale
    final double normalizedProbability =
        rawProbability > 1 ? rawProbability / 100 : rawProbability;
    final double displayPercentage =
        rawProbability > 1 ? rawProbability : rawProbability * 100;

    Color risksColor = _getRiskColor(normalizedProbability);
    String statusText = isAttack ? "KRİTİK TEHDİT ALGILANDI" : "GÜVENLİ TRAFİK";

    return Scaffold(
      appBar: AppBar(
        title: const Text("AI Saldırı Monitörü"),
        backgroundColor: darkBackground,
        actions: [
          Container(
            margin: const EdgeInsets.all(10),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: _isConnected
                  ? neonGreen.withOpacity(0.2)
                  : neonRed.withOpacity(0.2),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: _isConnected ? neonGreen : neonRed),
            ),
            child: Row(
              children: [
                Icon(_isConnected ? Icons.wifi : Icons.wifi_off,
                    size: 16, color: _isConnected ? neonGreen : neonRed),
                const SizedBox(width: 5),
                Text(_isConnected ? "CANLI" : "OFFLINE",
                    style: TextStyle(
                        fontSize: 12,
                        color: _isConnected ? neonGreen : neonRed)),
              ],
            ),
          )
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            // 1. GAUGE / SCORE CARD
            Card(
              color: cardColor,
              shape: RoundedRectangleBorder(
                  side: BorderSide(color: risksColor, width: 2),
                  borderRadius: BorderRadius.circular(16)),
              child: Padding(
                padding: const EdgeInsets.all(24.0),
                child: Column(
                  children: [
                    Stack(
                      alignment: Alignment.center,
                      children: [
                        SizedBox(
                          width: 150,
                          height: 150,
                          child: CircularProgressIndicator(
                            value: normalizedProbability.clamp(0.0, 1.0),
                            strokeWidth: 15,
                            color: risksColor,
                            backgroundColor: Colors.grey[800],
                          ),
                        ),
                        Column(
                          children: [
                            Text("${displayPercentage.toStringAsFixed(1)}%",
                                style: TextStyle(
                                    fontSize: 32,
                                    fontWeight: FontWeight.bold,
                                    color: risksColor)),
                            const Text("Güven Skoru",
                                style: TextStyle(color: textMuted)),
                          ],
                        )
                      ],
                    ),
                    const SizedBox(height: 20),
                    Text(statusText,
                        style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: risksColor,
                            shadows: [
                              Shadow(color: risksColor, blurRadius: 10)
                            ])),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 20),

            // 1.5. DEVICE INFO & ATTACK TYPE
            if (isAttack) ...[
              Row(
                children: [
                  Expanded(
                    child: Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: cardColor,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text("Hedef Cihaz",
                              style: TextStyle(color: textMuted, fontSize: 12)),
                          const SizedBox(height: 4),
                          Text(
                            _currentData?['device_name'] ?? 'Bilinmiyor',
                            style: const TextStyle(
                                fontWeight: FontWeight.bold, fontSize: 16),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: cardColor,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text("Saldırı Türü",
                              style: TextStyle(color: textMuted, fontSize: 12)),
                          const SizedBox(height: 4),
                          Text(
                            _currentData?['attack_type'] ?? 'Genel Saldırı',
                            style: const TextStyle(
                                fontWeight: FontWeight.bold,
                                fontSize: 16,
                                color: neonRed),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ],

            const SizedBox(height: 20),

            // 2. XAI EXPLANATION CHART
            if (isAttack && explanations.isNotEmpty) ...[
              const Align(
                  alignment: Alignment.centerLeft,
                  child: Text("🛑 Saldırı Nedenleri (XAI)",
                      style: TextStyle(
                          fontSize: 18, fontWeight: FontWeight.bold))),
              const SizedBox(height: 10),
              ListView.builder(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: explanations.length > 5 ? 5 : explanations.length,
                itemBuilder: (context, index) {
                  final exp = explanations[index];

                  // FIXED: Handle both 'percentage' (Azure) and 'impact_value' (Legacy)
                  // Use [0.0] as default if null.
                  var impactRaw = exp['percentage'] ?? exp['impact_value'];

                  double impact = 0.0;
                  if (impactRaw != null) {
                    if (impactRaw is num) {
                      impact = impactRaw.toDouble().abs();
                    } else if (impactRaw is String) {
                      impact = double.tryParse(impactRaw)?.abs() ?? 0.0;
                    }
                  }

                  // Handle Infinite/NaN
                  if (impact.isNaN || impact.isInfinite) impact = 0.0;

                  // Direction handling (optional, dependent on backend)
                  final bool positive =
                      true; // Simplified for now since 'direction' might be missing in new payload

                  return Container(
                    margin: const EdgeInsets.only(bottom: 8),
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                        color: cardColor,
                        borderRadius: BorderRadius.circular(8)),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(exp['name'] ?? exp['feature'] ?? 'Unknown',
                                style: const TextStyle(
                                    fontWeight: FontWeight.bold)),
                            Text(
                                impact < 0.0001 && impact > 0
                                    ? "< 0.0001"
                                    : impact.toStringAsFixed(4),
                                style: const TextStyle(
                                    fontSize: 12, color: textMuted)),
                          ],
                        ),
                        const SizedBox(height: 5),
                        // Bar
                        Row(
                          children: [
                            Expanded(
                              flex: (impact * 100).clamp(0, 100).toInt(),
                              child: Container(
                                  height: 8,
                                  color: positive ? neonRed : neonGreen),
                            ),
                            Expanded(
                              flex: 100 - (impact * 100).clamp(0, 100).toInt(),
                              child: Container(
                                  height: 8, color: Colors.transparent),
                            )
                          ],
                        )
                      ],
                    ),
                  );
                },
              ),
            ] else if (_isConnected) ...[
              const Center(
                  child: Padding(
                padding: EdgeInsets.all(20.0),
                child: Text("✅ Sistem Normal. Anormali tespit edilmedi.",
                    style: TextStyle(color: neonGreen)),
              ))
            ],

            const SizedBox(height: 20),

            // 3. FLOW DETAILS
            Theme(
              data:
                  Theme.of(context).copyWith(dividerColor: Colors.transparent),
              child: ExpansionTile(
                title: const Text("📡 Ağ Trafik Detayları"),
                children: [
                  Container(
                    height: 200,
                    decoration: BoxDecoration(
                        color: Colors.black26,
                        borderRadius: BorderRadius.circular(8)),
                    child: SingleChildScrollView(
                      padding: const EdgeInsets.all(10),
                      child: Text(jsonEncode(flowDetails),
                          style: const TextStyle(
                              fontFamily: 'Courier',
                              color: textLight,
                              fontSize: 12)),
                    ),
                  )
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
