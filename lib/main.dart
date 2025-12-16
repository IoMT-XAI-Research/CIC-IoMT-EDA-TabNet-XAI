import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'firebase_options.dart';
import 'api_service.dart';
import 'dart:async';

// ANA RENKLER

const Color darkBackground = Color(0xFF121212);
const Color cardColor = Color(0xFF242424);
const Color neonGreen = Color(0xFF00FF41);
const Color neonRed = Color(0xFFFF073A);
const Color textLight = Color(0xFFE0E0E0);
const Color textMuted = Color(0xFFAAAAAA);
const Color accentBlue = Color(0xFF00BFFF);
const Color neonYellow = Color(0xFFFFD700);

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
  String? _errorText;
  bool _isLoading = false;
  bool _isPasswordVisible = false;

  Future<void> _signUp() async {
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
      final userCredential =
          await FirebaseAuth.instance.createUserWithEmailAndPassword(
        email: _emailController.text.trim(),
        password: _passwordController.text.trim(),
      );

      await userCredential.user?.sendEmailVerification();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
                'Kayıt Başarılı! Doğrulama e-postası ${userCredential.user!.email!} adresine gönderildi.'),
            backgroundColor: neonGreen,
          ),
        );
        Navigator.of(context).pop();
      }
    } on FirebaseAuthException catch (e) {
      if (mounted) {
        setState(() {
          if (e.code == 'weak-password') {
            _errorText = 'Şifre çok zayıf (min. 6 karakter).';
          } else if (e.code == 'email-already-in-use') {
            _errorText = 'Bu e-posta adresi zaten kullanımda.';
          } else if (e.code == 'invalid-email') {
            _errorText = 'Geçersiz e-posta formatı.';
          } else {
            _errorText = 'Kayıt Hatası: ${e.message}';
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
    });
  }

  // ✅ Backend'teki /devices listesini çek ve id=1'in status'una göre isAlert'i ayarla
  Future<void> _checkAttackStatus() async {
    try {
      // Demo: Fetch hospitals first, pick first one, then check devices.
      final hospitals = await _apiService.getHospitals();
      if (hospitals.isEmpty) return;

      final devices = await _apiService.getDevices(hospitals[0]['unique_code']);

      // Burada demo için id = 1 cihazı hedef kabul ediyoruz.
      dynamic target;
      try {
        target = devices.firstWhere((d) => d['id'] == 1);
      } catch (_) {
        target = null;
      }

      bool newIsAlert = false;
      if (target != null) {
        final status = (target['status'] ?? '').toString().toUpperCase();
        newIsAlert = status == 'ATTACK';
      }

      if (!mounted) return;

// Yeni helper fonksiyonu kullan
      _setAlertState(newIsAlert);
    } catch (e) {
      // Demo sırasında ortalığı kirletmemek için sadece logla
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
          deviceName: deviceName,
          isAlert: isCurrentlyAlert,
          userRole: 'Teknik Personel',
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final userEmail = FirebaseAuth.instance.currentUser?.email ?? 'Kullanıcı';
    bool isOxygenSensorAlert = isAlert;

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
            onPressed: () => FirebaseAuth.instance.signOut(),
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
                InfoCard(title: 'Toplam Cihaz', value: 'Multi'),
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
        onSimulate: () {},
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

// 10. CİHAZ DETAY VE XAI ANALİZ EKRANI (GÜNCELLENDİ - Aşama 4)

class DeviceDetailScreen extends StatefulWidget {
  final String deviceName;
  final bool isAlert;
  final String userRole;

  const DeviceDetailScreen(
      {super.key,
      required this.deviceName,
      required this.isAlert,
      required this.userRole});

  @override
  State<DeviceDetailScreen> createState() => _DeviceDetailScreenState();
}

class _DeviceDetailScreenState extends State<DeviceDetailScreen> {
  late String _currentRole;

  @override
  void initState() {
    super.initState();
    _currentRole = widget.userRole;
  }

  Map<String, String> getAdaptiveExplanation() {
    // 🔹 Teknik Personel (Daha detaylı ve teknik)
    String techReason =
        "SHAP analizi, Outbound Packet Rate ve Connection Time parametrelerindeki anormal artış nedeniyle DDoS saldırısı tespiti.";
    // 🔹 Yönetsel Personel (Daha özet ve acil)
    String execReason =
        "Oksijen Sensöründe tespit edilen olağandışı veri trafiği nedeniyle sistem güvenliği risk altındadır. Acil müdahale gerekir.";

    String summary = widget.isAlert
        ? (_currentRole == 'Teknik Personel' ? techReason : execReason)
        : 'Sistem stabil ve tüm trafik normaldir.';

    return {
      "reason": summary,
      "confidence": widget.isAlert ? '98.5%' : '0%',
    };
  }

  void _toggleRole() {
    setState(() {
      if (_currentRole == 'Teknik Personel') {
        _currentRole = 'Yönetsel Personel';
      } else {
        _currentRole = 'Teknik Personel';
      }

      // Kullanıcıya geri bildirim ver
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Açıklama rolü değişti: $_currentRole'),
          backgroundColor: accentBlue,
          duration: const Duration(seconds: 1),
        ),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    Color statusColor = widget.isAlert ? neonRed : neonGreen;
    String statusText = widget.isAlert ? 'KRİTİK ALARM' : 'NORMAL ÇALIŞIYOR';
    Map<String, String> analysis = getAdaptiveExplanation();
    double score = double.parse(analysis["confidence"]!.replaceAll('%', ''));

    return Scaffold(
      appBar: AppBar(
        title: Text('${widget.deviceName} Detay Analizi',
            style: const TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: darkBackground,
        actions: [
          Padding(
            padding: const EdgeInsets.only(right: 8.0),
            child: TextButton.icon(
              onPressed: _toggleRole,
              icon: const Icon(Icons.person_pin, size: 18, color: neonYellow),
              label: Text(
                _currentRole, // Mevcut rolü göster
                style: const TextStyle(
                    color: neonYellow, fontWeight: FontWeight.bold),
              ),
              style: TextButton.styleFrom(
                foregroundColor: neonYellow.withOpacity(0.2),
              ),
            ),
          )
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // --- CİHAZ DURUMU KARTI ---
            DetailCard(
              title: 'Cihaz Durumu',
              icon: widget.isAlert
                  ? Icons.warning_amber_rounded
                  : Icons.check_circle_outline,
              iconColor: statusColor,
              content: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    statusText,
                    style: TextStyle(
                        color: statusColor,
                        fontSize: 20,
                        fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  // 🔹 Adaptif Açıklama Metni: Bu metin butona basınca değişecek
                  Text('Analiz Özeti: ${analysis["reason"]}',
                      style: TextStyle(color: textMuted)),
                ],
              ),
            ),

            // --- XAI: GÜVEN SKORU GRAFİĞİ ---
            SettingsHeader('Yapay Zeka Analiz Raporu (SHAP / TabNet)'),
            DetailCard(
              title: 'Saldırı Güven Skoru',
              icon: Icons.score,
              iconColor: neonYellow,
              content: ConfidenceChart(score: score),
            ),

            // --- XAI: FORCE PLOT ---
            DetailCard(
              title: 'XAI - Etki Analizi (Force Plot)',
              icon: Icons.bar_chart,
              iconColor: neonYellow,
              content: ForcePlotSimulator(
                baseValue: 0.50,
                finalScore: score / 100,
                isAlert: widget.isAlert,
                contributions: widget.isAlert
                    ? [
                        {
                          'feature': 'Outbound Packet Rate',
                          'value': 0.30,
                          'isPositive': true
                        },
                        {
                          'feature': 'Connection Time',
                          'value': 0.15,
                          'isPositive': true
                        },
                        {
                          'feature': 'Normal Trafik (Negatif Etki)',
                          'value': 0.035,
                          'isPositive': false
                        },
                      ]
                    : [],
              ),
            ),

            // --- XAI: SHAPLEY DEĞERLERİ (ETKİ FAKTÖRLERİ) ---
            SettingsHeader('XAI: Etki Faktörleri (Özellik Listesi)'),

            if (widget.isAlert) ...[
              FactorBar(
                  name: 'Outbound Packet Rate (%200 Artış)',
                  percentage: 70,
                  barColor: neonRed),
              FactorBar(
                  name: 'Connection Time', percentage: 25, barColor: neonRed),
              FactorBar(
                  name: 'Kullanım Dışı Protokol Girişi',
                  percentage: 5,
                  barColor: neonRed),
            ] else ...[
              FactorBar(
                  name: 'Normal Veri Akışı',
                  percentage: 95,
                  barColor: neonGreen),
              FactorBar(name: 'Hata Oranı', percentage: 5, barColor: textMuted),
            ],

            const SizedBox(height: 30),
            // Acil Eylem Butonu
            Center(
              child: ElevatedButton.icon(
                onPressed: () {
                  Navigator.of(context).pop();
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                        content: Text(
                            '${widget.deviceName} bağlantısı kesildi ve izole edildi.'),
                        backgroundColor: neonRed),
                  );
                },
                icon: const Icon(Icons.flash_off, color: darkBackground),
                label: const Text('ACİL BAĞLANTIYI KES',
                    style: TextStyle(
                        color: darkBackground, fontWeight: FontWeight.bold)),
                style: ElevatedButton.styleFrom(
                  backgroundColor: neonRed,
                  padding:
                      const EdgeInsets.symmetric(horizontal: 25, vertical: 15),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// 11. AYARLAR EKRANI

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;
    final userEmail = user?.email ?? 'Bilinmiyor';

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
              subtitle: userEmail,
            ),
            SettingsItem(
              icon: Icons.vpn_key,
              title: 'Şifreyi Değiştir',
              onTap: () {
                FirebaseAuth.instance.sendPasswordResetEmail(email: userEmail);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                      content: Text(
                          '$userEmail adresine şifre sıfırlama bağlantısı gönderildi.'),
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
                onPressed: () => FirebaseAuth.instance.signOut(),
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
    Color color = isAlert ? neonRed : neonGreen;
    String text = isAlert ? 'SALDIRI TESPİT EDİLDİ!' : 'SİSTEM GÜVENLİ';

    return Center(
      child: Column(
        children: [
          AnimatedBuilder(
            animation: controller,
            builder: (context, child) {
              double blurRadius = isAlert ? 40 * controller.value : 25;
              double spreadRadius = isAlert ? 5 * controller.value : 0;

              return Container(
                width: 180,
                height: 180,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: color, width: 4),
                  boxShadow: [
                    BoxShadow(
                      color: color
                          .withOpacity(isAlert ? controller.value * 0.8 : 0.4),
                      blurRadius: blurRadius,
                      spreadRadius: spreadRadius,
                    ),
                  ],
                ),
                child: Center(
                  child: Text(
                    text,
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: color,
                    ),
                  ),
                ),
              );
            },
          ),
          const SizedBox(height: 10),
          const Text('Genel IoMT Güvenlik Durumu',
              style: TextStyle(color: textMuted)),
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
          const SizedBox(height: 10),
          const Text(
            'Cihaz: Oksijen Sensörü - Oda 302\nSaldırı Türü: DDoS Saldırısı (ML Tespiti)',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 16, color: Colors.white),
          ),
          const SizedBox(height: 20),
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

  @override
  void initState() {
    super.initState();
    _fetchHospitals();
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

  @override
  Widget build(BuildContext context) {
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
                  ),
                );
              },
            ),
      floatingActionButton: FloatingActionButton(
        backgroundColor: neonGreen,
        onPressed: _showAddHospitalDialog,
        child: const Icon(Icons.add, color: darkBackground),
      ),
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

  @override
  void initState() {
    super.initState();
    _loadInitialData();
  }

  Future<void> _loadInitialData() async {
    setState(() => _isLoading = true);
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
    if (_selectedHospitalCode == null) {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Lütfen önce bir hastane seçin.')));
      return;
    }

    final nameCtrl = TextEditingController();
    final ipCtrl = TextEditingController();

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
                await _api.createDevice(
                    nameCtrl.text, ipCtrl.text, _selectedHospitalCode!);
                Navigator.pop(ctx);
                _fetchDevices();
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

  void _navigateToDetail(
      BuildContext context, String deviceName, bool isCurrentlyAlert) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => DeviceDetailScreen(
          deviceName: deviceName,
          isAlert: isCurrentlyAlert,
          userRole: 'Teknik Personel',
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
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
                          final isAlert = (d['status'] == 'ATTACK');
                          return GestureDetector(
                            onTap: () => _navigateToDetail(
                                context, d['name'] ?? 'Cihaz', isAlert),
                            child: DeviceItem(
                              name: d['name'] ?? 'Bilinmeyen Cihaz',
                              status: isAlert ? 'ATTACK detected' : 'SAFE',
                              isAlert: isAlert,
                            ),
                          );
                        },
                      ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        backgroundColor: neonGreen,
        onPressed: _showAddDeviceDialog,
        child: const Icon(Icons.add, color: darkBackground),
      ),
    );
  }
}

// 14. ETKİNLİK KAYIT DEFTERİ EKRANI (YENİ EKLENDİ - Aşama 2)

class ActivityLogScreen extends StatelessWidget {
  const ActivityLogScreen({super.key});

  final List<Map<String, dynamic>> logEntries = const [
    {
      'time': '10:30',
      'type': 'KRİTİK UYARI',
      'device': 'Oksijen Sensörü - Oda 302',
      'message': 'DDoS Saldırısı Tespit Edildi (Güven Skoru: 98.5%)',
      'icon': Icons.warning_amber_rounded,
      'color': neonRed
    },
    {
      'time': '10:31',
      'type': 'MÜDAHALE',
      'device': 'Oksijen Sensörü - Oda 302',
      'message': 'Acil İzolasyon Komutu Uygulandı (Kullanıcı: Admin)',
      'icon': Icons.flash_off,
      'color': accentBlue
    },
    {
      'time': '09:00',
      'type': 'DURUM',
      'device': 'Akıllı Tansiyon Cihazı',
      'message': 'Günlük Rapor: 12 saat stabil trafik.',
      'icon': Icons.check_circle_outline,
      'color': neonGreen
    },
    {
      'time': 'Dün 14:00',
      'type': 'HAFİF UYARI',
      'device': 'İlaç Pompası - Dozaj 1',
      'message': 'Anormal DNS sorgu sayısı (Düşük Risk)',
      'icon': Icons.notification_important_outlined,
      'color': neonYellow
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Etkinlik Kayıtları',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: darkBackground,
      ),
      body: ListView.builder(
        padding: const EdgeInsets.only(top: 10.0),
        itemCount: logEntries.length,
        itemBuilder: (context, index) {
          final log = logEntries[index];
          return LogItem(log: log);
        },
      ),
    );
  }
}

class LogItem extends StatelessWidget {
  final Map<String, dynamic> log;
  const LogItem({super.key, required this.log});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: cardColor,
      margin: const EdgeInsets.symmetric(horizontal: 15, vertical: 5),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: ListTile(
        leading: Icon(log['icon'] as IconData, color: log['color'] as Color),
        title: Text(log['message'] as String,
            style: const TextStyle(color: textLight)),
        subtitle: Text(
            '${log['time']} | ${log['device']} | Tip: ${log['type']}',
            style: TextStyle(color: textMuted, fontSize: 12)),
        trailing: Text(log['time'] as String,
            style: TextStyle(
                color: log['color'] as Color, fontWeight: FontWeight.bold)),
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
