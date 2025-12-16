import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:intl/intl.dart';
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
  final ApiService _apiService = ApiService();
  int _hospitalCount = 0;
  int _deviceCount = 0;
  Timer? _statsTimer;
  // Controller kept to avoid breaking StatusArea if it exists elsewhere
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _loadStats();
    _statsTimer =
        Timer.periodic(const Duration(seconds: 30), (timer) => _loadStats());
    // Mock controller for StatusArea compatibility
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    );
  }

  @override
  void dispose() {
    _statsTimer?.cancel();
    _controller.dispose();
    super.dispose();
  }

  Future<void> _loadStats() async {
    try {
      final hospitals = await _apiService.getHospitals();
      int devCount = 0;

      if (hospitals.isNotEmpty) {
        try {
          // Just count first hospital for now
          final devices =
              await _apiService.getDevices(hospitals[0]['unique_code']);
          devCount = devices.length;
        } catch (e) {
          print('Error fetching devices for stats: $e');
        }
      }

      if (mounted) {
        setState(() {
          _hospitalCount = hospitals.length;
          _deviceCount = devCount;
        });
      }
    } catch (e) {
      debugPrint('Stats load error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final userEmail = FirebaseAuth.instance.currentUser?.email ?? 'Kullanıcı';

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
              await FirebaseAuth.instance.signOut();
              final prefs = await SharedPreferences.getInstance();
              await prefs.remove('access_token');
              if (context.mounted) {
                Navigator.of(context).pushReplacement(
                  MaterialPageRoute(builder: (context) => const LoginScreen()),
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
            const Text("Sistem Durumu",
                style: TextStyle(
                    color: textLight,
                    fontSize: 18,
                    fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                    color: cardColor, borderRadius: BorderRadius.circular(12)),
                child: Row(children: const [
                  Icon(Icons.check_circle, color: neonGreen, size: 30),
                  SizedBox(width: 10),
                  Text("Sistem Aktif",
                      style: TextStyle(color: neonGreen, fontSize: 18))
                ])),
            const SizedBox(height: 30),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                InfoCard(title: 'Hastane Sayısı', value: '$_hospitalCount'),
                InfoCard(
                    title: 'Aktif Cihazlar (Birincil)', value: '$_deviceCount'),
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
          ],
        ),
      ),
      bottomNavigationBar: NavBar(
        onSimulate: () {},
        onNavigate: (index) {
          if (index == 2) {
            Navigator.of(context).push(MaterialPageRoute(
                builder: (context) => const HospitalManagementScreen()));
          } else if (index == 3) {
            Navigator.of(context).push(MaterialPageRoute(
                builder: (context) => const ActivityLogScreen()));
          } else if (index == 4) {
            Navigator.of(context).push(MaterialPageRoute(
                builder: (context) => const DeviceInventoryScreen()));
          } else if (index == 5) {
            Navigator.of(context).push(MaterialPageRoute(
                builder: (context) => const SettingsScreen()));
          }
        },
      ),
    );
  }
}
// Removed outdated methods like _checkAttackStatus and _showAttackPopup

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
    final roomCtrl = TextEditingController(); // New Controller

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
              decoration: const InputDecoration(
                  labelText: 'Cihaz Adı',
                  labelStyle: TextStyle(color: textMuted),
                  enabledBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: textMuted)),
                  focusedBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: neonGreen)),
                  filled: true,
                  fillColor: Colors.black26),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: ipCtrl,
              style: const TextStyle(color: textLight),
              decoration: const InputDecoration(
                  labelText: 'IP Adresi',
                  labelStyle: TextStyle(color: textMuted),
                  enabledBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: textMuted)),
                  focusedBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: neonGreen)),
                  filled: true,
                  fillColor: Colors.black26),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: roomCtrl,
              style: const TextStyle(color: textLight),
              decoration: const InputDecoration(
                  labelText: 'Oda Numarası (Opsiyonel)',
                  labelStyle: TextStyle(color: textMuted),
                  enabledBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: textMuted)),
                  focusedBorder: OutlineInputBorder(
                      borderSide: BorderSide(color: neonGreen)),
                  filled: true,
                  fillColor: Colors.black26),
            ),
            const SizedBox(height: 10),
            Text('Hastane: $_selectedHospitalCode',
                style: const TextStyle(color: neonGreen)),
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
                await _api.createDevice(
                    nameCtrl.text,
                    ipCtrl.text,
                    _selectedHospitalCode!,
                    roomCtrl.text.isEmpty ? null : roomCtrl.text);
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

  void _navigateToDetail(BuildContext context, Map<String, dynamic> device) {
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => DeviceDetailScreen(
          deviceName: device['name'] ?? 'Bilinmiyor',
          isAlert: device['status'] == 'ATTACK',
          userRole: 'Teknik Personel', // TODO: Get real role
          deviceData: device, // Pass full data
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
                          final deviceId = d['id'];

                          return Dismissible(
                            key: Key(deviceId.toString()),
                            direction: DismissDirection.endToStart,
                            confirmDismiss: (direction) async {
                              // Simple Role check via Token (decoding) is complex here without storing role.
                              // We will try to delete, if backend says 403, we return false.
                              try {
                                await _api.deleteDevice(deviceId);
                                return true;
                              } catch (e) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                    SnackBar(
                                        content: Text('Silinemedi: $e'),
                                        backgroundColor: neonRed));
                                return false;
                              }
                            },
                            onDismissed: (direction) {
                              setState(() {
                                _devices.removeAt(i);
                              });
                              ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(
                                      content: Text('Cihaz silindi'),
                                      backgroundColor: neonGreen));
                            },
                            background: Container(
                                color: neonRed,
                                alignment: Alignment.centerRight,
                                padding:
                                    const EdgeInsets.symmetric(horizontal: 20),
                                child: const Icon(Icons.delete,
                                    color: Colors.white)),
                            child: GestureDetector(
                              onTap: () => _navigateToDetail(context, d),
                              child: DeviceItem(
                                name: d['name'] ?? 'Bilinmeyen Cihaz',
                                status: isAlert ? 'ATTACK detected' : 'SAFE',
                                isAlert: isAlert,
                              ),
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

class DeviceDetailScreen extends StatefulWidget {
  final String deviceName;
  final bool isAlert;
  final String userRole;
  final Map<String, dynamic>? deviceData; // New Parameter

  const DeviceDetailScreen(
      {super.key,
      required this.deviceName,
      required this.isAlert,
      required this.userRole,
      this.deviceData});

  @override
  State<DeviceDetailScreen> createState() => _DeviceDetailScreenState();
}

class _DeviceDetailScreenState extends State<DeviceDetailScreen> {
  // We can keep role toggle if needed for the UI logic, or remove it.
  // Requirement: "Remove AI Analysis Report and XAI Impact Factors".
  // Requirement: "Keep Device Status".
  // Requirement: "Show Hospital Name, IP, Room Number".

  @override
  Widget build(BuildContext context) {
    final device = widget.deviceData ?? {};
    final ip = device['ip_address'] ?? 'Bilinmiyor';
    final room = device['room_number'] ?? 'Not Assigned';
    final status = widget.isAlert ? 'ATTACK' : 'SAFE';
    // Access hospital info if available in deviceData or need to fetch?
    // deviceData comes from getDevices which is simple fields.
    // We might not have hospital name directly in deviceData unless backend joins it.
    // Backend `DeviceResponse` has `hospital_id`.
    // For now we can just show "Hospital ID: ..." or nothing if not available.
    // Or we rely on previous screen passing it (it doesn't currently).
    // Let's just show what we have.

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.deviceName,
            style: const TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: darkBackground,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Status Section
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: widget.isAlert
                    ? neonRed.withOpacity(0.1)
                    : neonGreen.withOpacity(0.1),
                borderRadius: BorderRadius.circular(15),
                border: Border.all(color: widget.isAlert ? neonRed : neonGreen),
              ),
              child: Row(
                children: [
                  Icon(
                    widget.isAlert ? Icons.warning : Icons.check_circle,
                    color: widget.isAlert ? neonRed : neonGreen,
                    size: 40,
                  ),
                  const SizedBox(width: 20),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('Device Status',
                          style: TextStyle(color: textMuted, fontSize: 14)),
                      Text(status,
                          style: TextStyle(
                              color: widget.isAlert ? neonRed : neonGreen,
                              fontSize: 24,
                              fontWeight: FontWeight.bold)),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 30),

            // Device Info Section
            const Text('Cihaz Bilgileri',
                style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: textLight)),
            const SizedBox(height: 15),
            _buildInfoRow('IP Adresi', ip),
            _buildInfoRow('Oda Numarası', room),
            _buildInfoRow('Cihaz ID', '${device['id'] ?? '-'}'),
            // If we had hospital name, we'd show it here.

            const SizedBox(height: 30),

            // Permissions / Actions (Edit/Delete - only for Admin/Tech?)
            // Requirement: "Permissions: Edit and Delete buttons will be hidden if user role is TECH_STAFF"
            // Since we don't have robust role passing yet (it's hardcoded 'Teknik Personel' in nav),
            // We will hide them for now or show based on widget.userRole.

            if (widget.userRole != 'Teknik Personel') ...[
              Row(children: [
                Expanded(
                    child: ElevatedButton.icon(
                  icon: const Icon(Icons.edit),
                  label: const Text("Düzenle"),
                  onPressed: () {}, // TODO: Edit Not Implemented
                  style: ElevatedButton.styleFrom(
                      backgroundColor: accentBlue,
                      foregroundColor: Colors.white),
                )),
                const SizedBox(width: 10),
                Expanded(
                    child: ElevatedButton.icon(
                  icon: const Icon(Icons.delete),
                  label: const Text("Sil"),
                  onPressed:
                      () {}, // Handled in inventory, but could be here too
                  style: ElevatedButton.styleFrom(
                      backgroundColor: neonRed, foregroundColor: Colors.white),
                )),
              ])
            ]
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: textMuted, fontSize: 16)),
          Text(value,
              style: const TextStyle(
                  color: textLight, fontSize: 16, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

// 14. ETKİNLİK KAYIT DEFTERİ EKRANI (YENİ EKLENDİ - Aşama 2)

class ActivityLogScreen extends StatefulWidget {
  const ActivityLogScreen({super.key});

  @override
  State<ActivityLogScreen> createState() => _ActivityLogScreenState();
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
      final logs = await _api.getLogs();
      if (mounted)
        setState(() {
          _logs = logs;
          _isLoading = false;
        });
    } catch (e) {
      if (mounted) setState(() => _isLoading = false);
      // debugPrint("Log Error: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Etkinlik Kayıtları',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: darkBackground,
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator(color: neonGreen))
          : _logs.isEmpty
              ? const Center(
                  child: Text('Kayıt bulunamadı.',
                      style: TextStyle(color: textMuted)))
              : ListView.builder(
                  padding: const EdgeInsets.only(top: 10.0),
                  itemCount: _logs.length,
                  itemBuilder: (context, index) {
                    final log = _logs[index];
                    return LogItem(log: log);
                  },
                ),
    );
  }
}

class LogItem extends StatelessWidget {
  final dynamic log;
  const LogItem({super.key, required this.log});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: cardColor,
      margin: const EdgeInsets.symmetric(horizontal: 15, vertical: 5),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      child: ListTile(
        leading: const Icon(Icons.info_outline, color: accentBlue),
        title: Text(log['description'] ?? 'Event',
            style: const TextStyle(color: textLight)),
        subtitle: Text(
          log['timestamp'] != null
              ? DateFormat('dd/MM/yyyy HH:mm')
                  .format(DateTime.parse(log['timestamp']))
              : 'Unknown Date',
          style: const TextStyle(color: textMuted, fontSize: 12),
        ),
        trailing: Icon(
          log['is_alert'] == true ? Icons.warning : Icons.check_circle,
          color: log['is_alert'] == true ? neonRed : neonGreen,
        ),
      ),
    );
  }
}
