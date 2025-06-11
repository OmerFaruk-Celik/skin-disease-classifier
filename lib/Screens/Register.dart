import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';

class Register extends StatefulWidget {
  const Register({super.key});
  @override
  State<Register> createState() => _RegisterState();
}

class _RegisterState extends State<Register> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();
  final _confirmCtrl = TextEditingController();

  bool _loading = false;

  // Şifre göster/gizle için bool değişkenler
  bool _obscurePass = true;
  bool _obscureConfirm = true;

  // Güçlü şifre regex
  final _passwordPattern =
      RegExp(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[!@#\$&*~]).{8,}$');

  Future<void> register() async {
    if (!_formKey.currentState!.validate()) return;

    if (_passCtrl.text.trim() != _confirmCtrl.text.trim()) {
      _showSnack('Şifreler uyuşmuyor');
      return;
    }

    setState(() => _loading = true);
    try {
      await FirebaseAuth.instance.createUserWithEmailAndPassword(
        email: _emailCtrl.text.trim(),
        password: _passCtrl.text.trim(),
      );

      _nameCtrl.clear();
      _emailCtrl.clear();
      _passCtrl.clear();
      _confirmCtrl.clear();

      _showSnack('Kayıt başarılı!');
    } on FirebaseAuthException catch (e) {
      if (e.code == 'weak-password') {
        _showSnack('Şifre çok zayıf.');
      } else if (e.code == 'email-already-in-use') {
        _showSnack('Bu e-posta zaten kayıtlı.');
      } else {
        _showSnack('Hata: ${e.code}');
      }
    } catch (e) {
      _showSnack('Bilinmeyen hata: $e');
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        body: LayoutBuilder(
          builder: (context, constraints) => SingleChildScrollView(
            child: ConstrainedBox(
              constraints: BoxConstraints(minHeight: constraints.maxHeight),
              child: SafeArea(
                child: Column(
                  children: [
                    const SizedBox(height: 40),
                    _buildLogoSection(),
                    _buildFormSection(constraints),
                  ],
                ),
              ),
            ),
          ),
        ),
      );

  // ── Logo bölümü
  Widget _buildLogoSection() => Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: const [
              Text('Cilt ',
                  style: TextStyle(
                      fontSize: 50,
                      fontWeight: FontWeight.w900,
                      color: Color(0xFF132346))),
              Text('hastalığı',
                  style: TextStyle(
                      fontSize: 50,
                      fontWeight: FontWeight.w900,
                      color: Color(0xFF4AD5CD))),
            ],
          ),
          const SizedBox(height: 20),
          const Text('Lütfen aşağıdaki bilgileri doldur.',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w200)),
        ],
      );

  // ── Form bölümü
  Widget _buildFormSection(BoxConstraints constraints) => Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 30),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              _field('Enter your full name', _nameCtrl, TextInputType.name),
              _field('Enter Email', _emailCtrl, TextInputType.emailAddress,
                  email: true),
              _field('Enter Password', _passCtrl, TextInputType.text,
                  isPassword: true),
              _field('Confirm Password', _confirmCtrl, TextInputType.text,
                  isPassword: true, confirm: true),
              const SizedBox(height: 30),
              _registerButton(),
              _loginPrompt(),
              SizedBox(height: constraints.maxHeight * 0.1),
            ],
          ),
        ),
      );

  // ── Tek satır input widget'ı
  Widget _field(String hint, TextEditingController ctrl, TextInputType type,
      {bool isPassword = false, bool email = false, bool confirm = false}) {
    Icon? icon;

    if (email) {
      icon = const Icon(Icons.email, color: Color.fromRGBO(74, 213, 205, 1));
    } else if (isPassword) {
      icon = const Icon(Icons.lock, color: Color.fromRGBO(74, 213, 205, 1));
    } else if (confirm) {
      icon = const Icon(Icons.lock_outline, color: Color.fromRGBO(74, 213, 205, 1));
    } else {
      icon = const Icon(Icons.person, color: Color.fromRGBO(74, 213, 205, 1));
    }

    bool obscureText = false;
    if (isPassword) {
      obscureText = _obscurePass;
    } else if (confirm) {
      obscureText = _obscureConfirm;
    }

    return Container(
      decoration: BoxDecoration(
        color: const Color.fromRGBO(74, 213, 205, .1),
        borderRadius: BorderRadius.circular(100),
      ),
      margin: const EdgeInsets.symmetric(vertical: 10),
      padding: const EdgeInsets.symmetric(vertical: 7, horizontal: 25),
      child: TextFormField(
        controller: ctrl,
        obscureText: obscureText,
        keyboardType: type,
        decoration: InputDecoration(
          border: InputBorder.none,
          hintText: hint,
          prefixIcon: icon,
          suffixIcon: (isPassword || confirm)
              ? IconButton(
                  icon: Icon(
                    obscureText ? Icons.visibility_off : Icons.visibility,
                    color: const Color.fromRGBO(74, 213, 205, 1),
                  ),
                  onPressed: () {
                    setState(() {
                      if (isPassword) {
                        _obscurePass = !_obscurePass;
                      } else if (confirm) {
                        _obscureConfirm = !_obscureConfirm;
                      }
                    });
                  },
                )
              : null,
        ),
        validator: (value) {
          if (value == null || value.trim().isEmpty) {
            return 'Bu alan boş bırakılamaz';
          }
          if (email && !value.contains('@')) {
            return 'Geçerli bir e-posta girin';
          }
          if (isPassword && !_passwordPattern.hasMatch(value)) {
            return 'Güvenliğiniz için biraz \ndaha güçlü bir şifre seçin:\n'
                'Büyük harf, küçük harf,\n rakam ve özel karakter kullanın.';
          }
          if (confirm && value != _passCtrl.text) {
            return 'Şifreler uyuşmuyor';
          }
          return null;
        },
      ),
    );
  }

  // ── Kayıt butonu
  Widget _registerButton() => SizedBox(
        width: double.infinity,
        child: ElevatedButton(
          onPressed: _loading ? null : register,
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF4AD5CD),
            padding: const EdgeInsets.symmetric(vertical: 15),
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(100)),
          ),
          child: _loading
              ? const SizedBox(
                  width: 20,
                  height: 20,
                  child:
                      CircularProgressIndicator(strokeWidth: 3, color: Colors.white),
                )
              : const Text('Register',
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 22,
                      fontWeight: FontWeight.w700)),
        ),
      );

  // ── Giriş sayfasına yönlendirme
  Widget _loginPrompt() => Padding(
        padding: const EdgeInsets.symmetric(vertical: 15),
        child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          const Text('Already have an account?',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w200)),
          GestureDetector(
            onTap: () {
              Navigator.pushReplacementNamed(context, '/LogIn');
            },
            child: const Text(' Sign In',
                style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w200,
                    color: Color(0xFF4AD5CD))),
          ),
        ]),
      );

  void _showSnack(String msg) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));

  @override
  void dispose() {
    _nameCtrl.dispose();
    _emailCtrl.dispose();
    _passCtrl.dispose();
    _confirmCtrl.dispose();
    super.dispose();
  }
}
