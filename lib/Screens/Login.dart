import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class LogIn extends StatefulWidget {
  const LogIn({super.key});

  @override
  State<LogIn> createState() => _LogInState();
}

class _LogInState extends State<LogIn> {
  final _formKey  = GlobalKey<FormState>();
  final _emailCtrl = TextEditingController();
  final _passCtrl  = TextEditingController();

  bool _loading     = false;
  bool _obscurePass = true;

  

  /*────────── AUTH ──────────*/
  Future<void> _signIn() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _loading = true);

    try {
      await FirebaseAuth.instance.signInWithEmailAndPassword(
        email: _emailCtrl.text.trim(),
        password: _passCtrl.text.trim(),
      );
      _showSnack('Login successful!');
      // Giriş başarılı → Home sayfasına
      Navigator.pushReplacementNamed(context, '/HomePage');
      _emailCtrl.clear();
      _passCtrl.clear();
    } on FirebaseAuthException catch (e) {
      final msg = switch (e.code) {
        'user-not-found'      => 'No user found for that email.',
        'wrong-password'      => 'Wrong password.',
        'invalid-credential'  => 'Invalid credentials.',
        _                    => 'Error: ${e.message}',
      };
      _showSnack(msg);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _resetPassword() async {
    final email = _emailCtrl.text.trim();
    if (email.isEmpty) {
      _showSnack('Please enter your email first.');
      return;
    }
    try {
      await FirebaseAuth.instance.sendPasswordResetEmail(email: email);
      _showSnack('Password reset link sent.');
    } catch (e) {
      _showSnack('Failed: $e');
    }
  }

  void _showSnack(String msg) =>
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));



   @override
  void dispose() {
    _emailCtrl.dispose();
    _passCtrl.dispose();
    super.dispose();
  }
  

  /*────────── UI ──────────*/
  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: EdgeInsets.only(top: size.height * .12),
          child: Column(
            children: [
              _buildTitleSection(),
              _buildFormSection(),
              _buildForgotPassword(),
              _buildLoginButton(),
              const _OrDivider(),
              _buildGoogleButton(),
              _buildSignUpRedirect(),
            ],
          ),
        ),
      ),
    );
  }

  /*───────── Widgets ─────────*/
  Widget _buildTitleSection() => Column(
        children: const [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('Cilt ',
                  style: TextStyle(
                      fontSize: 50, fontWeight: FontWeight.w900,
                      color: Color(0xFF132346))),
              Text('hastalığı',
                  style: TextStyle(
                      fontSize: 50, fontWeight: FontWeight.w900,
                      color: Color(0xFF4AD5CD))),
            ],
          ),
          SizedBox(height: 40),
          Text('Lütfen e-posta ve şifrenizi yazınız.',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.w200)),
          SizedBox(height: 30),
        ],
      );

  Widget _buildFormSection() => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 30),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              _inputField(
                controller: _emailCtrl,
                hint: 'Enter Email',
                keyboard: TextInputType.emailAddress,
                isPassword: false,
                validator: (v) =>
                    v == null || v.isEmpty || !v.contains('@')
                        ? 'Valid email required' : null,
              ),
              _inputField(
                controller: _passCtrl,
                hint: 'Enter Password',
                keyboard: TextInputType.text,
                isPassword: true,
                validator: (v) =>
                    v == null || v.isEmpty ? 'Password required' : null,
              ),
            ],
          ),
        ),
      );

  Widget _inputField({
    required TextEditingController controller,
    required String hint,
    required TextInputType keyboard,
    required bool isPassword,
    required String? Function(String?) validator,
  }) =>
      Container(
        margin: const EdgeInsets.symmetric(vertical: 10),
        padding: const EdgeInsets.symmetric(vertical: 7, horizontal: 10),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(.8),
          borderRadius: BorderRadius.circular(12),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(.2),
              blurRadius: 6,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: TextFormField(
          controller: controller,
          obscureText: isPassword ? _obscurePass : false,
          keyboardType: keyboard,
          validator: validator,
          decoration: InputDecoration(
            border: InputBorder.none,
            hintText: hint,
            prefixIcon: Icon(
              isPassword ? Icons.lock : Icons.email,
              color: const Color(0xFF4AD5CD),
            ),
            suffixIcon: isPassword
                ? IconButton(
                    icon: Icon(
                      _obscurePass ? Icons.visibility_off : Icons.visibility,
                      color: const Color(0xFF4AD5CD),
                    ),
                    onPressed: () =>
                        setState(() => _obscurePass = !_obscurePass),
                  )
                : null,
          ),
        ),
      );

  Widget _buildForgotPassword() => TextButton(
        onPressed: _resetPassword,
        child: const Text('Forget Password',
            style: TextStyle(color: Colors.grey, fontSize: 16)),
      );

  Widget _buildLoginButton() => Padding(
        padding: const EdgeInsets.symmetric(horizontal: 30),
        child: ElevatedButton(
          onPressed: _loading ? null : _signIn,
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF4AD5CD),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(100),
            ),
            padding: const EdgeInsets.symmetric(vertical: 15, horizontal: 25),
            minimumSize: const Size(double.infinity, 50),
          ),
          child: _loading
              ? const CircularProgressIndicator(color: Colors.white)
              : const Text('Log In',
                  style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w700,
                      fontSize: 22)),
        ),
      );

  Widget _buildGoogleButton() => Container(
        margin: const EdgeInsets.symmetric(vertical: 10, horizontal: 30),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(100),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(.1),
              blurRadius: 6,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: InkWell(
          borderRadius: BorderRadius.circular(100),
          onTap: () {
            // Google ile giriş işlemi burada yapılacak
            _showSnack('Google ile giriş yakında!');
          },
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 25),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Image.asset('assets/images/google_symbol.png',
                    height: 24, width: 24),
                const SizedBox(width: 10),
                const Text('Continue with Google',
                    style: TextStyle(
                        color: Colors.black54,
                        fontWeight: FontWeight.w600,
                        fontSize: 16)),
              ],
            ),
          ),
        ),
      );

  Widget _buildSignUpRedirect() => Padding(
        padding: const EdgeInsets.symmetric(vertical: 15),
        child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          const Text("Don't have an account?",
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.w200)),
          GestureDetector(
            onTap: () => Navigator.pushReplacementNamed(context, '/Register'),
            child: const Text(' Sign Up',
                style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w200,
                    color: Color(0xFF4AD5CD))),
          ),
        ]),
      );
}

/*────────── Bölücü ─────────*/
class _OrDivider extends StatelessWidget {
  const _OrDivider();

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 10),
        child: Row(children: const [
          Expanded(child: Divider(color: Colors.grey, thickness: .5, indent: 40, endIndent: 10)),
          Text('OR',
              style: TextStyle(color: Colors.grey, fontWeight: FontWeight.bold)),
          Expanded(child: Divider(color: Colors.grey, thickness: .5, indent: 10, endIndent: 40)),
        ]));
}
