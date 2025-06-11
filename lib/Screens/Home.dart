import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;

    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text('Hoş geldiniz'),
        centerTitle: true,
        actions: [
          IconButton(
            tooltip: 'Çıkış Yap',
            icon: const Icon(Icons.logout_rounded),
            onPressed: () async {
              await FirebaseAuth.instance.signOut();
              if (context.mounted) {
                Navigator.pushNamedAndRemoveUntil(
                  context,
                  '/LogIn',
                  (_) => false,
                );
              }
            },
          ),
        ],
      ),
      body: SafeArea(
        child: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [Color(0xFFEBFDFC), Color(0xFFF8FFFF)],
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
            ),
          ),
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              _AccountCard(user: user),
              const SizedBox(height: 40),
              const Text(
                'Hoş geldiniz!\nCilt hastalığı tespiti yapmaya başlayın.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              const Spacer(),
              CustomActionButton(
                label: 'Kamerayı Aç',
                icon: Icons.camera_alt_outlined,
                onPressed: () => Navigator.pushNamed(context, '/CameraScanUI'),
              ),
              const SizedBox(height: 14),
              CustomActionButton(
                label: 'Galeriden Fotoğraf Seç',
                icon: Icons.photo_library_outlined,
                isPrimary: false,
                onPressed: () => Navigator.pushNamed(context, '/GaleriScanUI'),
              ),
              const Spacer(flex: 2),
            ],
          ),
        ),
      ),
    );
  }
}

/*–––– Alt bileşenler ––––*/

class _AccountCard extends StatelessWidget {
  const _AccountCard({required this.user});
  final User? user;

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      elevation: 4,
      child: ListTile(
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
        leading: CircleAvatar(
          radius: 28,
          backgroundImage:
              user?.photoURL != null ? NetworkImage(user!.photoURL!) : null,
          child: user?.photoURL == null
              ? const Icon(Icons.person_outline, size: 32)
              : null,
        ),
                    title: Text(
                (user?.displayName != null && user!.displayName!.trim().isNotEmpty)
                    ? user!.displayName!
                    : 'Kullanıcı',
                style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 18),
              ),
        subtitle: Row(
          children: [
            const Icon(Icons.email_outlined, size: 16, color: Colors.grey),
            const SizedBox(width: 4),
            Expanded(
              child: Text(
                user?.email ?? '',
                style: const TextStyle(color: Colors.grey),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
        trailing: IconButton(
          tooltip: 'Profili Gör',
          icon: const Icon(Icons.edit, color: Color(0xFF4AD5CD)),
          onPressed: () => Navigator.pushNamed(context, '/UserProfile'),
        ),
      ),
    );
  }
}

class CustomActionButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final VoidCallback onPressed;
  final bool isPrimary;

  const CustomActionButton({
    required this.label,
    required this.icon,
    required this.onPressed,
    this.isPrimary = true,
  });

  @override
  Widget build(BuildContext context) {
    final style = isPrimary
        ? ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF4AD5CD),
            minimumSize: const Size(double.infinity, 58),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(14),
            ),
          )
        : OutlinedButton.styleFrom(
            minimumSize: const Size(double.infinity, 58),
            side: const BorderSide(color: Color(0xFF4AD5CD), width: 2),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(14),
            ),
          );

    return isPrimary
        ? ElevatedButton.icon(
            onPressed: onPressed,
            icon: Icon(icon, size: 28),
            label: Text(label, style: const TextStyle(fontSize: 18)),
            style: style,
          )
        : OutlinedButton.icon(
            onPressed: onPressed,
            icon: Icon(icon, size: 28),
            label: Text(label, style: const TextStyle(fontSize: 18)),
            style: style,
          );
  }
}
