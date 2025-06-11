import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class UserProfile extends StatefulWidget {
  const UserProfile({super.key});

  @override
  State<UserProfile> createState() => _UserProfileState();
}

class _UserProfileState extends State<UserProfile> {
  final User? user = FirebaseAuth.instance.currentUser;

  String aboutMe = '';
  String phone = '';
  String address = '';

  void _openEditModal() {
    final aboutController = TextEditingController(text: aboutMe);
    final phoneController = TextEditingController(text: phone);
    final addressController = TextEditingController(text: address);

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Padding(
        padding: MediaQuery.of(context).viewInsets,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text("Profili Düzenle", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
              const SizedBox(height: 10),
              TextField(
                controller: aboutController,
                maxLines: 2,
                decoration: const InputDecoration(labelText: "Hakkımda", border: OutlineInputBorder()),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: phoneController,
                keyboardType: TextInputType.phone,
                decoration: const InputDecoration(labelText: "Telefon", border: OutlineInputBorder()),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: addressController,
                decoration: const InputDecoration(labelText: "Adres", border: OutlineInputBorder()),
              ),
              const SizedBox(height: 20),
              Align(
                alignment: Alignment.centerRight,
                child: ElevatedButton(
                  onPressed: () {
                    if (aboutController.text.trim().isEmpty ||
                        phoneController.text.trim().isEmpty ||
                        addressController.text.trim().isEmpty) {
                      return;
                    }

                    setState(() {
                      aboutMe = aboutController.text.trim();
                      phone = phoneController.text.trim();
                      address = addressController.text.trim();
                    });

                    Navigator.pop(context);

                    // Temizleme işlemi
                    aboutController.clear();
                    phoneController.clear();
                    addressController.clear();
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF4AD5CD),
                  ),
                  child: const Text("Kaydet"),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final displayName = user?.displayName ?? 'Kullanıcı';
    final email = user?.email ?? '';
    final photoURL = user?.photoURL;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        toolbarHeight: 30,
        leading: IconButton(
          onPressed: () => Navigator.pop(context),
          icon: const Icon(Icons.arrow_back, size: 30, weight: 3),
        ),
      ),
      body: SingleChildScrollView(
        child: Container(
          color: Colors.white,
          width: double.infinity,
          padding: const EdgeInsets.all(18),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text("My Profile", style: TextStyle(fontSize: 26, fontWeight: FontWeight.bold)),
                  InkWell(
                    onTap: _openEditModal,
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 5, horizontal: 15),
                      decoration: BoxDecoration(
                        color: const Color(0xFF4AD5CD),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: const Row(
                        children: [
                          Text("Edit ", style: TextStyle(color: Colors.white, fontSize: 18)),
                          Icon(Icons.edit, color: Colors.white, size: 20),
                        ],
                      ),
                    ),
                  )
                ],
              ),
              const SizedBox(height: 20),
              Column(
                children: [
                  Center(
                    child: CircleAvatar(
                      radius: 50,
                      backgroundImage: photoURL != null ? NetworkImage(photoURL) : null,
                      child: photoURL == null
                          ? const Icon(Icons.person_outline, size: 50)
                          : null,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(displayName, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w600)),
                  Text(email, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500, color: Colors.blueGrey)),
                ],
              ),
              const SizedBox(height: 30),
              const Text("About Me", style: TextStyle(color: Colors.black, fontWeight: FontWeight.w600, fontSize: 22)),
              UserProfileWidget(opt: "Hakkımda", val: aboutMe.isNotEmpty ? aboutMe : "Henüz bir şey yazılmamış."),
              const SizedBox(height: 10),
              const Text("İletişim", style: TextStyle(color: Colors.black, fontWeight: FontWeight.w600, fontSize: 22)),
              UserProfileWidget(opt: "Telefon", val: phone.isNotEmpty ? phone : "Belirtilmemiş"),
              UserProfileWidget(opt: "Adres", val: address.isNotEmpty ? address : "Belirtilmemiş"),
            ],
          ),
        ),
      ),
    );
  }
}

class UserProfileWidget extends StatelessWidget {
  final String opt;
  final String val;

  const UserProfileWidget({required this.opt, required this.val});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 5),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(opt, style: const TextStyle(fontSize: 16, color: Colors.grey)),
          const SizedBox(height: 5),
          Text(val, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w500)),
          const Divider(thickness: 1),
        ],
      ),
    );
  }
}
