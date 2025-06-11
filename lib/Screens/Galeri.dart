import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class GaleriScanUI extends StatefulWidget {
  const GaleriScanUI({super.key});

  @override
  State<GaleriScanUI> createState() => _GaleriScanUIState();
}

class _GaleriScanUIState extends State<GaleriScanUI> {
  final ImagePicker _picker = ImagePicker();
  File? _capturedImage;
  bool _isLoading = false;

  Future<void> _pickImageFromGallery() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() => _capturedImage = File(image.path));
    }
  }

  Future<Map<String, dynamic>> _uploadAndPredict(File image) async {
    final uploadUri = Uri.parse('https://omerfarukcelik.duckdns.org:5001/images');

    final request = http.MultipartRequest('POST', uploadUri)
      ..files.add(await http.MultipartFile.fromPath('file', image.path));

    final uploadResponse = await request.send();

    // 200, 201, 202 gibi başarılı kodlar kabul edilir
    if (uploadResponse.statusCode < 200 || uploadResponse.statusCode >= 300) {
      final body = await uploadResponse.stream.bytesToString();
      throw Exception('Yükleme başarısız: $body');
    }

    final predictUri = Uri.parse('https://omerfarukcelik.duckdns.org:5001/tahmin');
    final predictResponse = await http.get(predictUri);

    if (predictResponse.statusCode != 200) {
      throw Exception('Tahmin alınamadı: ${predictResponse.statusCode} – ${predictResponse.body}');
    }

    return jsonDecode(predictResponse.body) as Map<String, dynamic>;
  }

  void _handlePredictButton() async {
    if (_capturedImage == null) {
      _showSnackBar('Lütfen önce bir fotoğraf seçin.');
      return;
    }

    setState(() => _isLoading = true);

    try {
      final result = await _uploadAndPredict(_capturedImage!);

      if (!mounted) return;
      Navigator.pushNamed(
        context,
        '/ResultScreen',
        arguments: result,
      );
    } catch (e) {
      _showSnackBar('Hata: $e');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _showSnackBar(String message, {Color color = Colors.redAccent}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: color,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;

    return Stack(
      children: [
        Scaffold(
          backgroundColor: Colors.black,
          appBar: AppBar(
            backgroundColor: const Color(0xFF272727),
            elevation: 0,
            centerTitle: true,
            leading: const BackButton(color: Color(0xFFBCBCBC)),
            title: const Text(
              "Galeriden Yükle",
              style: TextStyle(color: Color(0xFFBCBCBC), fontSize: 24),
            ),
          ),
          body: Stack(
            children: [
              Positioned(
                top: size.height * 0.12,
                left: (size.width - size.width * 0.95) / 2,
                child: Container(
                  width: size.width * 0.95,
                  height: size.width * 0.95,
                  decoration: const BoxDecoration(
                    color: Color(0xFF383636),
                    shape: BoxShape.circle,
                  ),
                  child: ClipOval(
                    child: _capturedImage == null
                        ? const Center(
                            child: Icon(Icons.photo_library_outlined,
                                color: Colors.white24, size: 80),
                          )
                        : Image.file(
                            _capturedImage!,
                            fit: BoxFit.cover,
                            width: double.infinity,
                            height: double.infinity,
                          ),
                  ),
                ),
              ),
              Align(
                alignment: Alignment.bottomCenter,
                child: Container(
                  height: size.height * 0.17,
                  decoration: const BoxDecoration(
                    color: Color(0xFF272727),
                    borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      IconButton(
                        icon: const Icon(Icons.send, color: Colors.white, size: 40),
                        onPressed: _isLoading ? null : _handlePredictButton,
                      ),
                      Opacity(
                        opacity: _isLoading ? 0.5 : 1.0,
                        child: InkWell(
                          onTap: _isLoading ? null : _pickImageFromGallery,
                          child: Container(
                            height: 80,
                            width: 80,
                            decoration: const BoxDecoration(
                              color: Color(0xFF4AD5CD),
                              shape: BoxShape.circle,
                            ),
                            child: Center(
                              child: Container(
                                height: 45,
                                width: 45,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  border: Border.all(color: Colors.white, width: 5),
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                      IconButton(
                        icon: const Icon(Icons.delete, color: Colors.white, size: 40),
                        onPressed: _isLoading
                            ? null
                            : () {
                                if (_capturedImage == null) {
                                  _showSnackBar('Silinecek bir fotoğraf bulunamadı.',
                                      color: Colors.orangeAccent);
                                } else {
                                  setState(() => _capturedImage = null);
                                  _showSnackBar('Fotoğraf kaldırıldı.', color: Colors.grey);
                                }
                              },
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
        if (_isLoading)
          Container(
            color: Colors.black54,
            child: const Center(child: CircularProgressIndicator()),
          ),
      ],
    );
  }
}
