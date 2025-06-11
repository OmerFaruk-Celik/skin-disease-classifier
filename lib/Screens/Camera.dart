import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;


class CameraScanUI extends StatefulWidget {
  const CameraScanUI({super.key});

  @override
  State<CameraScanUI> createState() => _CameraScanUIState();
}

class _CameraScanUIState extends State<CameraScanUI> {
  final ImagePicker _picker = ImagePicker();
  File? _capturedImage;
  bool _isLoading = false;

  //-----------------------------
  // FOTOĞRAF ÇEK (izin kontrolü olmadan, sadece image_picker)
  //-----------------------------
  Future<void> _takePhoto() async {
    try {
      final XFile? image = await _picker.pickImage(source: ImageSource.camera);
      if (image != null) {
        setState(() => _capturedImage = File(image.path));
      }
    } catch (e) {
      _showSnackBar('Kamera açılırken bir hata oluştu. Lütfen tekrar deneyin.');
    }
  }

  //-----------------------------
  // FOTOĞRAFI YÜKLE + TAHMİN AL
  //-----------------------------
  Future<Map<String, dynamic>> _uploadAndPredict(File image) async {
    //------------------------------------------------------------------
    // 1) Fotoğrafı POST /images ile sunucuya gönder
    //------------------------------------------------------------------
    final uploadUri = Uri.parse('https://omerfarukcelik.duckdns.org:5001/images');

    final request = http.MultipartRequest('POST', uploadUri)
      ..files.add(await http.MultipartFile.fromPath('file', image.path));

    final uploadResponse = await request.send();

    // 200, 201, 202 gibi başarılı kodlar kabul edilir
    if (uploadResponse.statusCode < 200 || uploadResponse.statusCode >= 300) {
      final body = await uploadResponse.stream.bytesToString();
      throw Exception('Yükleme başarısız: $body');
    }

    //------------------------------------------------------------------
    // 2) GET /tahmin ile tahmin iste
    //------------------------------------------------------------------
    final predictUri = Uri.parse('https://omerfarukcelik.duckdns.org:5001/tahmin');
    final predictResponse = await http.get(predictUri);

    if (predictResponse.statusCode != 200) {
      throw Exception(
          'Tahmin alınamadı: ${predictResponse.statusCode} – ${predictResponse.body}');
    }

    //------------------------------------------------------------------
    // 3) JSON cevabı Map<String, dynamic> olarak döndür
    //------------------------------------------------------------------
    return jsonDecode(predictResponse.body) as Map<String, dynamic>;
  }

  //-----------------------------
  // GÖNDER BUTONUNU YÖNET
  //-----------------------------
  void _handlePredictButton() async {
    if (_capturedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Lütfen önce bir fotoğraf seçin.'),
          backgroundColor: Colors.redAccent,
          behavior: SnackBarBehavior.floating,
        ),
      );
      return;
    }

    setState(() => _isLoading = true);

    try {
      final result = await _uploadAndPredict(_capturedImage!);

      // Tahmin ekranına geçiş
      if (!mounted) return;
      Navigator.pushNamed(
        context,
        '/ResultScreen',
        arguments: result, // Örn. {"hasDisease":true, "diseaseName":"Egzama", ...}
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Hata: $e'),
            backgroundColor: Colors.redAccent,
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  // SnackBar gösterimi için yardımcı fonksiyon
  void _showSnackBar(String message, {Color color = Colors.redAccent}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: color,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  //-----------------------------
  // WIDGET AĞACI
  //-----------------------------
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
              'Fotoğraf Çek',
              style: TextStyle(color: Color(0xFFBCBCBC), fontSize: 24),
            ),
          ),
          body: Stack(
            children: [
              // Fotoğraf önizlemesi
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
                            child:
                                Icon(Icons.camera_alt, color: Colors.white24, size: 80),
                          )
                        : Image.file(_capturedImage!,
                            fit: BoxFit.cover,
                            width: double.infinity,
                            height: double.infinity,
                          ),
                  ),
                ),
              ),

              // Alt Menü
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
                      // Gönder (tahmin) butonu
                      IconButton(
                        icon: const Icon(Icons.send, color: Colors.white, size: 40),
                        onPressed: _isLoading ? null : _handlePredictButton,
                      ),

                      // Fotoğraf çekme butonu
                      Opacity(
                        opacity: _isLoading ? 0.5 : 1.0,
                        child: InkWell(
                          onTap: _isLoading ? null : _takePhoto,
                          child: Container(
                            margin: const EdgeInsets.symmetric(vertical: 5),
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

                      // Silme butonu
                      IconButton(
                        icon: const Icon(Icons.delete, color: Colors.white, size: 40),
                        onPressed: _isLoading
                            ? null
                            : () async {
                                if (_capturedImage == null) {
                                  _showSnackBar('Silinecek bir fotoğraf bulunamadı.', color: Colors.orangeAccent);
                                } else {
                                  // Uzun basınca onay diyaloğu
                                  final bool? confirm = await showDialog<bool>(
                                    context: context,
                                    builder: (context) => AlertDialog(
                                      title: const Text('Fotoğrafı Sil'),
                                      content: const Text('Fotoğrafı silmek istediğinize emin misiniz?'),
                                      actions: [
                                        TextButton(
                                          onPressed: () => Navigator.pop(context, false),
                                          child: const Text('Vazgeç'),
                                        ),
                                        TextButton(
                                          onPressed: () => Navigator.pop(context, true),
                                          child: const Text('Sil'),
                                        ),
                                      ],
                                    ),
                                  );
                                  if (confirm == true) {
                                    setState(() => _capturedImage = null);
                                    _showSnackBar('Fotoğraf kaldırıldı.', color: Colors.grey);
                                  }
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

        //--------------------------------------
        // Yükleme göstergesi (tam ekran overlay)
        //--------------------------------------
        if (_isLoading)
          Container(
            color: Colors.black54,
            child: const Center(child: CircularProgressIndicator()),
          ),
      ],
    );
  }
}
