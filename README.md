Cilt Hastalığı Tanı ve Öneri Mobil Uygulaması
<p align="center">
<a href="https://flutter.dev" target="_blank"><img src="https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white" alt="Flutter"></a>
<a href="https://www.python.org" target="_blank"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
<a href="https://flask.palletsprojects.com/" target="_blank"><img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"></a>
<a href="https://www.tensorflow.org" target="_blank"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
<a href="https://firebase.google.com/" target="_blank"><img src="https://img.shields.io/badge/Firebase-FFCA28?style=for-the-badge&logo=firebase&logoColor=black" alt="Firebase"></a>
</p>
<p align="center">
<i>Yapay zeka ve mobil teknolojiyi birleştirerek cilt sağlığına erişimi demokratikleştiren, platformdan bağımsız bir erken teşhis destek sistemi.</i>
</p>
<p align="center">
<img src="https://img.shields.io/github/last-commit/omerfaruk-celik/cilt-hastaligi-projesi" alt="Son Commit">
<img src="https://img.shields.io/github/contributors/omerfaruk-celik/cilt-hastaligi-projesi" alt="Katkıda Bulunanlar">
<img src="https://img.shields.io/github/license/omerfaruk-celik/cilt-hastaligi-projesi" alt="Lisans">
</p>
📖 İçindekiler

    Proje Hakkında

        Problem Alanı ve Çözüm Yaklaşımı

        Temel Özellikler

    Teknik Yığın (Tech Stack)

    Sistem Mimarisi

        Mikroservis Yapısı

        Veri Akışı Diyagramı

    Kurulum ve Çalıştırma

        Ön Gereksinimler

        Backend Kurulumu (Flask API)

        Frontend Kurulumu (Flutter)

    API Endpoint Dokümantasyonu

    Yapay Zeka Modeli

        Model Mimarisi ve Eğitim Süreci

        Performans Metrikleri

    Yazılım Geliştirme Süreçleri

        Agile & Kanban

        Git & GitHub Flow

    Test Stratejisi

    Gelecek Geliştirmeler

    Ekip

    Lisans

🧐 Proje Hakkında

Bu proje, BMU 326 Yazılım Mühendisliği dersi kapsamında, modern mühendislik pratikleri kullanılarak geliştirilmiş uçtan uca bir yapay zeka uygulamasıdır. Temel amacı, kullanıcıların mobil cihazları aracılığıyla cilt lezyonlarının fotoğraflarını analiz ederek, potansiyel cilt hastalıkları hakkında hızlı, erişilebilir ve güvenilir bir ön bilgi almalarını sağlamaktır.
🎯 Problem Alanı ve Çözüm Yaklaşımı

Problem: Dermatolojik rahatsızlıklar yaygın olmasına rağmen, uzman hekime erişimdeki zorluklar ve maliyetler nedeniyle erken teşhis sıklıkla gecikmektedir. Bu durum, tedavi süreçlerini zorlaştırabilmekte ve sağlık risklerini artırabilmektedir.

Çözüm: Bu projede, derin öğrenme tabanlı bir görüntü sınıflandırma modeli geliştirilerek, bu modelin servis edildiği bir REST API ve bu API'yi kullanan bir mobil uygulama oluşturulmuştur. Kullanıcı, sisteme bir fotoğraf yükler; sistem bu fotoğrafı analiz eder ve saniyeler içinde potansiyel bir tanı ve ilgili temel önerileri sunar. Bu sistem, tıbbi bir teşhisin yerini tutmamakla birlikte, kullanıcıyı bir uzmana danışması için teşvik eden kritik bir ilk adım görevi görür.
✨ Temel Özellikler

    Yapay Zeka Destekli Tanı: Yüksek doğrulukla eğitilmiş TensorFlow Lite modeli ile anlık görüntü analizi.

    Platformdan Bağımsız Mobil Uygulama: Flutter ile geliştirilen uygulama, hem Android hem de iOS üzerinde doğal performansla çalışır.

    Güvenli Kimlik Doğrulama: E-posta/şifre ile güvenli kullanıcı kaydı ve girişi için Firebase Authentication entegrasyonu.

    Anlık Geri Bildirim: Kullanıcıya özel, anlaşılır sonuç ekranı ve temel sağlık önerileri.

    Esnek Mimari: Mikroservis yapısı sayesinde her bir bileşenin (mobil, API, veritabanı) bağımsız olarak geliştirilmesi ve ölçeklendirilmesi.

🛠️ Teknik Yığın (Tech Stack)
Kategori	Teknoloji / Kütüphane	Amaç
Frontend	Flutter (Dart)	Platformdan bağımsız mobil uygulama geliştirme
	firebase_auth, firebase_core	Kullanıcı kimlik doğrulama
	http, image_picker	API iletişimi ve cihaz kamerasından/galerisinden görüntü alma
Backend	Python, Flask	RESTful API sunucusu oluşturma
	TensorFlow Lite Runtime	Optimize edilmiş AI modelini çalıştırma
	Pillow, NumPy	Görüntü işleme ve sayısal operasyonlar
AI Model Geliştirme	TensorFlow, Keras	Transfer Learning ile derin öğrenme modeli eğitimi
	Pandas, Matplotlib, Seaborn	Veri analizi, işleme ve sonuçların görselleştirilmesi
Altyapı & DevOps	Git & GitHub	Sürüm kontrolü ve işbirliği
	Jira	Agile proje yönetimi (Kanban)
🏛️ Sistem Mimarisi
Mikroservis Yapısı

Proje, üç ana bağımsız servisten oluşan bir mikroservis mimarisi üzerine inşa edilmiştir:

    Flutter Client: Kullanıcı arayüzü ve state yönetiminden sorumludur. Backend ile REST API üzerinden asenkron olarak iletişim kurar.

    Flask Prediction API: Yapay zeka modelini barındırır ve /images gibi endpoint'ler üzerinden tahmin isteklerini karşılar. Durumsuz (stateless) bir yapıya sahiptir.

    Firebase Authentication Service: Kullanıcı verilerini ve kimlik doğrulama mantığını yöneten harici bir servistir.

Bu ayrım, her bir servisin diğerlerini etkilemeden bağımsız olarak geliştirilmesine, test edilmesine ve dağıtılmasına olanak tanır.
Veri Akışı Diyagramı

      
sequenceDiagram
    participant User
    participant Flutter App
    participant Flask API
    participant TensorFlow Lite Model
    participant Firebase Auth

    User->>Flutter App: Fotoğraf Yükle
    Flutter App->>Flask API: POST /images (Görüntü ile)
    Flask API->>TensorFlow Lite Model: Görüntüyü analiz et
    TensorFlow Lite Model-->>Flask API: Tahmin sonucu (JSON)
    Flask API-->>Flutter App: 201 Created (Başarılı)
    
    Flutter App->>Flask API: GET /tahmin
    Flask API-->>Flutter App: Tahmin detayları (JSON)
    Flutter App->>User: Sonuçları Göster

    Note over User, Firebase Auth: Kullanıcı Giriş/Kayıt Akışı
    User->>Flutter App: E-posta/Şifre Gir
    Flutter App->>Firebase Auth: signInWithEmailAndPassword()
    Firebase Auth-->>Flutter App: Kullanıcı Token'ı
    Flutter App->>User: Ana Sayfaya Yönlendir

    

IGNORE_WHEN_COPYING_START
Use code with caution. Mermaid
IGNORE_WHEN_COPYING_END
🚀 Kurulum ve Çalıştırma
Ön Gereksinimler

    Git

    Flutter SDK (3.10+)

    Python (3.9+)

    Android Studio / VS Code

    Aktif bir Firebase projesi

Backend Kurulumu (Flask API)

    Depoyu klonlayın ve backend dizinine gidin:

          
    git clone https://github.com/omerfaruk-celik/cilt-hastaligi-projesi.git
    cd cilt-hastaligi-projesi/backend

        

    IGNORE_WHEN_COPYING_START

Use code with caution. Bash
IGNORE_WHEN_COPYING_END

Sanal ortam (virtual environment) oluşturun ve aktive edin:

      
python -m venv venv
source venv/bin/activate  # macOS/Linux için
# venv\Scripts\activate  # Windows için

    

IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

Gerekli paketleri kurun:

      
pip install -r requirements.txt

    

IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

SSL sertifikası oluşturun (eğer yoksa):
Geliştirme ortamı için openssl ile kendinden imzalı bir sertifika oluşturabilirsiniz.

Sunucuyu başlatın:

      
python server.py

    

IGNORE_WHEN_COPYING_START

    Use code with caution. Bash
    IGNORE_WHEN_COPYING_END

    Sunucu varsayılan olarak https://0.0.0.0:5001 adresinde çalışacaktır.

Frontend Kurulumu (Flutter)

    Firebase projenizi Flutter'a entegre edin:

        Firebase konsolundan projenizi oluşturun.

        flutterfire configure komutunu kullanarak firebase_options.dart dosyasını oluşturun.

        Firebase Authentication'ı etkinleştirin (E-posta/Şifre metodu).

    Flutter projesinin ana dizinine gidin:

          
    cd .. # Projenin ana dizini

        

    IGNORE_WHEN_COPYING_START

Use code with caution. Bash
IGNORE_WHEN_COPYING_END

Paketleri yükleyin:

      
flutter pub get

    

IGNORE_WHEN_COPYING_START
Use code with caution. Bash
IGNORE_WHEN_COPYING_END

Uygulamayı çalıştırın:
Bir emülatörün veya fiziksel cihazın bağlı olduğundan emin olun.

      
flutter run

    

IGNORE_WHEN_COPYING_START

    Use code with caution. Bash
    IGNORE_WHEN_COPYING_END

    Not: main.dart içindeki HttpOverrides kodu, geliştirme ortamında kendinden imzalı SSL sertifikasını kabul etmek için gereklidir. Canlı ortamda kaldırılmalıdır.

🌐 API Endpoint Dokümantasyonu
Method	Endpoint	Açıklama	Yanıt (Başarılı)
POST	/images	Multipart-form olarak bir görüntü dosyası yükler, analiz eder ve sonucu kaydeder.	201 Created - {'message': 'Dosya yüklendi', 'filename': ...}
POST	/upload_image	Ham image/jpeg verisi alır (ESP32 gibi cihazlar için).	201 Created - {'result': {...}, 'filename': ...}
GET	/tahmin	Son yüklenen görüntü için yapılan tahmin sonucunu JSON formatında döndürür.	200 OK - {'hasDisease': true, 'diseaseName': ..., 'probabilities': ...}
GET	/images/<filename>	Belirtilen filename'e sahip görüntüyü sunar.	Görüntü dosyası
GET	/tahmin/<filename>	Belirtilen filename'e sahip tahmin metin dosyasını sunar.	Metin dosyası
🤖 Yapay Zeka Modeli
Model Mimarisi ve Eğitim Süreci

    Model: MobileNetV3Large mimarisi üzerinde Transfer Learning tekniği kullanılmıştır. ImageNet üzerinde ön-eğitilmiş ağırlıklar kullanılarak modelin öğrenme süreci hızlandırılmış ve performansı artırılmıştır.

    Veri Kümesi: Model, HAM10000 gibi halka açık dermatolojik görüntü veri kümeleri kullanılarak eğitilmiştir.

    Eğitim:

        Feature Extraction: İlk aşamada, MobileNetV3'nin temel katmanları dondurulmuş ve sadece üzerine eklenen Dense sınıflandırma katmanları eğitilmiştir.

        Fine-Tuning: Daha sonra, temel modelin üst katmanlarının bir kısmı çözülerek, daha düşük bir öğrenme oranı (learning_rate) ile tüm model yeniden eğitilmiştir. Bu, modelin veri kümesine daha iyi adapte olmasını sağlamıştır.

    Optimizasyon: Eğitim sonrası model, mobil cihazlarda yüksek performansla çalışması için TensorFlow Lite formatına dönüştürülmüş ve nicemlenmiştir (quantization).

Performans Metrikleri

Modelin performansı, accuracy, precision, recall ve F1-score gibi standart metriklerle değerlendirilmiştir. Özellikle dengesiz veri kümelerinde daha anlamlı bir sonuç veren Focal Loss fonksiyonu, eğitim sürecinde kullanılmıştır.
🔄 Yazılım Geliştirme Süreçleri
Agile & Kanban

Proje, Kanban metodolojisi kullanılarak çevik bir yaklaşımla yönetilmiştir. Jira üzerinde oluşturulan proje panosu, iş akışını görselleştirmek ve takım içi senkronizasyonu sağlamak için kullanılmıştır. Görevler Epic > Story > Task hiyerarşisi ile tanımlanarak projenin kapsamı netleştirilmiştir.

[Buraya Jira Kanban panosunun bir ekran görüntüsü eklenecektir.]
Git & GitHub Flow

Sürüm kontrolü için Git, iş akışı yönetimi için ise GitHub Flow modeli benimsenmiştir.

    main branch'i her zaman kararlı ve dağıtıma hazır kodu temsil eder.

    Tüm yeni geliştirmeler, main'den oluşturulan feature/* branch'lerinde yapılır.

    Tamamlanan işler, Pull Request (PR) ve Code Review süreçlerinden geçtikten sonra main branch'ine birleştirilir.

Bu yapı, kod kalitesini artırmış ve takım içi işbirliğini güçlendirmiştir.
✅ Test Stratejisi

Projenin kalitesini güvence altına almak için çok katmanlı bir test stratejisi uygulanmıştır:

    Birim Testleri (Unit Tests): Flutter'da validator fonksiyonları gibi tekil iş mantıklarının doğruluğunu test etmek için yazılmıştır.

    Widget Testleri: Flutter'da UI bileşenlerinin farklı girdilere göre doğru şekilde render edilip edilmediğini doğrulamak için kullanılmıştır.

    Manuel Testler: Uygulamanın uçtan uca kullanıcı akışları (kayıt olma, fotoğraf yükleme, sonuç görme) manuel olarak farklı cihazlarda test edilmiştir.

🔮 Gelecek Geliştirmeler

    Kullanıcı Geçmişi: Kullanıcıların daha önce yaptığı tüm analiz sonuçlarını ve fotoğrafları görebileceği bir "Geçmiş" ekranı.

    Google ile Giriş: Firebase kullanarak Google hesabı ile tek tıkla giriş yapma özelliği.

    Daha Fazla Hastalık: Modeli daha çeşitli ve daha fazla sayıda cilt hastalığını tanıyacak şekilde yeniden eğitmek.

    CI/CD Pipeline: GitHub Actions ile main branch'ine yapılan her birleştirmede backend API'sinin otomatik olarak bir bulut platformuna (örn: Render, Heroku) dağıtılması.

👥 Ekip

    Ahmet Al Rusutm - (GitHub)

    Adham Wasim Sherif - (GitHub)

    Ömer Faruk Çelik - (GitHub)

📜 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına göz atınız.
