Proje Geliştirme Süreci ve Git Flow Dal Yönetimi
Bu doküman, Flutter tabanlı bir mobil uygulama ve Flask ile geliştirilmiş bir backend içeren projemizin geliştirme sürecini ve Git Flow dal stratejisini açıklar. Her dal, Jira’daki epik ve görevlerle (story/task) ilişkilendirilmiştir. Proje, kullanıcı arayüzü (UI), backend altyapısı, yapay zeka modeli entegrasyonu ve kamera modülü gibi bileşenleri içerir. Aşağıda, her dalın amacı, sözlü sınav için açıklamalar, ilgili dosyalar ve Git komutları detaylı bir şekilde sunulmuştur.
Git Flow Stratejisi
Projemiz, Git Flow dal yönetimine uygun olarak geliştirilmiştir. Kullanılan dallar:

main: Üretim ortamına hazır, stabil sürümler. Yalnızca release veya hotfix dallarından gelen birleştirmeler kabul edilir.
develop: Geliştirme ortamındaki en güncel kodlar. Tüm feature dalları buraya birleştirilir.
feature/: Yeni özellikler için geçici dallar (ör. feature/TASK-2-ui-scaffolding). Geliştirme tamamlanınca develop’a birleştirilir.
release/: Sürüm yayınlamadan önce son düzenlemeler için dallar (ör. release/v1.0.0). Tamamlanınca main ve develop’a birleştirilir.
hotfix/: Üretimde acil hata düzeltmeleri için (bu projede kullanılmadı).

Dal: feature/TASK-2-ui-scaffolding
Jira Karşılığı:

Epik: Kullanıcı Arayüzü (UI) Geliştirme
Story/Task: TASK-1, TASK-2 (Ana ekran ve temel sayfa yapılarının oluşturulması)

Açıklama (Hocanıza Söyleyecekleriniz):"Hocam, bu dalda projenin kullanıcı arayüzünün temelini oluşturduk. Flutter kullanarak Login, Register ve Home ekranlarının görsel iskeletini kodladık. StatelessWidget ve StatefulWidget’larla, henüz işlevsel olmayan ama tasarım açısından tamamlanmış arayüzler hazırladık. Bu, projenin frontend tarafındaki ilk adımdı ve kullanıcı deneyimi için temel oluşturdu."
İlgili Dosyalar:

lib/Screens/Login.dart
lib/Screens/Register.dart
lib/Screens/Home.dart

Git Komutları:
# 1. develop dalından yeni bir özellik dalı oluştur
git checkout develop
git checkout -b feature/TASK-2-ui-scaffolding

# 2. Flutter UI dosyaları eklenir
git add lib/Screens/
git commit -m "feat(ui): TASK-2 - Login ve Register ekranlarının temel UI'ı oluşturuldu"

# 3. develop dalına birleştir
git checkout develop
git merge --no-ff feature/TASK-2-ui-scaffolding
git branch -d feature/TASK-2-ui-scaffolding

Dal: feature/TASK-5-backend-and-initial-model
Jira Karşılığı:

Epik: Veritabanı ve Backend Altyapısı / Yapay Zeka Model Entegrasyonu
Story/Task: TASK-3, TASK-4, TASK-5

Açıklama (Hocanıza Söyleyecekleriniz):"Bu dalda, projenin backend altyapısını kurduk. Python ve Flask ile server.py dosyasını oluşturarak REST API’yi hayata geçirdik. Kullanıcı kimlik doğrulaması için Firebase’ı entegre ettik ve Flutter tarafında gerekli yapılandırmayı yaptık. Ayrıca, train.py ile eğittiğimiz ilk yapay zeka modelini (.tflite formatında) API’ye ekledik. /images endpoint’i üzerinden basit tahminler yapabilen bir sistem kurduk."
İlgili Dosyalar:

backend/server.py
backend/train.py (ilk versiyon)
lib/firebase_options.dart

Git Komutları:
# 1. develop dalından yeni dal oluştur
git checkout develop
git checkout -b feature/TASK-5-backend-and-initial-model

# 2. Backend ve Firebase dosyaları eklenir
git add backend/ lib/firebase_options.dart
git commit -m "feat(backend,ai): TASK-5 - Flask API ve ilk AI modeli entegre edildi"

# 3. develop dalına birleştir
git checkout develop
git merge --no-ff feature/TASK-5-backend-and-initial-model
git branch -d feature/TASK-5-backend-and-initial-model

Dal: feature/TASK-11-model-improvement
Jira Karşılığı:

Epik: Yapay Zeka Model Entegrasyonu
Story/Task: TASK-11 (Modelin veri dengeleme ile iyileştirilmesi)

Açıklama (Hocanıza Söyleyecekleriniz):"İlk modelimizin performansını analiz ettiğimizde, veri setindeki sınıf dengesizliğinin sonuçları olumsuz etkilediğini fark ettik. Bu dalda, train.py script’ini veri artırma (augmentation) ve sınıf ağırlıklandırma (class weights) teknikleriyle güncelledik. Bu refactor işlemi, modelin doğruluğunu artırarak daha adil tahminler yapmasını sağladı."
İlgili Dosyalar:

backend/train.py (güncellenmiş hali)

Git Komutları:
# 1. develop dalından yeni dal oluştur
git checkout develop
git checkout -b feature/TASK-11-model-improvement

# 2. train.py güncellenir
git add backend/train.py
git commit -m "refactor(ai): TASK-11 - Model eğitimi veri dengeleme ile iyileştirildi"

# 3. develop dalına birleştir
git checkout develop
git merge --no-ff feature/TASK-11-model-improvement
git branch -d feature/TASK-11-model-improvement

Dal: feature/TASK-6-camera-module
Jira Karşılığı:

Epik: Görüntü İşleme ve Girdi Toplama
Story/Task: TASK-6 (Kamera modülü ile görsel veri elde edilmesi)

Açıklama (Hocanıza Söyleyecekleriniz):"Bu dalda, mobil uygulama ile backend arasındaki bağlantıyı kurduk. Flutter’da image_picker paketini kullanarak CameraScanUI ve GaleriScanUI arayüzlerini geliştirdik. Kullanıcı, telefonun kamerasıyla fotoğraf çekebilir veya galerisinden bir resim seçebilir. Seçilen görüntüler, http paketiyle backend’teki /images endpoint’ine POST isteğiyle gönderiliyor. Bu, uygulamanın görsel veri toplama işlevini sağladı."
İlgili Dosyalar:

lib/Screens/Camera.dart
lib/Screens/Galeri.dart

Git Komutları:
# 1. develop dalından yeni dal oluştur
git checkout develop
git checkout -b feature/TASK-6-camera-module

# 2. Kamera ve galeri dosyaları eklenir
git add lib/
git commit -m "feat(mobile): TASK-6 - Kamera ve galeri ile görüntü yükleme özelliği eklendi"

# 3. develop dalına birleştir
git checkout develop
git merge --no-ff feature/TASK-6-camera-module
git branch -d feature/TASK-6-camera-module

Dal: feature/TASK-12-upgrade-to-mobilenetv3
Jira Karşılığı:

Epik: Yapay Zeka Model Entegrasyonu
Story/Task: TASK-12 (Model mimarisinin MobileNetV3’e yükseltilmesi)

Açıklama (Hocanıza Söyleyecekleriniz):"Modelimizin performansını artırmak için daha modern bir mimariye geçtik. Bu dalda, train.py script’ini MobileNetV3Large mimarisini kullanacak şekilde güncelledik. Bu değişiklik, modelin doğruluğunu artırdı ve özellikle mobil cihazlar için çıkarım hızını optimize etti. MobileNetV3’ün hafif ve verimli yapısı, projemize büyük katkı sağladı."
İlgili Dosyalar:

backend/train.py (MobileNetV3 versiyonu)

Git Komutları:
# 1. develop dalından yeni dal oluştur
git checkout develop
git checkout -b feature/TASK-12-upgrade-to-mobilenetv3

# 2. train.py güncellenir
git add backend/train.py
git commit -m "feat(ai): TASK-12 - Model mimarisi MobileNetV3'e geçirildi"

# 3. develop dalına birleştir
git checkout develop
git merge --no-ff feature/TASK-12-upgrade-to-mobilenetv3
git branch -d feature/TASK-12-upgrade-to-mobilenetv3

Dal: release/v1.0.0
Jira Karşılığı:

Epik: Kullanıcı Geri Bildirimi / Test ve Hata Ayıklama
Story/Task: TASK-8, TASK-9, TASK-15, TASK-16

Açıklama (Hocanıza Söyleyecekleriniz):"Hocam, bu dalda projenin ilk stabil sürümünü (v1.0.0) hazırladık. Backend’de server.py’ye try-except blokları ve loglama ekleyerek API’yi daha sağlam hale getirdik. Flutter’da ResultScreen, kullanıcı profili ve tüm arayüzleri son tasarımlarıyla tamamladık. Ayrıca, proje dokümantasyonunu (README.md) güncelledik. Bu dal, TASK-15 (backend finalizasyonu) ve TASK-16 (Flutter finalizasyonu) görevlerini birleştirerek uçtan uca çalışan, kullanıcı dostu bir uygulama ortaya çıkardı."
İlgili Dosyalar:

backend/server.py
lib/ altındaki tüm dosyalar (ör. lib/Screens/ResultScreen.dart, lib/Screens/Profile.dart)
README.md

Git Komutları:
# 1. develop dalından release dalı oluştur
git checkout develop
git checkout -b release/v1.0.0

# 2. Tüm dosyalar (Flutter, Backend, README) güncellenir
git add .
git commit -m "feat!: TASK-15, TASK-16 - Backend ve Flutter son hali tamamlandı"

# 3. main ve develop dallarına birleştir
git checkout main
git merge --no-ff release/v1.0.0
git tag -a v1.0.0 -m "Sürüm 1.0.0: İlk Stabil Sürüm"
git checkout develop
git merge --no-ff release/v1.0.0
git branch -d release/v1.0.0

Notlar

Git Flow Uyumluluğu: Tüm feature dalları develop’a birleştirildi. release/v1.0.0 dalı, hem main hem develop’a birleştirilerek stabil sürüm yayınlandı.
Jira Entegrasyonu: Her dal, ilgili epik ve görevlerle eşleştirildi, böylece proje yönetimi şeffaf hale geldi.
Sürüm Yönetimi: Her önemli aşama (v0.1.0’dan v1.0.0’a) etiketlendi, ancak sadece v1.0.0 main dalına yansıdı.
Teknolojiler: Flutter (UI ve kamera), Flask (REST API), Firebase (kimlik doğrulama), TensorFlow (AI modeli), MobileNetV3 (gelişmiş AI mimarisi).

