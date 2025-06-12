Proje Geliştirme Süreci ve Git Flow Dal Yönetimi
Bu doküman, projenin geliştirme sürecini, Git Flow dal stratejisini, Jira görevleriyle ilişkilerini ve kullanılan Git komutlarını özetler. Her dal, belirli bir özelliği veya sürümü temsil eder ve Jira epik/task’leriyle eşleştirilmiştir. Proje, Flutter ile mobil uygulama, Flask ile backend ve yapay zeka modeli entegrasyonunu içerir.
Git Flow Stratejisi
Projemiz, Git Flow iş akışına uygun olarak yönetilmiştir. Kullanılan dallar:

main: Üretim ortamına hazır, stabil sürümler.
develop: Geliştirme ortamındaki en güncel kodlar.
feature/: Yeni özellikler için geçici dallar.
release/: Sürüm yayınlamadan önce son düzenlemeler için dallar.

Dal: feature/TASK-2-ui-scaffolding
Jira Karşılığı:

Epik: Kullanıcı Arayüzü (UI) Geliştirme
Story/Task: TASK-1, TASK-2 (Ana ekran ve temel sayfa yapılarının oluşturulması)

Açıklama (Hocanıza Söyleyecekleriniz):"Hocam, bu dalda projenin frontend temelini attık. Kullanıcının uygulamayı açtığında göreceği Login, Register ve Home ekranlarının görsel iskeletini Flutter ile kodladık. StatelessWidget ve StatefulWidget kullanarak, henüz işlevsel olmayan ama tasarım olarak tamamlanmış arayüzler hazırladık. Bu, projenin kullanıcı arayüzü için ilk adımdı."
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

Açıklama (Hocanıza Söyleyecekleriniz):"Bu dalda, projenin backend altyapısını kurduk. Flask ile REST API’sini (server.py) oluşturduk ve kullanıcı kimlik doğrulaması için Firebase’ı entegre ettik. Ayrıca, daha önce eğittiğimiz ilk yapay zeka modelini (.tflite formatında) API’ye ekledik. /images endpoint’i üzerinden basit tahminler yapabilen bir sistem kurduk."
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

Açıklama (Hocanıza Söyleyecekleriniz):"İlk modelimizin sınıf dengesizliği nedeniyle düşük performans gösterdiğini fark ettik. Bu dalda, train.py script’ini veri artırma (augmentation) ve sınıf ağırlıklandırma (class weights) teknikleriyle güncelledik. Böylece modelimiz daha adil ve doğru tahminler yapar hale geldi. Bu, önemli bir refactor işlemiydi."
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

Açıklama (Hocanıza Söyleyecekleriniz):"Bu dalda, mobil uygulama ile backend arasındaki bağlantıyı kurduk. Flutter’da image_picker paketini kullanarak CameraScanUI ve GaleriScanUI arayüzlerini geliştirdik. Kullanıcı, kamerayla fotoğraf çekip veya galerisinden seçim yaparak backend’e HTTP POST ile veri gönderiyor. Bu, uygulamanın görsel veri toplama işlevini sağladı."
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

Açıklama (Hocanıza Söyleyecekleriniz):"Daha iyi performans için model mimarimizi MobileNetV3Large’e yükselttik. Bu dalda, train.py script’ini bu modern ve verimli mimariyi kullanacak şekilde güncelledik. Bu değişiklik, modelin doğruluğunu ve çıkarım hızını artırdı, özellikle mobil cihazlar için optimize edildi."
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

Açıklama (Hocanıza Söyleyecekleriniz):"Projenin final aşamasında, ilk stabil sürümü (v1.0.0) hazırlamak için release dalı açtık. Backend’de server.py’ye try-except blokları ve loglama ekleyerek API’yi sağlamlaştırdık. Flutter’da ResultScreen, kullanıcı profili ve tüm arayüzleri tamamlayarak uygulamayı kullanıcı dostu hale getirdik. Ayrıca README.md’yi güncelledik. Bu dal, projenin uçtan uca çalıştığı stabil sürümü temsil ediyor."
İlgili Dosyalar:

backend/server.py
lib/ altındaki tüm dosyalar
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
git tag -a v1.0.0 -m "Sürüm 1.0.0: İlk

