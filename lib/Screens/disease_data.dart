// lib/data/disease_data.dart

// Hastalık etiketleri (AI modelinizin tahmin ettiği sınıflar)
final List<String> diseaseNames = [
  "bcc",     // Bazal Hücreli Karsinom
  "df",      // Dermatofibrom
  "mel",     // Melanom
  "nv",      // Melanositik Nevus
  "vasc",    // Vasküler lezyon
  "ak",      // Aktinik Keratoz (örnek)
  "rosacea", // Rozasea (örnek)
];

// Her hastalık için öneriler (sıralama, yukarıdaki listeyle birebir uyuşmalı)
final List<List<String>> diseaseSuggestions = [
  [ // bcc - Bazal Hücreli Karsinom
    "Güneşten korunmak için güneş kremi kullanın.",
    "Şüpheli lezyonları zamanında kontrol ettirin.",
    "Düzenli dermatoloji kontrollerine gidin.",
    "Solaryumdan kaçının.",
    "Koruyucu giysiler giyin.",
  ],
  [ // df - Dermatofibrom
    "Lezyonları kaşımaktan kaçının.",
    "Cilt tahrişini önleyin.",
    "Takip için dermatoloğa danışın.",
    "Kendi kendinize teşhis koymayın.",
    "Ciltte büyüme varsa kontrol ettirin.",
  ],
  [ // mel - Melanom
    "Güneş koruyucu kullanın.",
    "Açık tenliyseniz düzenli cilt taraması yaptırın.",
    "Benlerde renk/asimetri değişimlerini izleyin.",
    "Şüpheli benleri dermatoloğa gösterin.",
    "Aile öyküsü varsa dikkatli olun.",
  ],
  [ // nv - Melanositik Nevus
    "Benlerin şekil ve renk değişimlerini takip edin.",
    "UV ışınlarından korunmak için şapka/güneş kremi kullanın.",
    "Benleri koparmayın veya kesmeyin.",
    "Yıllık cilt kontrolü yaptırın.",
    "Aile öyküsü varsa dikkatli olun.",
  ],
  [ // vasc - Vasküler lezyon
    "Aşırı sıcak-soğuk farklarından kaçının.",
    "Güneş koruyucu kullanın.",
    "İnce damar yapısına sahip bölgeleri koruyun.",
    "Düzenli cilt kontrolü yaptırın.",
    "Kılcal damar genişlemesi varsa doktora danışın.",
  ],
  [ // ak - Aktinik Keratoz
    "Güneşe çıkmadan 30 dk önce koruyucu sürün.",
    "Geniş siperli şapka takın.",
    "Ciltteki pürüzlü bölgeleri kontrol ettirin.",
    "Sigara ve alkolü sınırlayın.",
    "Dermatoloğun önerdiği ürünleri kullanın.",
  ],
  [ // rosacea - Rozasea
    "Baharatlı yiyeceklerden kaçının.",
    "Güneş koruyucu kullanın.",
    "Aşırı sıcak suyla yıkanmayın.",
    "Nazik temizleyiciler kullanın.",
    "Alkol tüketimini azaltın.",
  ],
];
