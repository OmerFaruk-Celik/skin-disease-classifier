import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template_string
import os
import logging
from datetime import datetime
import threading
import re

# Flask uygulaması ve log ayarları
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Klasör yolları
UPLOAD_FOLDER = 'images'
TAHMIN_FOLDER = 'tahmin'
STATIC_FOLDER = 'static'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER_ABSOLUTE = os.path.join(APP_ROOT, UPLOAD_FOLDER)
TAHMIN_FOLDER_ABSOLUTE = os.path.join(APP_ROOT, TAHMIN_FOLDER)

os.makedirs(UPLOAD_FOLDER_ABSOLUTE, exist_ok=True)
os.makedirs(TAHMIN_FOLDER_ABSOLUTE, exist_ok=True)

# TFLite modelini yükle
interpreter = None
input_details = None
output_details = None
try:
    interpreter = tf.lite.Interpreter(model_path='model_quant_old.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("TFLite modeli başarıyla yüklendi.")
except Exception as e:
    logger.error(f"TFLite model yüklenirken KRİTİK HATA: {e}", exc_info=True)
    exit()

classes = ['bcc', 'df', 'mel', 'nv', 'vasc']

# Default threshold for all classes
PREDICTION_THRESHOLD = 0.35
# Special threshold for "nv" class (index 3)
NV_THRESHOLD = 0.55

latest_image_filename_for_tahmin_endpoint = None
latest_tahmin_filename_for_tahmin_endpoint = None
processing_lock = threading.Lock()

def make_prediction(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        if input_details[0]['dtype'] == np.int8 or input_details[0]['dtype'] == np.uint8:
            scale, zero_point = input_details[0]['quantization']
            if scale != 0:
                img_array = (img_array / scale + zero_point).astype(input_details[0]['dtype'])

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions_raw = interpreter.get_tensor(output_details[0]['index'])

        if output_details[0]['dtype'] == np.int8 or output_details[0]['dtype'] == np.uint8:
            output_scale, output_zero_point = output_details[0]['quantization']
            if output_scale != 0:
                predictions_float = predictions_raw.astype(np.float32)
                predictions = (predictions_float - output_zero_point) * output_scale
            else:
                predictions = predictions_raw.astype(np.float32)
        else:
            predictions = predictions_raw.astype(np.float32)

        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = classes[predicted_class_index]
        max_probability = float(predictions[0][predicted_class_index])
        probabilities = {classes[i]: float(predictions[0][i]) for i in range(len(classes))}

        logger.info(f"Tahmin yapıldı: {predicted_class_name}, Olasılıklar: {probabilities}")

        # Class-specific threshold: "nv" için 0.8, diğerleri için 0.65
        if predicted_class_name == "nv":
            threshold = NV_THRESHOLD
        else:
            threshold = PREDICTION_THRESHOLD

        if max_probability < threshold:
            logger.warning(f"Tahminin güveni düşük [{predicted_class_name}] ({max_probability:.4f} < {threshold}), sonuç null döndürülüyor.")
            return None, None

        return predicted_class_name, probabilities
    except Exception as e:
        logger.error(f"Tahmin yaparken hata: {e}", exc_info=True)
        return None, None

@app.route('/images', methods=['POST'])
def upload_image_from_flutter():
    global latest_image_filename_for_tahmin_endpoint
    global latest_tahmin_filename_for_tahmin_endpoint

    if 'file' not in request.files:
        logger.warning("POST /images: İstekte 'file' parametresi bulunamadı.")
        return jsonify({'error': "'file' parametresi eksik"}), 400

    file = request.files['file']

    if not file.filename:
        logger.warning("POST /images: Boş dosya adı.")
        return jsonify({'error': 'Geçersiz dosya adı'}), 400

    if not re.match(r'.*\.(jpg|jpeg|png)$', file.filename, re.IGNORECASE):
        logger.warning(f"POST /images: Geçersiz dosya türü veya adı: {file.filename}")
        return jsonify({'error': 'Geçersiz dosya türü (sadece jpg, jpeg, png)'}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        new_filename = f"image_{timestamp}.{file.filename.rsplit('.', 1)[-1].lower()}"

        filepath = os.path.join(UPLOAD_FOLDER_ABSOLUTE, new_filename)
        file.save(filepath)
        logger.info(f"Görüntü başarıyla kaydedildi: {filepath}")

        predicted_class, probabilities = make_prediction(filepath)

        current_tahmin_filename = None
        if predicted_class and probabilities:
            tahmin_file_basename = f"tahmin_{timestamp}.txt"
            tahmin_filepath = os.path.join(TAHMIN_FOLDER_ABSOLUTE, tahmin_file_basename)

            with open(tahmin_filepath, 'w') as f:
                f.write(f"Tahmin edilen sınıf: {predicted_class}\n")
                f.write("Olasılıklar:\n")
                for cls, prob in probabilities.items():
                    f.write(f"{cls}: {prob:.4f}\n")
            current_tahmin_filename = tahmin_file_basename
            logger.info(f"Tahmin sonucu kaydedildi: {tahmin_filepath}")
        else:
            logger.warning(f"Görüntü için tahmin yapılamadı veya güven eşiğinin altında: {filepath}")

        with processing_lock:
            latest_image_filename_for_tahmin_endpoint = new_filename
            latest_tahmin_filename_for_tahmin_endpoint = current_tahmin_filename

        return jsonify({'message': 'Dosya başarıyla yüklendi', 'filename': new_filename}), 201

    except Exception as e:
        logger.error(f"POST /images: Görüntü işlenirken/kaydedilirken hata: {e}", exc_info=True)
        return jsonify({'error': 'Görüntü işlenirken sunucuda bir hata oluştu'}), 500

@app.route('/upload_image', methods=['POST'])
def upload_raw_image_from_esp32():
    global latest_image_filename_for_tahmin_endpoint
    global latest_tahmin_filename_for_tahmin_endpoint

    if 'image/jpeg' not in request.content_type:
        logger.warning(f"POST /upload_image: Geçersiz Content-Type: {request.content_type}")
        return jsonify({'error': 'Content-Type image/jpeg olmalı'}), 400

    image_data = request.get_data()
    if not image_data:
        logger.warning("POST /upload_image: Boş görüntü verisi alındı.")
        return jsonify({'error': 'Görüntü verisi alınamadı'}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f'image_{timestamp}.jpg'
        filepath = os.path.join(UPLOAD_FOLDER_ABSOLUTE, filename)
        with open(filepath, 'wb') as f:
            f.write(image_data)
        logger.info(f"Ham görüntü başarıyla kaydedildi: {filepath}")

        predicted_class, probabilities = make_prediction(filepath)
        current_tahmin_filename = None

        if predicted_class and probabilities:
            tahmin_file_basename = f"tahmin_{timestamp}.txt"
            tahmin_filepath = os.path.join(TAHMIN_FOLDER_ABSOLUTE, tahmin_file_basename)
            with open(tahmin_filepath, 'w') as f:
                f.write(f"Tahmin edilen sınıf: {predicted_class}\n")
                f.write("Olasılıklar:\n")
                for cls, prob in probabilities.items():
                    f.write(f"{cls}: {prob:.4f}\n")
            current_tahmin_filename = tahmin_file_basename
            logger.info(f"Ham görüntü için tahmin sonucu kaydedildi: {tahmin_filepath}")

            result_for_esp = {
                "hasDisease": True,
                "diseaseName": predicted_class,
                "probabilities": probabilities
            }
            with processing_lock:
                latest_image_filename_for_tahmin_endpoint = filename
                latest_tahmin_filename_for_tahmin_endpoint = current_tahmin_filename
            return jsonify({'result': result_for_esp, 'filename': filename, 'tahmin_file': current_tahmin_filename}), 201
        else:
            logger.warning(f"Ham görüntü için tahmin yapılamadı veya güven eşiğinin altında: {filepath}")
            with processing_lock:
                latest_image_filename_for_tahmin_endpoint = filename
                latest_tahmin_filename_for_tahmin_endpoint = None
            return jsonify({'error': 'Tahmin yapılamadı veya güven eşiğinin altında'}), 500

    except Exception as e:
        logger.error(f"POST /upload_image: Ham görüntü işlenirken hata: {e}", exc_info=True)
        return jsonify({'error': 'Görüntü işlenirken sunucuda bir hata oluştu'}), 500

@app.route('/tahmin', methods=['GET'])
def get_latest_tahmin_for_flutter():
    global latest_tahmin_filename_for_tahmin_endpoint
    global processing_lock

    tahmin_to_serve = None
    with processing_lock:
        tahmin_to_serve = latest_tahmin_filename_for_tahmin_endpoint

    if tahmin_to_serve:
        tahmin_filepath = os.path.join(TAHMIN_FOLDER_ABSOLUTE, tahmin_to_serve)
        if os.path.exists(tahmin_filepath):
            try:
                with open(tahmin_filepath, 'r') as f:
                    lines = f.readlines()

                if not lines or len(lines) < 2:
                    logger.error(f"GET /tahmin: {tahmin_to_serve} dosyası beklenen formatta değil.")
                    return jsonify({'error': 'Tahmin dosyası formatı geçersiz'}), 500

                predicted_class = lines[0].replace("Tahmin edilen sınıf:", "").strip()
                probabilities = {}
                try:
                    prob_start_index = next(i for i, line in enumerate(lines) if "Olasılıklar:" in line) + 1
                    for line in lines[prob_start_index:]:
                        if ':' in line:
                            parts = line.strip().split(':')
                            if len(parts) == 2:
                                cls = parts[0].strip()
                                prob_str = parts[1].strip()
                                try:
                                    probabilities[cls] = float(prob_str)
                                except ValueError:
                                    logger.warning(f"GET /tahmin: {tahmin_to_serve} dosyasında geçersiz olasılık değeri: {prob_str} for class {cls}")
                            else:
                                logger.warning(f"GET /tahmin: {tahmin_to_serve} dosyasında geçersiz olasılık satırı formatı: {line.strip()}")
                        else:
                            logger.warning(f"GET /tahmin: {tahmin_to_serve} dosyasında ':' içermeyen olasılık satırı: {line.strip()}")
                except StopIteration:
                    logger.error(f"GET /tahmin: {tahmin_to_serve} dosyasında 'Olasılıklar:' satırı bulunamadı.")
                    return jsonify({'error': "Tahmin dosyasında 'Olasılıklar:' başlığı eksik"}), 500

                result = {
                    "hasDisease": True,
                    "diseaseName": predicted_class,
                    "probabilities": probabilities
                }
                logger.info(f"GET /tahmin: {tahmin_to_serve} için sonuç başarıyla oluşturuldu.")
                return jsonify(result), 200
            except Exception as e:
                logger.error(f"GET /tahmin: {tahmin_to_serve} okunurken/işlenirken hata: {e}", exc_info=True)
                return jsonify({'error': 'Tahmin sonucu işlenirken sunucuda hata oluştu'}), 500
        else:
            logger.warning(f"GET /tahmin: En son tahmin dosyası ({tahmin_to_serve}) diskte bulunamadı.")
            return jsonify({'error': 'En son tahmin sonucu bulunamadı (dosya diskte yok)'}), 404
    else:
        logger.warning("GET /tahmin: Sunucuda kayıtlı en son tahmin bulunmuyor.")
        return jsonify({'error': 'Henüz işlenmiş bir tahmin yok'}), 404

@app.route('/images', methods=['GET'])
def list_images():
    try:
        files = sorted([
            f for f in os.listdir(UPLOAD_FOLDER_ABSOLUTE)
            if os.path.isfile(os.path.join(UPLOAD_FOLDER_ABSOLUTE, f)) and
               re.match(r'.*\.(jpg|jpeg|png)$', f, re.IGNORECASE)
        ], key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER_ABSOLUTE, x)), reverse=True)
        return jsonify({'files': files}), 200
    except Exception as e:
        logger.error(f"GET /images listesi alınırken hata: {e}", exc_info=True)
        return jsonify({'error': 'Dosya listesi alınamadı'}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    logger.info(f"GET /images/{filename} isteği alındı.")
    try:
        return send_from_directory(UPLOAD_FOLDER_ABSOLUTE, filename, as_attachment=False)
    except FileNotFoundError:
        logger.warning(f"GET /images/{filename}: Dosya bulunamadı.")
        return jsonify({'error': 'Dosya bulunamadı'}), 404
    except Exception as e:
        logger.error(f"GET /images/{filename} sunulurken hata: {e}", exc_info=True)
        return jsonify({'error': 'Dosya sunulurken hata'}), 500

@app.route('/tahmin/<path:filename>', methods=['GET'])
def serve_tahmin_file(filename):
    logger.info(f"GET /tahmin/{filename} isteği alındı.")
    try:
        return send_from_directory(TAHMIN_FOLDER_ABSOLUTE, filename, as_attachment=False, mimetype='text/plain')
    except FileNotFoundError:
        logger.warning(f"GET /tahmin/{filename}: Tahmin dosyası bulunamadı.")
        return jsonify({'error': 'Tahmin dosyası bulunamadı'}), 404
    except Exception as e:
        logger.error(f"GET /tahmin/{filename} sunulurken hata: {e}", exc_info=True)
        return jsonify({'error': 'Tahmin dosyası sunulurken hata'}), 500

LIVE_VIEW_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sunucu Canlı Görüntü</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4; text-align: center; }
        h1 { color: #333; }
        #liveImage { border: 2px solid #ccc; border-radius: 8px; max-width: 800px; max-height: 600px; width: auto; height: auto; background-color: #fff; padding: 5px; }
        p { color: #666; }
        .upload-form { margin-top: 20px; padding:15px; background-color: #fff; border-radius: 8px; display: inline-block; }
    </style>
</head>
<body>
    <h1>Sunucu Anlık Görüntü Akışı</h1>
    <p>Bu sayfa, sunucuya en son yüklenen resmi otomatik olarak yeniler.</p>
    <img id="liveImage" src="/latest_image_feed" alt="Canlı Görüntü Yükleniyor...">
    <p id="timestamp">Son Güncelleme: Bekleniyor...</p>
    <div class="upload-form">
        <h2>Test için Resim Yükle (POST /images)</h2>
        <form id="uploadForm" action="/images" method="post" enctype="multipart/form-data">
            <input type="file" name="file" id="fileInput" accept="image/jpeg,image/png" required>
            <button type="submit">Yükle ve Tahmin Et</button>
        </form>
        <div id="uploadResult" style="margin-top:10px;"></div>
    </div>
    <script>
        const imageElement = document.getElementById('liveImage');
        const timestampElement = document.getElementById('timestamp');
        const refreshInterval = 2000;
        function refreshImage() {
            imageElement.src = "/latest_image_feed?" + new Date().getTime();
        }
        function updateTimestamp() {
            timestampElement.textContent = "Son Güncelleme: " + new Date().toLocaleTimeString();
        }
        imageElement.onload = updateTimestamp;
        imageElement.onerror = function() {
            timestampElement.textContent = "Görüntü yüklenemedi veya henüz görüntü yok.";
        };
        setInterval(refreshImage, refreshInterval);
        const uploadForm = document.getElementById('uploadForm');
        const uploadResultDiv = document.getElementById('uploadResult');
        if (uploadForm) {
            uploadForm.addEventListener('submit', async function(event) {
                event.preventDefault();
                const formData = new FormData(uploadForm);
                uploadResultDiv.textContent = 'Yükleniyor...';
                try {
                    const response = await fetch('/images', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    if (response.ok) {
                        uploadResultDiv.textContent = 'Yükleme Başarılı: ' + JSON.stringify(result);
                    } else {
                        uploadResultDiv.textContent = 'Yükleme Hatası: ' + JSON.stringify(result);
                    }
                } catch (error) {
                    uploadResultDiv.textContent = 'Bir hata oluştu: ' + error;
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
@app.route('/live_view')
def live_view_page():
    return render_template_string(LIVE_VIEW_HTML)

@app.route('/latest_image_feed')
def latest_image_feed_endpoint():
    global latest_image_filename_for_tahmin_endpoint
    global processing_lock

    current_image_filename_local = None
    with processing_lock:
        current_image_filename_local = latest_image_filename_for_tahmin_endpoint

    if current_image_filename_local:
        image_path_to_serve = os.path.join(UPLOAD_FOLDER_ABSOLUTE, current_image_filename_local)
        if os.path.exists(image_path_to_serve):
            logger.debug(f"Feed: En son görüntü sunuluyor: {image_path_to_serve}")
            try:
                return send_file(image_path_to_serve, mimetype='image/jpeg',
                                 as_attachment=False, download_name='live.jpg', max_age=0)
            except Exception as e:
                logger.error(f"Feed: {image_path_to_serve} gönderilirken hata: {e}", exc_info=True)
                return jsonify({'error': 'Görüntü dosyası gönderilirken hata'}), 500
        else:
            logger.warning(f"Feed: {image_path_to_serve} diskte bulunamadı ama adı kayıtlıydı.")

    logger.info("Feed: Gösterilecek kaydedilmiş görüntü bulunamadı veya bir hata oluştu. Şeffaf PNG dönülüyor.")
    from io import BytesIO
    import base64
    transparent_png_data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
    return send_file(BytesIO(transparent_png_data), mimetype='image/png', max_age=0)

if __name__ == '__main__':
    flask_port = 5001
    logger.info(f"Flask sunucusu HTTPS olarak başlatılıyor. https://0.0.0.0:{flask_port}")

    cert_path = os.path.join(APP_ROOT, 'cert.pem')
    key_path = os.path.join(APP_ROOT, 'key.pem')

    if not os.path.exists(cert_path) or not os.path.exists(key_path):
        logger.critical(f"KRİTİK HATA: SSL sertifika ({cert_path}) veya anahtar ({key_path}) dosyası bulunamadı! Sunucu başlatılamıyor.")
        print(f"HATA: SSL sertifika ({cert_path}) veya anahtar ({key_path}) dosyası bulunamadı!")
        exit()

    try:
        app.run(host='0.0.0.0', port=flask_port, debug=True, use_reloader=False,
                ssl_context=(cert_path, key_path))
    except Exception as e:
        logger.critical(f"Sunucu başlatılırken KRİTİK HATA: {e}", exc_info=True)
