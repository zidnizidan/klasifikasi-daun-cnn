import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Muat model
model = tf.keras.models.load_model("model_klasifikasi_daun.h5")

# Gambar uji
img_path = "daun_ujian.jpg"  # ubah jika nama file berbeda
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Label kelas (urutkan sesuai dengan train_generator.class_indices)
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Target_Spot',
    'Tomato___Tomato_YellowLeaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus'
]

# Prediksi
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
confidence = np.max(prediction)

# Tampilkan hasil
print("📸 Gambar:", img_path)
print("🧠 Prediksi Kelas:", class_names[predicted_index])
print("✅ Keyakinan: {:.2f}%".format(confidence * 100))
