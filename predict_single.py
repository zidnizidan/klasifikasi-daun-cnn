# predict_single.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os

# Load model
model = load_model("model_klasifikasi_daun.h5")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Gambar input (bisa diganti dengan gambar lain)
img_path = "daun_ujian.jpg"

# Preprocessing gambar
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Prediksi
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
confidence = round(100 * np.max(prediction), 2)

# Output hasil
print("📸 Gambar:", os.path.basename(img_path))
print("🧠 Prediksi Kelas:", class_names[predicted_index])
print("✅ Confidence:", confidence, "%")
