import tensorflow as tf

# Charger le modèle entraîné
model = tf.keras.models.load_model("Webapp/models/vgg16_tl.h5", compile=False)

# Conversion TFLite SANS quantization agressive
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []   # ⚠️ pas de quantization
tflite_model = converter.convert()

# Sauvegarde
with open("vgg16_tl.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion VGG16 -> TFLite réussie !")
