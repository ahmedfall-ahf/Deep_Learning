from flask import Flask, render_template, request
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# ✅ Charger le modèle une seule fois au démarrage
MODEL_PATH = os.path.join(app.root_path, "../src/models/vgg16_tl.tflite")

# Vérifie que le modèle existe
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le modèle TFLite est introuvable à l'emplacement : {MODEL_PATH}")

# ✅ Préparer l’interpréteur TFLite (léger et rapide)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# ✅ Détails des tenseurs (une seule fois)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Taille d’entrée du modèle
IMG_SIZE = (128, 128)


@app.route("/", methods=["GET", "POST"])
def index():
    pred_text = None
    imgpath = None

    if request.method == "POST":
        f = request.files.get("file")
        if f and f.filename != "":
            upload_folder = os.path.join(app.root_path, "static", "uploads")
            os.makedirs(upload_folder, exist_ok=True)

            filepath = os.path.join(upload_folder, f.filename)
            f.save(filepath)

            # ✅ Prétraitement d'image optimisé
            img = load_img(filepath, target_size=IMG_SIZE)
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0).astype(np.float32)

            # ✅ Inférence TFLite
            interpreter.set_tensor(input_details[0]['index'], x)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            score = float(output_data[0][0])
            label = "Parasitized" if score < 0.5 else "Uninfected"
            pred_text = f"{label} (score={score:.3f})"

            imgpath = f"static/uploads/{f.filename}"

    return render_template("index.html", pred=pred_text, imgpath=imgpath)

print("🔹 Préchargement du modèle TFLite...")
interpreter.set_tensor(input_details[0]["index"], np.zeros((1, 128, 128, 3), dtype=np.float32))
interpreter.invoke()
print("✅ Modèle prêt à l’emploi.")

if __name__ == "__main__":
    # ✅ Ne pas activer le debug sur Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
