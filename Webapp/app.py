from flask import Flask, render_template, request
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# --- Chargement du modèle ---
MODEL_PATH = os.path.join(app.root_path, "../src/models/vgg16_tl.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

IMG_SIZE = (128, 128)


@app.route("/", methods=["GET", "POST"])
def index():
    pred_text = None
    imgpath = None

    if request.method == "POST":
        f = request.files.get("file")

        if f and f.filename != "":
            upload_folder = os.path.join(app.root_path, "static/uploads")
            os.makedirs(upload_folder, exist_ok=True)

            filepath = os.path.join(upload_folder, f.filename)
            f.save(filepath)

            # --- Prétraitement EXACT comme l'entraînement ---
            img = load_img(filepath, target_size=IMG_SIZE)
            x = img_to_array(img) / 255.0  # ⭐ OBLIGATOIRE
            x = np.expand_dims(x, axis=0)

            # --- Prédiction ---
            preds = model.predict(x)
            print("🔍 Output du modèle :", preds)

            score = float(preds[0][0])

            # 0 = Uninfected, 1 = Parasitized
            label = "Parasitized" if score < 0.5 else "Uninfected"

            pred_text = f"{label} (score={score:.3f})"
            imgpath = f"static/uploads/{f.filename}"

    return render_template("index.html", pred=pred_text, imgpath=imgpath)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
