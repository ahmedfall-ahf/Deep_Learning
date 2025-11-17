import os
import logging
from flask import Flask, render_template, request, abort
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deep-learning-app")

IMG_SIZE = (128, 128)
MODEL_NAME = "vgg16_tl.tflite"
MODEL_PATH = os.path.join(app.root_path, MODEL_NAME)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modèle TFLite introuvable : {MODEL_PATH}")

# Charger l’interpréteur TFLite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

logger.info(f"TFLite input : {input_details}")
logger.info(f"TFLite output : {output_details}")

@app.route("/", methods=["GET", "POST"])
def index():
    pred_text = None
    imgpath = None

    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            return render_template("index.html", pred="Aucun fichier", imgpath=None)

        upload_folder = os.path.join(app.root_path, "static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, f.filename)
        f.save(filepath)
        logger.info(f"Fichier reçu : {filepath}")

        try:
            img = load_img(filepath, target_size=IMG_SIZE)
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0).astype(np.float32)

            # TFLite inference
            interpreter.set_tensor(input_details[0]['index'], x)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            score = float(output_data[0][0])

            logger.info(f"Résultat brut TFLite : {output_data}")

            label = "Parasitized" if score < 0.5 else "Uninfected"
            pred_text = f"{label} (score={score:.3f})"
            imgpath = f"static/uploads/{f.filename}"

        except Exception as e:
            logger.exception("Erreur pendant la prédiction")
            abort(500, description=f"Erreur prédiction : {e}")

    return render_template("index.html", pred=pred_text, imgpath=imgpath)

@app.route("/_health")
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
