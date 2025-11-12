from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)
import os
len(os.listdir("../data/cell_images/train/Parasitized")), len(os.listdir("../data/cell_images/train/Uninfected"))

# ✅ Utilisation d'un chemin absolu pour le modèle
MODEL_PATH = os.path.join(app.root_path, "../src/models/vgg16_tl.h5")
model = load_model(MODEL_PATH)

IMG_SIZE = (128, 128)

@app.route("/", methods=["GET", "POST"])
def index():
    pred_text = None
    imgpath = None

    if request.method == "POST":
        f = request.files.get("file")
        if f:
            # ✅ Sauvegarde correcte du fichier dans /static
            upload_folder = os.path.join(app.root_path, "static")
            os.makedirs(upload_folder, exist_ok=True)

            filename = f.filename
            filepath = os.path.join(upload_folder, filename)
            f.save(filepath)

            # ✅ Chargement et prédiction
            img = load_img(filepath, target_size=IMG_SIZE)
            x = img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            p = model.predict(x)[0][0]
            
            print("Sortie brute du modèle:", model.predict(x))
            label = "Parasitized" if p < 0.5 else "Uninfected"
            pred_text = f"{label} (score={p:.3f})"

            # ✅ Chemin relatif pour l’affichage HTML
            imgpath = f"static/{filename}"

    return render_template("index.html", pred=pred_text, imgpath=imgpath)

if __name__ == "__main__":
    app.run(debug=True)
