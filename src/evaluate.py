import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
from keras.models import load_model
from src.utils import create_generators

DATA_DIR = r"C:\Users\Hp\OneDrive\Desktop\ProjetDL\data\cell_images"
BATCH_SIZE = 32

def plot_histories(histories):
    for name, h in histories.items():
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(h['accuracy'], label='train_acc')
        plt.plot(h['val_accuracy'], label='val_acc')
        plt.title(f"Accuracy {name}")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(h['loss'], label='train_loss')
        plt.plot(h['val_loss'], label='val_loss')
        plt.title(f"Loss {name}")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    with open("../results/histories.json") as f:
        histories = json.load(f)
    plot_histories(histories)

    # Evaluation sur validation (ou test) : charger meilleur modèle et prédire
    train_gen, val_gen = create_generators(DATA_DIR, batch_size=BATCH_SIZE)
    # Exemple pour charger un modèle
    model = load_model("../models/simple_cnn.h5")
    val_gen.reset()
    preds = model.predict(val_gen)
    y_pred = (preds > 0.5).astype(int)
    y_true = val_gen.classes[:len(y_pred)]
    print(classification_report(y_true, y_pred))
