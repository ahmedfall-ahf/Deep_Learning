import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import create_generators
from models import simple_cnn, deep_cnn, transfer_vgg16
import matplotlib.pyplot as plt
import json

DATA_DIR = r"C:\Users\Hp\OneDrive\Desktop\ProjetDL\data\cell_images\train" # adapte le chemin si besoin
BATCH_SIZE = 32
EPOCHS = 15
IMG_SIZE = (128,128)

def compile_and_train(model, name, train_gen, val_gen, epochs=EPOCHS):
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint(f"models/{name}.h5", monitor='val_accuracy', save_best_only=True, mode='max')
    early = EarlyStopping(monitor='val_accuracy', patience=4, mode='max', restore_best_weights=True)
    hist = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[checkpoint, early])
    return hist

if __name__ == "__main__":
    train_gen, val_gen = create_generators(DATA_DIR, batch_size=BATCH_SIZE)
    models = {
        "simple_cnn": simple_cnn(),
        "deep_cnn": deep_cnn(),
        "vgg16_tl": transfer_vgg16(freeze_base=True)
    }

    histories = {}
    for name, model in models.items():
        print(f"--- Training {name} ---")
        h = compile_and_train(model, name, train_gen, val_gen, epochs=EPOCHS)
        histories[name] = h.history

    # Sauvegarder les historiques pour comparaison
    with open("results/histories.json", "w") as f:
        json.dump(histories, f)
