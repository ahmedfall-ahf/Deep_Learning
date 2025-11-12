import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

IMG_SIZE = (128, 128)

def show_sample_images(base_dir, n=4):
    import random
    classes = ['Parasitized', 'Uninfected']
    plt.figure(figsize=(10,5))
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(base_dir, cls)
        imgs = os.listdir(cls_dir)
        for j in range(n):
            p = random.choice(imgs)
            img = load_img(os.path.join(cls_dir,p), target_size=IMG_SIZE)
            plt.subplot(2, n, i*n + j + 1)
            plt.imshow(img)
            plt.title(f"{cls}")
            plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_generators(data_dir, batch_size=32, seed=42):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       validation_split=0.2,
                                       rotation_range=15,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True)
    train_gen = train_datagen.flow_from_directory(data_dir,
                                                  target_size=IMG_SIZE,
                                                  batch_size=batch_size,
                                                  class_mode='binary',
                                                  subset='training',
                                                  seed=seed)
    val_gen = train_datagen.flow_from_directory(data_dir,
                                                target_size=IMG_SIZE,
                                                batch_size=batch_size,
                                                class_mode='binary',
                                                subset='validation',
                                                seed=seed)
    # Optionnel: test generator séparé si tu as dossier test
    return train_gen, val_gen
