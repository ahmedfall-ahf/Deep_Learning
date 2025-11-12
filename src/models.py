from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.applications import VGG16
from keras import Input

IMG_SHAPE = (128, 128, 3)

def simple_cnn():
    model = Sequential([
        Input(IMG_SHAPE),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def deep_cnn():
    model = Sequential([
        Input(IMG_SHAPE),
        Conv2D(32,(3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32,(3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64,(3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64,(3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128,(3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model

def transfer_vgg16(freeze_base=True):
    base = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
    if freeze_base:
        for layer in base.layers:
            layer.trainable = False

    from keras.models import Model
    x = base.output
    from keras.layers import GlobalAveragePooling2D
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=preds)
    return model
