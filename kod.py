import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

DATASET_PATH = "Slike Auta"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")

#Postotak podjele podataka
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

#Kategorije vozila
categories = ["Big Truck", "City Car", "Multi Purpose Vehicle", "Sedan", "Sport Utility Vehicle", "Truck", "Van"]

model_path = "model_vozila_v2.keras"

#Automatski provjeri postoji li model
train_model = not os.path.exists(model_path)

#Ako model treba trenirati, prvo podijeli slike
if train_model:
    print("📂 Počinjem dijeliti slike na train/val/test...")

    for folder in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        os.makedirs(folder, exist_ok=True)

    for category in categories:
        category_path = os.path.join(DATASET_PATH, category)

        if not os.path.exists(category_path):
            print(f"⚠️ Folder {category} ne postoji, preskačem...")
            continue

        images = [img for img in os.listdir(category_path) if img.lower().endswith((".jpg", ".png", ".jpeg"))]

        if not images:
            print(f"⚠️ Nema slika u {category}, preskačem...")
            continue

        random.shuffle(images)

        train_count = int(len(images) * TRAIN_SPLIT)
        val_count = int(len(images) * VAL_SPLIT)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        for img in train_images:
            os.makedirs(os.path.join(TRAIN_PATH, category), exist_ok=True)
            shutil.copy(os.path.join(category_path, img), os.path.join(TRAIN_PATH, category, img))
        for img in val_images:
            os.makedirs(os.path.join(VAL_PATH, category), exist_ok=True)
            shutil.copy(os.path.join(category_path, img), os.path.join(VAL_PATH, category, img))
        for img in test_images:
            os.makedirs(os.path.join(TEST_PATH, category), exist_ok=True)
            shutil.copy(os.path.join(category_path, img), os.path.join(TEST_PATH, category, img))

        print(f"✅ {category}: {len(train_images)} trening, {len(val_images)} validacija, {len(test_images)} test")

    print("🚀 Podjela slika je završena!")

#Parametri slike
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_generator = datagen.flow_from_directory(TRAIN_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_generator = datagen.flow_from_directory(VAL_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

NUM_CLASSES = len(train_generator.class_indices)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers[-10:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

#Ako model ne postoji, treniraj ga
if train_model:
    print("🔄 Počinjem trenirati model...")
    EPOCHS = 10
    history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=[reduce_lr])
    model.save(model_path)
    print("✅ Model je uspješno spremljen.")

    #Vizualizacija treninga
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Trening performanse modela (Poboljšana verzija)")
    plt.show()

else:
    print("📂 Učitavam već trenirani model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model uspješno učitan.")
    except Exception as e:
        print(f"⚠️ Greška prilikom učitavanja modela: {e}")
        print("🔄 Pokrećem novo treniranje...")
        history = model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[reduce_lr])
        model.save(model_path)
        print("✅ Model je uspješno spremljen.")

#valuacija na testnim podacima
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(TEST_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

test_loss, test_acc = model.evaluate(test_generator)
print(f"✅ Testna točnost: {test_acc:.4f}")
print(f"✅ Testni gubitak: {test_loss:.4f}")

