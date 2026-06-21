import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

IMG_SIZE = (150, 150)
BATCH_SIZE = 8
EPOCHS = 8
MODEL_PATH = "model/pneumonia_model.h5"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("Индексы классов:", train_generator.class_indices)

model = Sequential([
    Input(shape=(150, 150, 3)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

test_loss, test_accuracy = model.evaluate(test_generator)

print(f"\nТочность на тестовой выборке: {test_accuracy:.4f}")
print(f"Потери на тестовой выборке: {test_loss:.4f}")

os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)
print(f"\nМодель сохранена: {MODEL_PATH}")

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Точность обучения")
plt.plot(history.history["val_accuracy"], label="Точность валидации")
plt.title("График точности модели")
plt.xlabel("Эпоха")
plt.ylabel("Точность")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model/training_accuracy.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Потери обучения")
plt.plot(history.history["val_loss"], label="Потери валидации")
plt.title("График потерь модели")
plt.xlabel("Эпоха")
plt.ylabel("Потери")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model/training_loss.png")
plt.close()

print("Графики обучения сохранены в папке model/")