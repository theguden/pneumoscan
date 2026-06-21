import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Пути
TRAIN_DIR = "dataset_brain/train"
VAL_DIR = "dataset_brain/val"
TEST_DIR = "dataset_brain/test"

IMG_SIZE = (150, 150)
BATCH_SIZE = 16
EPOCHS = 8

# Генераторы
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Классы:", train_generator.class_indices)

# Модель
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(4, activation='softmax')  # 4 класса
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Обучение
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Оценка
test_loss, test_acc = model.evaluate(test_generator)
print(f"Точность на тесте: {test_acc:.4f}")
print(f"Потери: {test_loss:.4f}")

# Сохранение
model.save("model/brain_model.keras")
print("Модель сохранена: model/brain_model.keras")

# Графики
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Accuracy")
plt.savefig("model/brain_accuracy.png")
plt.clf()

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Loss")
plt.savefig("model/brain_loss.png")

print("Графики сохранены в model/")