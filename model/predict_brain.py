import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "model/brain_model.keras"
IMG_SIZE = (150, 150)

model = load_model(MODEL_PATH)


def format_percent(value):
    return f"{value * 100:.2f}%"


def get_class_label(index):
    labels = [
        "Глиома",
        "Менингиома",
        "Опухоль гипофиза",
        "Норма"
    ]
    return labels[index]


def analyze_brain_image(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]

    predicted_index = np.argmax(prediction)

    result = {
        "class_name": get_class_label(predicted_index),
        "glioma": format_percent(prediction[0]),
        "meningioma": format_percent(prediction[1]),
        "pituitary": format_percent(prediction[2]),
        "normal": format_percent(prediction[3])
    }

    return result