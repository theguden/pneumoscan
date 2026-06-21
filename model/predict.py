import os
import cv2
import numpy as np
import tensorflow as tf
import keras


from keras.src.layers.core.dense import Dense
original_dense_from_config = Dense.from_config

def patched_dense_from_config(cls, config):
    if 'quantization_config' in config:
        del config['quantization_config']
    return original_dense_from_config(config)

Dense.from_config = classmethod(patched_dense_from_config)
# =====================================================================

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "model/pneumonia_model.h5"
IMG_SIZE = (150, 150)

model = load_model(MODEL_PATH)


def format_percent(value):
    return f"{value * 100:.2f}%"


def get_class_label(class_name):
    labels = {
        "normal": "Норма",
        "viral": "Вирусная пневмония",
        "bacterial": "Бактериальная пневмония",
        "covid": "COVID-19"
    }
    return labels.get(class_name, class_name)


def is_xray_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return False

    b, g, r = cv2.split(img)

    diff_rg = np.mean(np.abs(r.astype("float") - g.astype("float")))
    diff_rb = np.mean(np.abs(r.astype("float") - b.astype("float")))
    diff_gb = np.mean(np.abs(g.astype("float") - b.astype("float")))

    color_diff = (diff_rg + diff_rb + diff_gb) / 3

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)

    if color_diff > 25:
        return False

    if brightness < 25 or brightness > 230:
        return False

    return True


def prepare_image(image_path):
    img = image.load_img(image_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def find_last_conv_layer_name():
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def make_gradcam_heatmap(img_array, predicted_index):
    last_conv_layer_name = find_last_conv_layer_name()

    if last_conv_layer_name is None:
        return None

    # Важно: явно вызываем модель, чтобы Keras построил вычислительный граф
    _ = model(img_array)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.layers[-1].output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, predicted_index]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    return heatmap.numpy()


def save_gradcam_image(image_path, heatmap):
    original_img = cv2.imread(image_path)

    if original_img is None or heatmap is None:
        return None

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_img, 0.65, heatmap_color, 0.35, 0)

    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    gradcam_filename = f"{name}_gradcam{ext}"
    gradcam_path = os.path.join("static", "uploads", gradcam_filename)

    cv2.imwrite(gradcam_path, overlay)

    return f"/static/uploads/{gradcam_filename}"


def analyze_image(image_path):
    if not is_xray_image(image_path):
        return {
            "error": "Файл загружен, но не прошел предварительную проверку. Загрузите рентгеновский снимок грудной клетки."
        }

    img_array = prepare_image(image_path)

    # прогреваем модель для Grad-CAM
    _ = model(img_array)

    prediction = model.predict(img_array, verbose=0)[0]

    result_dict = {
        "bacterial": float(prediction[0]),
        "covid": float(prediction[1]),
        "normal": float(prediction[2]),
        "viral": float(prediction[3])
    }

    predicted_class = max(result_dict, key=result_dict.get)
    max_prob = result_dict[predicted_class]

    class_order = ["bacterial", "covid", "normal", "viral"]
    predicted_index = class_order.index(predicted_class)

    heatmap = make_gradcam_heatmap(img_array, predicted_index)
    gradcam_url = save_gradcam_image(image_path, heatmap)

    if max_prob >= 0.7:
        confidence_level = "Высокая"
    elif max_prob >= 0.5:
        confidence_level = "Средняя"
    else:
        confidence_level = "Низкая"

    recommendation = ""
    if confidence_level == "Низкая":
        recommendation = "Рекомендуется повторная проверка изображения."

    return {
        "class_name": get_class_label(predicted_class),
        "confidence": confidence_level,
        "recommendation": recommendation,
        "gradcam_url": gradcam_url,

        "normal": format_percent(result_dict["normal"]),
        "viral": format_percent(result_dict["viral"]),
        "bacterial": format_percent(result_dict["bacterial"]),
        "covid": format_percent(result_dict["covid"])
    }
