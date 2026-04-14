import tensorflow as tf
import cv2
import numpy as np

MODEL_PATH = "model/model.keras"
model = None

def load_model_once():
    global model
    if model is None:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False   # 🔥 FIX HERE
        )
    return model


def predict_image(img_path):

    model = load_model_once()

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256,256))
    img = img / 255.0
    img = img.reshape(1,256,256,1)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return {
            "label": "PNEUMONIA DETECTED",
            "confidence": round(pred*100,2)
        }
    else:
        return {
            "label": "NORMAL",
            "confidence": round((1-pred)*100,2)
        }